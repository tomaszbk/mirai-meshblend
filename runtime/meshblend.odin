package meshblend

import "core:math"
import "core:mem"
import "core:slice"
import "mirai:core"
import render "mirai:engine/render"

// Blends the surfaces of two meshes where they meet, before they are lit:
// near the seam each takes the base color, metallic, normal, roughness and
// reflectance of the other, so light, shadows, GI and reflections follow the
// blended surface.
//
// GPU pipeline (Surface stage, fully deterministic, zero temporal jitter):
//   1. seed    — wave-deduplicated entity lookup into a 4-byte pixel_info
//                buffer + exact 3D-contact seam detection in 10x10 shared tile.
//   2. prepare — builds GPU-driven JFA indirect dispatches (0 work if no seams).
//   3. jfa     — bounded half-res Jump Flooding (up to 7 steps, even start so
//                output always lands in jfa_b); final step emits active 8x8 tiles.
//   4. resolve — indirect compute over active tiles only, computing full PBR
//                blends (OKLab color, metallic, facing-safe normal, roughness,
//                reflectance, rest-space noise, nested views) into RGBA32_Uint.
//   5. apply   — indirect quad draw over active tiles only into G-buffer targets.

BLEND_TILE        :: 8
MIN_PIXELS_HEIGHT :: 1080
MAX_RADIUS        :: 256
JFA_STEPS         :: 7
WORK_HEADER_WORDS :: 8 // [0..3] resolve dispatch, [4..7] apply draw

@(component,description="Blends the surfaces of this mesh and its descendants where they meet other meshes. A descendant with its own one decides for its subtree; disabled leaves it out.")
Mesh_Blend :: struct {
    width:f32 `mirai:"min=0,unit=m"`, // how far from the seam the surfaces blend
    enabled:bool,
}

MESH_BLEND_DEFAULT :: Mesh_Blend{width=0.10,enabled=true}

@(storage,setting)
Mesh_Blend_Settings :: struct {
    noise:f32 `mirai:"min=0,max=1"`,          // how much the blend breaks up along the seam
    noise_scale:f32 `mirai:"min=0.1,max=16"`, // of the noise, as a part of the width
    min_pixels:f32 `mirai:"min=0,max=64,unit=px,advanced"`, // how wide a seam stays on the screen far away, in pixels of an image 1080 high
    max_widen:f32 `mirai:"min=1,max=8,advanced"`,           // how many times its own width a seam widens at most to stay min_pixels wide
    slope:f32 `mirai:"min=1,max=8,advanced"`,               // narrows steep (~90 deg) intersections by up to this factor relative to coplanar joins
}

MESH_BLEND_SETTINGS_DEFAULT :: Mesh_Blend_Settings{noise=0.35,noise_scale=1,min_pixels=4,max_widen=3,slope=2}

// Shared cbuffer for the compute passes (48 bytes).
@(private)
Gpu_Params :: struct {
    noise,noise_scale,min_pixels,max_widen:f32,
    values:u32,
    info_offset:u32,
    tiles_max:u32,
    step:u32,
    slope:f32,
    pad:[3]u32,
}
#assert(size_of(Gpu_Params)==48)

// The settings as the passes take them: clamped within valid ranges, defaults for non-finite values.
@(private)
params_of :: proc(settings:Mesh_Blend_Settings) -> Gpu_Params {
    pick :: proc(value,fallback,low,high:f32) -> f32 {
        return clamp(fallback if math.is_nan(value) || math.is_inf(value) else value,low,high)
    }
    d:=MESH_BLEND_SETTINGS_DEFAULT
    return {
        noise=pick(settings.noise,d.noise,0,1),
        noise_scale=pick(settings.noise_scale,d.noise_scale,0.1,16),
        min_pixels=pick(settings.min_pixels,d.min_pixels,0,64),
        max_widen=pick(settings.max_widen,d.max_widen,1,8),
        slope=pick(settings.slope,d.slope,1,8),
    }
}

// Compact 4-byte per-entity blend value indexed by entity_index:
//   bits  0..15: dense group id (1..65535; 0 = does not participate)
//   bits 16..31: positive IEEE-754 f16 width in metres (0 = does not participate)
@(private)
Blend_Value :: distinct u32

@(private)
pack_blend_value :: proc "contextless" (width:f32,group:u16) -> Blend_Value {
    if group==0 || !(width>0) || math.is_nan(width) || math.is_inf(width) {return 0}
    w:=clamp(width,1e-4,65504.0)
    h:=transmute(u16)f16(w)
    if h==0 {return 0}
    return Blend_Value(u32(group)|(u32(h)<<16))
}

@(private)
blend_value_group :: proc "contextless" (v:Blend_Value) -> u16 {return u16(u32(v)&0xffff)}

@(private)
blend_value_width :: proc "contextless" (v:Blend_Value) -> f32 {return f32(transmute(f16)u16(u32(v)>>16))}

VALUES_BUFFER :: "mesh blend values"
ARGS_BUFFER   :: "mesh blend args"
WORK_BUFFER   :: "mesh blend work"
ARGS_BYTES    :: u64((4+JFA_STEPS*4)*size_of(u32)) // seam_count, max_half_radius, pad[2], jfa_dispatch[7]
VALUES_UPLOAD :: 1<<20 // bytes of values uploaded per frame at most

// Per-world CPU/GPU bookkeeping. Owns no GPU handles (the renderer owns pass_buffers/pass_textures).
@(storage,host,by_world,free="mesh_blend_runtime_free")
Mesh_Blend_Runtime :: struct {
    values:[dynamic]Blend_Value,
    scratch:[dynamic]Blend_Value,
    author_groups:[dynamic]u16, // by author entity_index -> dense 16-bit group id (1..65535)
    seen:[5]u64,
    world:core.World_Id,
    active:int,
    values_capacity:u64,
    work_capacity:u64,
    dirty_lo,dirty_hi:int,
    uploaded:bool, // true once the GPU buffer has received at least one full upload
    pipelines:[4]render.Pipeline, // seed, prepare, jfa, resolve
    params:Gpu_Params,
}

mesh_blend_runtime_free :: proc(state:^Mesh_Blend_Runtime) {
    delete(state.values)
    delete(state.scratch)
    delete(state.author_groups)
    state^={}
}

@(step="release")
mesh_blend_release :: proc(state:^Mesh_Blend_Runtime,id:core.World_Id,renderer:^render.Renderer) {
    if state!=nil && renderer!=nil {
        render.buffer_release(renderer,id,VALUES_BUFFER)
        render.buffer_release(renderer,id,ARGS_BUFFER)
        render.buffer_release(renderer,id,WORK_BUFFER)
    }
}

// Rebuilds the per-entity packed values table top-down from Mesh_Blend roots alone
// (O(participating subtrees), independent of total scene instance/batch count).
// Computes the tight dirty range [dirty_lo, dirty_hi) against the previous values.
@(private)
values_build :: proc(
    state:^Mesh_Blend_Runtime,
    blends:^core.Table(Mesh_Blend,$A),
    hierarchy:core.Ref(core.Hierarchy),
    ids:^core.Table(core.Id,$C),
) -> (active:int) {
    clear(&state.scratch)
    slice.zero(state.author_groups[:])
    next_group:u16=1

    assign_group :: proc(author_groups:^[dynamic]u16,next_group:^u16,author:core.Entity) -> u16 {
        idx:=core.entity_index(author)
        if idx>=len(author_groups) {
            if resize(author_groups,idx+1)!=nil {return 0}
        }
        if g:=author_groups[idx];g!=0 {return g}
        if next_group^==0 {return max(u16)} // saturated at 65535 distinct groups
        g:=next_group^
        next_group^=g+1 if g<max(u16) else 0
        author_groups[idx]=g
        return g
    }

    set_value :: proc(out:^[dynamic]Blend_Value,entity:core.Entity,val:Blend_Value) -> int {
        idx:=core.entity_index(entity)
        if idx>=len(out) {
            if resize(out,idx+1)!=nil {return 0}
        }
        out[idx]=val
        return 1 if val!=0 else 0
    }

    author_stack:[core.HIERARCHY_DEPTH+2]core.Entity
    node_stack:[core.HIERARCHY_DEPTH+2]core.Entity

    for row in 0..<core.count(blends) {
        marker:=core.at(blends,row)
        if !marker.enabled || !(marker.width>0) || math.is_nan(marker.width) || math.is_inf(marker.width) {continue}
        root:=core.entity_at(blends,row)
        root_group:=assign_group(&state.author_groups,&next_group,root)
        active+=set_value(&state.scratch,root,pack_blend_value(marker.width,root_group))
        if hierarchy.ptr==nil {continue}

        // Top-down DFS over root's subtree, stopping at any descendant with its own Mesh_Blend.
        depth:=0
        author_stack[0]=root
        node_stack[0]=core.first_child(hierarchy,root)
        for depth>=0 {
            at:=node_stack[depth]
            if at==core.NONE {depth-=1;continue}
            node_stack[depth]=core.next_sibling(hierarchy,at)
            if core.has(blends,at) {continue}
            author:=at if core.has(ids,at) else author_stack[depth]
            group:=assign_group(&state.author_groups,&next_group,author)
            active+=set_value(&state.scratch,at,pack_blend_value(marker.width,group))
            if child:=core.first_child(hierarchy,at);child!=core.NONE && depth+1<len(node_stack) {
                depth+=1
                author_stack[depth]=author
                node_stack[depth]=child
            }
        }
    }

    // Diff state.scratch against state.values to find the tight dirty range [lo, hi).
    old_n:=len(state.values)
    new_n:=len(state.scratch)
    common:=min(old_n,new_n)
    lo:=max(old_n,new_n)
    hi:=0
    for i in 0..<common {
        if state.values[i]!=state.scratch[i] {
            if i<lo {lo=i}
            hi=i+1
        }
    }
    if new_n>old_n {
        if old_n<lo {lo=old_n}
        hi=new_n
    } else if old_n>new_n {
        // Zero tail in-place if we had non-zero values past new_n so GPU reads 0 if params.values shrinks.
        // Because params.values is updated to new_n, GPU bounds-checks object < params.values,
        // so shrinking does not require uploading the truncated tail!
    }

    // Swap scratch into values.
    state.values,state.scratch=state.scratch,state.values
    if lo<hi {
        if state.dirty_lo<state.dirty_hi {
            state.dirty_lo=min(state.dirty_lo,lo)
            state.dirty_hi=max(state.dirty_hi,hi)
        } else {
            state.dirty_lo=lo
            state.dirty_hi=hi
        }
    }
    if state.dirty_hi>len(state.values) {state.dirty_hi=len(state.values)}
    if state.dirty_lo>state.dirty_hi {state.dirty_lo=state.dirty_hi}
    return active
}

@(step="render",loops="game editor",after="render_sync")
mesh_blend_render :: proc(
    state:^Mesh_Blend_Runtime,synced:core.Ref(render.Render_Sync),renderer:^render.Renderer,allocator:mem.Allocator,
    blends:^core.Table(Mesh_Blend,.Read),hierarchy:core.Ref(core.Hierarchy),ids:^core.Table(core.Id,.Read),settings:Mesh_Blend_Settings,
) {
    if state==nil || renderer==nil || synced.ptr==nil {return}
    sync:=synced.ptr
    h_version:=hierarchy.ptr.version if hierarchy.ptr!=nil else 0
    world_key:=render.world_key(sync.world)
    seen:=[5]u64{core.layout(blends),core.version(blends),h_version,core.layout(ids),world_key}
    if seen!=state.seen || sync.world!=state.world {
        world_changed:=sync.world!=state.world
        state.seen,state.world=seen,sync.world
        if state.values.allocator.procedure==nil {
            state.values.allocator=allocator
            state.scratch.allocator=allocator
            state.author_groups.allocator=allocator
        }
        if world_changed {
            clear(&state.values)
            state.dirty_lo,state.dirty_hi=0,0
            state.uploaded=false
        }
        state.active=values_build(state,blends,hierarchy,ids)
        need:=u64(max(len(state.values),64)*size_of(Blend_Value))
        for state.values_capacity<need {state.values_capacity=max(state.values_capacity*2,need)}
    }
    if state.active==0 {return}
    state.pipelines={
        render.stage_pipeline(renderer,{stage=.Surface,shader="mesh_blend_seed",compute=true,name="mesh blend seed"}),
        render.stage_pipeline(renderer,{stage=.Surface,shader="mesh_blend_prepare",compute=true,name="mesh blend prepare"}),
        render.stage_pipeline(renderer,{stage=.Surface,shader="mesh_blend_jfa",compute=true,name="mesh blend jfa"}),
        render.stage_pipeline(renderer,{stage=.Surface,shader="mesh_blend_resolve",compute=true,name="mesh blend resolve"}),
    }
    for pipe in state.pipelines {if pipe==0 {return}}
    state.params=params_of(settings)
    state.params.values=u32(len(state.values))
    render.frame_pass(renderer,.Surface,mesh_blend_record,state)
}

@(private)
mesh_blend_record :: proc(pass:^render.Pass,user:rawptr) {
    state:=cast(^Mesh_Blend_Runtime)user
    if pass.world!=render.world_key(state.world) {return}
    if pass.width==0 || pass.height==0 {return}

    // 1. Values buffer (per world, dirty-range upload).
    values:=render.Buffer_Desc{name=VALUES_BUFFER,size=state.values_capacity}
    ok,made:=render.pass_buffer(pass,2,values)
    if !ok {return}
    if made {
        state.dirty_lo=0
        state.dirty_hi=len(state.values)
        state.uploaded=false
    }
    if state.dirty_lo<state.dirty_hi {
        chunk:=min(state.dirty_hi-state.dirty_lo,VALUES_UPLOAD/size_of(Blend_Value))
        if chunk>0 && render.pass_buffer_write(pass,values,u64(state.dirty_lo*size_of(Blend_Value)),&state.values[state.dirty_lo],chunk*size_of(Blend_Value)) {
            state.dirty_lo+=chunk
        }
        if state.dirty_lo>=state.dirty_hi {
            state.dirty_lo,state.dirty_hi=0,0
            state.uploaded=true
        } else if !state.uploaded {
            // Only skip rendering on the very first upload before the GPU buffer holds valid data.
            return
        }
    }

    // 2. Args & work buffers (per world, power-of-two growth).
    tiles_x:=(pass.width+BLEND_TILE-1)/BLEND_TILE
    tiles_y:=(pass.height+BLEND_TILE-1)/BLEND_TILE
    tiles_max:=tiles_x*tiles_y
    info_offset:=u32(WORK_HEADER_WORDS)+tiles_max
    work_words:=u64(info_offset)+u64(pass.width)*u64(pass.height)
    work_need:=work_words*size_of(u32)
    for state.work_capacity<work_need {state.work_capacity=max(state.work_capacity*2,max(work_need,u64(4096)))}

    args:=render.Buffer_Desc{name=ARGS_BUFFER,size=ARGS_BYTES}
    work:=render.Buffer_Desc{name=WORK_BUFFER,size=state.work_capacity}
    if a_ok,_:=render.pass_buffer(pass,3,args);!a_ok {return}
    if w_ok,_:=render.pass_buffer(pass,4,work);!w_ok {return}
    if !render.pass_buffer_fill(pass,args,0,16,0) {return}
    if !render.pass_buffer_fill(pass,work,0,u64(WORK_HEADER_WORDS*size_of(u32)),0) {return}

    // 3. View textures (3 slots total: jfa_a, jfa_b at half-res, surface at full-res).
    hw:=(pass.width+1)/2
    hh:=(pass.height+1)/2
    jfa_a:=render.Pass_Texture_Desc{name="mesh blend jfa a",format=.RGBA32_Uint,width=hw,height=hh}
    jfa_b:=render.Pass_Texture_Desc{name="mesh blend jfa b",format=.RGBA32_Uint,width=hw,height=hh}
    surface:=render.Pass_Texture_Desc{name="mesh blend surface",format=.RGBA32_Uint,width=pass.width,height=pass.height}

    // 4. Target-dependent apply pipeline (Albedo, Normal, plus Reflectance when active).
    available:=render.pass_surface_targets(pass)
    reflectance:=.Reflectance in available
    surface_targets:=render.Surface_Targets{.Albedo,.Normal}
    if reflectance {surface_targets+={.Reflectance}}
    apply_pipe:=render.pass_stage_pipeline(pass,{
        stage=.Surface,shader="mesh_blend_apply",targets=surface_targets,name="mesh blend apply",
        fragment="mesh_blend_apply_reflectance" if reflectance else "",
    })
    if apply_pipe==0 {return}

    params:=state.params
    params.info_offset=info_offset
    params.tiles_max=tiles_max

    // Pass 1: exact 3D-contact seam detection + pixel_info build.
    render.pass_timestamp(pass,"mesh blend seed")
    render.pass_pipeline(pass,state.pipelines[0])
    render.pass_surface_read(pass,0,.Linear_Depth)
    render.pass_surface_read(pass,1,.Instance)
    render.pass_buffer(pass,2,values)
    render.pass_buffer(pass,3,args)
    render.pass_buffer(pass,4,work)
    if t_ok,_:=render.pass_texture(pass,5,jfa_a);!t_ok {return}
    if !render.pass_params(pass,&params,size_of(params)) {return}
    render.pass_dispatch(pass,tiles_x,tiles_y)

    // Pass 2: build GPU-driven JFA indirect dispatch arguments (0 work if seam_count == 0).
    render.pass_timestamp(pass,"mesh blend jfa")
    render.pass_pipeline(pass,state.pipelines[1])
    render.pass_buffer(pass,0,args)
    render.pass_buffer(pass,1,work)
    render.pass_dispatch(pass,1)

    // Pass 3: bounded half-res Jump Flooding (steps 64..1, even start -> final result always in jfa_b).
    render.pass_pipeline(pass,state.pipelines[2])
    render.pass_surface_read(pass,0,.Linear_Depth)
    render.pass_buffer(pass,3,work)
    for k in 0..<JFA_STEPS {
        src:=jfa_a if (k&1)==0 else jfa_b
        dst:=jfa_b if (k&1)==0 else jfa_a
        if !render.pass_texture_sample(pass,1,src) {return}
        if t_ok,_:=render.pass_texture(pass,2,dst);!t_ok {return}
        params.step=u32(1)<<u32(JFA_STEPS-1-k)
        if !render.pass_params(pass,&params,size_of(params)) {return}
        render.pass_dispatch_indirect(pass,args,u64((4+k*4)*size_of(u32)))
    }

    // Pass 4: sparse full-PBR blend over active 8x8 tiles only.
    render.pass_timestamp(pass,"mesh blend resolve")
    render.pass_pipeline(pass,state.pipelines[3])
    render.pass_surface_read(pass,0,.Linear_Depth)
    render.pass_surface_read(pass,1,.Albedo)
    render.pass_surface_read(pass,2,.Normal)
    if !render.pass_texture_sample(pass,3,jfa_b) {return}
    render.pass_buffer(pass,4,work)
    if t_ok,_:=render.pass_texture(pass,5,surface);!t_ok {return}
    params.step=0
    if !render.pass_params(pass,&params,size_of(params)) {return}
    render.pass_dispatch_indirect(pass,work,0)

    // Pass 5: sparse indirect quad apply over active 8x8 tiles only.
    if !render.pass_texture_sample(pass,0,surface) {return}
    render.pass_buffer(pass,1,work)
    render.pass_surface_begin(pass,surface_targets,"mesh blend apply")
    render.pass_pipeline(pass,apply_pipe)
    render.pass_draw_indirect(pass,work,u64(4*size_of(u32)))
    render.pass_end(pass)
}
