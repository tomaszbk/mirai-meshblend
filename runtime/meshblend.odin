package meshblend

import "core:math"
import "core:mem"
import "mirai:core"
import render "mirai:engine/render"
import "mirai:host"

// Blends the surfaces of two meshes where they meet, before they are lit:
// near the seam each takes the base color, the normal and the roughness of
// the other, so light, shadows and reflections follow the blended surface.
// A pass over the G-buffer finds the seam from the instance that drew each
// pixel (the instance ids the shaders of the plugin ask for).

@(component,description="Blends the surfaces of this mesh and its descendants where they meet other meshes. A descendant with its own one decides for its subtree; disabled leaves it out.")
Mesh_Blend :: struct {
    width:f32 `mirai:"min=0,unit=m"`, // how far from the seam the surfaces blend
    enabled:bool,
}

MESH_BLEND_DEFAULT :: Mesh_Blend{width=0.15,enabled=true}

@(storage,setting)
Mesh_Blend_Settings :: struct {
    noise:f32 `mirai:"min=0,max=1"`,          // how much the blend breaks up along the seam
    noise_scale:f32 `mirai:"min=0.1,max=16"`, // of the noise, as a part of the width
    min_pixels:f32 `mirai:"min=0,max=64,unit=px,advanced"`, // how wide a seam stays on the screen far away, in pixels of an image 1080 high
    max_widen:f32 `mirai:"min=1,max=8,advanced"`,           // how many times its own width a seam widens at most to stay min_pixels wide
}

MESH_BLEND_SETTINGS_DEFAULT :: Mesh_Blend_Settings{noise=0.25,noise_scale=1,min_pixels=4,max_widen=3}

@(private)
Gpu_Params :: struct {noise,noise_scale,min_pixels,max_widen:f32}

// The settings as the search takes them: within their ranges, the defaults
// for what is not a number.
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
    }
}

// What the step last gave the renderer: the values of each instance of
// Render_Sync.instances (width, group low and high bits, participates), the
// batches of the entities under a Mesh_Blend (the foliage of an entity, its
// object id) and the versions they came from.
@(storage,host,free="mesh_blend_runtime_free")
Mesh_Blend_Runtime :: struct {
    values:[dynamic][4]f32,
    batches:[dynamic]Given_Batch, // those it gave values, in the order of their records
    scratch:[dynamic]Given_Batch,
    allocator:mem.Allocator,
    seen:[8]u64,
    world:core.World_Id,
    active:int,
    pipelines:[2]render.Pipeline, // of this frame: the search and the apply
    params:Gpu_Params,
}

Given_Batch :: struct {record:int,handle:render.Batch_Handle}

mesh_blend_runtime_free :: proc(state:^Mesh_Blend_Runtime) {
    delete(state.values);delete(state.batches);delete(state.scratch)
    state^={}
}

@(private)
owner_of :: proc(blends:core.Read(Mesh_Blend),parents:core.Read(core.Parent),ids:core.Read(core.Id),entity:core.Entity) -> (owner,group:core.Entity) {
    at:=entity
    author:=core.NONE
    for _ in 0..=core.count(parents) {
        if author==core.NONE && core.has(ids,at) {author=at}
        if core.has(blends,at) {return at,author if author!=core.NONE else at}
        parent,known:=core.get(parents,at)
        if !known || parent.entity==core.NONE {return}
        at=parent.entity
    }
    return
}

// The values of an instance: the group is the entity that authored it, so
// the pieces of one model or the chunks of one terrain do not blend with
// each other.
@(private)
value_of :: proc(marker:Mesh_Blend,group:core.Entity) -> (value:[4]f32) {
    identity:=u32(core.entity_index(group))
    value.y,value.z=f32(identity&0xffff),f32(identity>>16)
    if marker.enabled && marker.width>0 && !math.is_nan(marker.width) && !math.is_inf(marker.width) {value.x,value.w=marker.width,1}
    return
}

@(step="render",loops="game editor",after="render_sync")
mesh_blend_render :: proc(
    state:^Mesh_Blend_Runtime,synced:core.Ref(render.Render_Sync),app:^host.App,
    blends:core.Read(Mesh_Blend),parents:core.Read(core.Parent),ids:core.Read(core.Id),settings:Mesh_Blend_Settings,
) {
    if app.renderer==nil {return}
    sync:=synced.ptr
    seen:=[8]u64{core.layout(blends),core.version(blends),core.layout(parents),core.version(parents),core.layout(ids),sync.stats.scene_sets,sync.stats.instance_updates,render.batches_revision(app)}
    count:=len(sync.instances)
    if seen!=state.seen || sync.world!=state.world || count!=len(state.values) {
        if state.allocator.procedure==nil {
            state.allocator=app.allocator
            state.values=make([dynamic][4]f32,state.allocator)
            state.batches=make([dynamic]Given_Batch,state.allocator)
            state.scratch=make([dynamic]Given_Batch,state.allocator)
        }
        state.seen,state.world=seen,sync.world
        if resize(&state.values,count)!=nil {clear(&state.values)}
        state.active=0
        for &value,i in state.values {
            value={}
            if core.count(blends)>0 {
                if owner,group:=owner_of(blends,parents,ids,core.Entity(sync.instances[i].object_id));owner!=core.NONE {
                    value=value_of(core.get(blends,owner) or_else {},group)
                }
            }
            if value.w>0 {state.active+=1}
            render.instance_values_set(app,i,value)
        }
        state.active+=batches_blend(state,app,blends,parents,ids,sync.world)
    }
    if state.active==0 {return}
    state.pipelines={
        render.stage_pipeline(app,{stage=.Surface,shader="mesh_blend",compute=true,name="mesh blend"}),
        render.stage_pipeline(app,{stage=.Surface,shader="mesh_blend_apply",targets={.Albedo,.Normal},blend=.Alpha_Color,name="mesh blend apply"}),
    }
    if state.pipelines[0]==0 || state.pipelines[1]==0 {return}
    state.params=params_of(settings)
    render.frame_pass(app,.Surface,mesh_blend_record,state)
}

@(private)
mesh_blend_record :: proc(pass:^render.Pass,user:rawptr) {
    state:=cast(^Mesh_Blend_Runtime)user
    // Per pixel: the blended base color and weight, and the blended normal and roughness.
    blended:=render.Pass_Texture_Desc{name="mesh blend",format=.RG32_Uint,width=pass.width,height=pass.height}
    render.pass_timestamp(pass,"mesh blend")
    render.pass_pipeline(pass,state.pipelines[0])
    render.pass_surface_read(pass,0,.Linear_Depth)
    render.pass_surface_read(pass,1,.Albedo)
    render.pass_surface_read(pass,2,.Normal)
    render.pass_surface_read(pass,3,.Instance)
    if ok,_:=render.pass_texture(pass,4,blended);!ok {return}
    if !render.pass_params(pass,&state.params,size_of(state.params)) {return}
    render.pass_dispatch(pass,(pass.width+7)/8,(pass.height+7)/8)
    if !render.pass_texture_sample(pass,0,blended) {return}
    render.pass_surface_begin(pass,{.Albedo,.Normal},"mesh blend apply")
    render.pass_pipeline(pass,state.pipelines[1])
    render.pass_draw(pass,3)
    render.pass_end(pass)
}

// The batches of the world whose entity is under a Mesh_Blend take its
// values, with the group of that entity: the instances of a batch, or of
// the batches of one entity, do not blend with each other. Those it gave
// values before and are no longer under one go back to none. Batches of
// other owners keep the values their owner gave them. How many take part.
@(private)
batches_blend :: proc(state:^Mesh_Blend_Runtime,app:^host.App,blends:core.Read(Mesh_Blend),parents:core.Read(core.Parent),ids:core.Read(core.Id),world:core.World_Id) -> (active:int) {
    key:=render.world_key(world)
    clear(&state.scratch)
    // Both go through the records in order: those given before are walked once.
    at:=0
    for index in 0..<render.batch_records(app) {
        info,ok:=render.batch_info(app,index)
        if !ok || info.world!=key {continue}
        value,under:=batch_value(blends,parents,ids,core.Entity(info.object_id))
        for at<len(state.batches) && state.batches[at].record<index {at+=1}
        given:=at<len(state.batches) && state.batches[at].record==index && state.batches[at].handle==info.handle
        if !under && !given {continue}
        if info.values!=value {render.batch_values_set(app,info.handle,value)}
        if under {append(&state.scratch,Given_Batch{index,info.handle})}
        if value.w>0 {active+=1}
    }
    clear(&state.batches)
    append(&state.batches,..state.scratch[:])
    return
}

// The values of the batches of an entity: those of the Mesh_Blend it is
// under, with the group of its author (the Foliage entity itself). False
// when no Mesh_Blend is over it.
@(private)
batch_value :: proc(blends:core.Read(Mesh_Blend),parents:core.Read(core.Parent),ids:core.Read(core.Id),entity:core.Entity) -> (value:[4]f32,under:bool) {
    if core.count(blends)==0 {return}
    owner,group:=owner_of(blends,parents,ids,entity)
    if owner==core.NONE {return}
    return value_of(core.get(blends,owner) or_else {},group),true
}
