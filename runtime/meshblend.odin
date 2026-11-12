package meshblend

import "core:math"
import "core:mem"
import "mirai:core"
import render "mirai:engine/render"

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

// mesh_blend.comp: the settings, and how many values the buffer of the world holds.
@(private)
Gpu_Params :: struct {noise,noise_scale,min_pixels,max_widen:f32,values:u32,pad:[3]u32}

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

// What the search reads of the entity an instance or a batch is drawn for,
// by its index: how far from the seam it blends (0: it takes no part) and
// its group, the index of the entity that authored it.
@(private)
Blend_Value :: struct {width:f32,group:u32}

VALUES_BUFFER :: "mesh blend values"
VALUES_UPLOAD :: 1<<20 // bytes of the values a frame sends to the GPU at most

// The values of a world, by entity index, those of the entities the renderer
// draws in it and the tables they came from; the buffer that holds them on
// the GPU and how many it holds as they are.
@(storage,host,by_world,free="mesh_blend_runtime_free")
Mesh_Blend_Runtime :: struct {
    values:[dynamic]Blend_Value,
    seen:[8]u64,
    world:core.World_Id,
    active:int,
    capacity:u64, // bytes of the buffer
    written:int,
    pipelines:[2]render.Pipeline, // of this frame: the search and the apply
    params:Gpu_Params,
}

mesh_blend_runtime_free :: proc(state:^Mesh_Blend_Runtime) {
    delete(state.values)
    state^={}
}

// A world that goes takes its buffer with it.
@(step="release")
mesh_blend_release :: proc(state:^Mesh_Blend_Runtime,id:core.World_Id,renderer:^render.Renderer) {
    if state!=nil && renderer!=nil {render.buffer_release(renderer,id,VALUES_BUFFER)}
}

@(private)
owner_of :: proc(blends:^core.Table(Mesh_Blend,$A),parents:^core.Table(core.Parent,$B),ids:^core.Table(core.Id,$C),entity:core.Entity) -> (owner,group:core.Entity) {
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

// The values of an entity under a Mesh_Blend: the group is the entity that
// authored it, so the pieces of one model, the chunks of one terrain or the
// instances of a batch do not blend with each other.
@(private)
value_of :: proc(marker:Mesh_Blend,group:core.Entity) -> (value:Blend_Value) {
    value.group=u32(core.entity_index(group))
    if marker.enabled && marker.width>0 && !math.is_nan(marker.width) && !math.is_inf(marker.width) {value.width=marker.width}
    return
}

// The values of an entity: those of the Mesh_Blend it is under, with the
// group of its author. False when no Mesh_Blend is over it.
@(private)
entity_value :: proc(blends:^core.Table(Mesh_Blend,$A),parents:^core.Table(core.Parent,$B),ids:^core.Table(core.Id,$C),entity:core.Entity) -> (value:Blend_Value,under:bool) {
    if core.count(blends)==0 {return}
    owner,group:=owner_of(blends,parents,ids,entity)
    if owner==core.NONE {return}
    return value_of(core.get(blends,owner) or_else {},group),true
}

@(step="render",loops="game editor",after="render_sync")
mesh_blend_render :: proc(
    state:^Mesh_Blend_Runtime,synced:core.Ref(render.Render_Sync),renderer:^render.Renderer,allocator:mem.Allocator,
    blends:^core.Table(Mesh_Blend,.Read),parents:^core.Table(core.Parent,.Read),ids:^core.Table(core.Id,.Read),settings:Mesh_Blend_Settings,
) {
    if state==nil || renderer==nil || synced.ptr==nil {return}
    sync:=synced.ptr
    seen:=[8]u64{core.layout(blends),core.version(blends),core.layout(parents),core.version(parents),core.layout(ids),
        render.instance_revision(renderer),u64(render.instance_count(renderer)),render.batches_revision(renderer)}
    if seen!=state.seen || sync.world!=state.world {
        state.seen,state.world=seen,sync.world
        if state.values.allocator.procedure==nil {state.values.allocator=allocator}
        state.active=values_give(state,renderer,blends,parents,ids)
        state.written=0
        // The buffer grows by powers of two, so that it is seldom made again.
        need:=u64(max(len(state.values),64)*size_of(Blend_Value))
        for state.capacity<need {state.capacity=max(state.capacity*2,need)}
    }
    if state.active==0 {return}
    state.pipelines={
        render.stage_pipeline(renderer,{stage=.Surface,shader="mesh_blend",compute=true,name="mesh blend"}),
        render.stage_pipeline(renderer,{stage=.Surface,shader="mesh_blend_apply",targets={.Albedo,.Normal},blend=.Alpha_Color,name="mesh blend apply"}),
    }
    if state.pipelines[0]==0 || state.pipelines[1]==0 {return}
    state.params=params_of(settings)
    state.params.values=u32(len(state.values))
    render.frame_pass(renderer,.Surface,mesh_blend_record,state)
}

// The values of every entity the renderer draws an instance or a batch for
// in the world, by entity index; the others take no part. How many do.
@(private)
values_give :: proc(state:^Mesh_Blend_Runtime,renderer:^render.Renderer,blends:^core.Table(Mesh_Blend,$A),parents:^core.Table(core.Parent,$B),ids:^core.Table(core.Id,$C)) -> (active:int) {
    clear(&state.values)
    if core.count(blends)==0 {return}
    give :: proc(values:^[dynamic]Blend_Value,value:Blend_Value,entity:core.Entity) -> int {
        at:=core.entity_index(entity)
        if at>=len(values) && resize(values,at+1)!=nil {return 0}
        values[at]=value
        return 1 if value.width>0 else 0
    }
    for i in 0..<render.instance_count(renderer) {
        entity:=core.Entity(render.instance_object(renderer,i))
        if value,under:=entity_value(blends,parents,ids,entity);under {active+=give(&state.values,value,entity)}
    }
    key:=render.world_key(state.world)
    for index in 0..<render.batch_records(renderer) {
        info,ok:=render.batch_info(renderer,index)
        if !ok || info.world!=key {continue}
        if value,under:=entity_value(blends,parents,ids,core.Entity(info.object_id));under {active+=give(&state.values,value,core.Entity(info.object_id))}
    }
    return
}

@(private)
mesh_blend_record :: proc(pass:^render.Pass,user:rawptr) {
    state:=cast(^Mesh_Blend_Runtime)user
    if pass.world!=render.world_key(state.world) {return}
    values:=render.Buffer_Desc{name=VALUES_BUFFER,size=state.capacity}
    ok,made:=render.pass_buffer(pass,5,values)
    if !ok {return}
    if made {state.written=0}
    if count:=len(state.values);state.written<count {
        n:=min(count-state.written,VALUES_UPLOAD/size_of(Blend_Value))
        if render.pass_buffer_write(pass,values,u64(state.written*size_of(Blend_Value)),&state.values[state.written],n*size_of(Blend_Value)) {state.written+=n}
        if state.written<count {return}
    }
    // Per pixel: the blended base color and weight, and the blended normal and roughness.
    blended:=render.Pass_Texture_Desc{name="mesh blend",format=.RG32_Uint,width=pass.width,height=pass.height}
    render.pass_timestamp(pass,"mesh blend")
    render.pass_pipeline(pass,state.pipelines[0])
    render.pass_surface_read(pass,0,.Linear_Depth)
    render.pass_surface_read(pass,1,.Albedo)
    render.pass_surface_read(pass,2,.Normal)
    render.pass_surface_read(pass,3,.Instance)
    if ok,_:=render.pass_texture(pass,4,blended);!ok {return}
    render.pass_buffer(pass,5,values)
    if !render.pass_params(pass,&state.params,size_of(state.params)) {return}
    render.pass_dispatch(pass,(pass.width+7)/8,(pass.height+7)/8)
    if !render.pass_texture_sample(pass,0,blended) {return}
    render.pass_surface_begin(pass,{.Albedo,.Normal},"mesh blend apply")
    render.pass_pipeline(pass,state.pipelines[1])
    render.pass_draw(pass,3)
    render.pass_end(pass)
}
