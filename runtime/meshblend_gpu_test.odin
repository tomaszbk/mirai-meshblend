package meshblend

import "core:c"
import "core:fmt"
import "core:math/linalg"
import "core:mem"
import "core:os"
import "core:strings"
import "core:testing"
import "mirai:core"
import render "mirai:engine/render"
import "mirai:host"
import gpu "mirai:render"
import "mirai:rhi"
import "mirai:std/base"
import mm "mirai:std/math"
import stbi "vendor:stb/image"

@(test)
mesh_blend_gpu_blends_pbr_seams_rejects_depth_gaps_and_is_allocation_free :: proc(t:^testing.T) {
    shader_dir,found:=os.lookup_env("MIRAI_MESHBLEND_SHADERS",context.temp_allocator)
    if !found || shader_dir=="" {return}
    shaders,problem:=gpu.shader_assets_load(shader_dir)
    if !testing.expect(t,problem=="",problem) {return}
    device,device_err:=rhi.device_create({app_name="meshblend-test",validation=true,allow_hardware_ray_tracing=true})
    if !testing.expect(t,device!=nil,device_err) {gpu.shader_assets_destroy(&shaders);return}
    defer rhi.device_destroy(device)

    tracker:mem.Tracking_Allocator
    mem.tracking_allocator_init(&tracker,context.allocator)
    defer mem.tracking_allocator_destroy(&tracker)
    allocator:=mem.tracking_allocator(&tracker)

    r:=new(gpu.Renderer)
    defer free(r)
    defer gpu.renderer_destroy(r)
    settings:=gpu.DEFAULT_SETTINGS
    settings.taa=false
    if !testing.expect(t,gpu.renderer_init(r,{
        device=device,width=256,height=256,settings=settings,shaders=&shaders,allocator=allocator,
    }),r.error) {return}

    tmp_root,tmp_err:=os.make_directory_temp("","mirai-meshblend-gpu-*",context.temp_allocator)
    if !testing.expect(t,tmp_err==nil) {return}
    defer base.remove_all(tmp_root)
    assets:host.Assets
    if !testing.expect(t,host.assets_init(&assets,{
        root=tmp_root,renderer=r,
    })) {return}
    defer host.assets_destroy(&assets)
    cube:=host.assets_shape(&assets,.Cube)
    sphere:=host.assets_shape(&assets,.Sphere)
    if !testing.expect(t,cube!=nil && sphere!=nil) {return}

    // Build ECS world with floor (dielectric green), sunk sphere (metallic red/gold), and floating box (depth gap).
    entities:core.Entities;core.entities_init(&entities);defer core.entities_destroy(&entities)
    blends:core.Table(Mesh_Blend);core.table_init(&blends,"Mesh_Blend");defer core.table_destroy(&blends)
    parents:core.Table(core.Parent);core.table_init(&parents,"Parent");defer core.table_destroy(&parents)
    ids:core.Table(core.Id);core.table_init(&ids,"Id");defer core.table_destroy(&ids)
    hierarchy:core.Hierarchy;defer {
        delete(hierarchy.parent);delete(hierarchy.first_child);delete(hierarchy.last_child)
        delete(hierarchy.next_sibling);delete(hierarchy.previous_sibling);delete(hierarchy.cut)
        delete(hierarchy.moved.items)
    }

    e_floor:=core.create(&entities)
    e_sphere:=core.create(&entities)
    e_float:=core.create(&entities)
    core.add(&ids,e_floor,core.Id{value=1})
    core.add(&ids,e_sphere,core.Id{value=2})
    core.add(&ids,e_float,core.Id{value=3})
    core.add(&blends,e_floor,Mesh_Blend{width=0.35,enabled=true})
    core.add(&blends,e_sphere,Mesh_Blend{width=0.35,enabled=true})
    core.add(&blends,e_float,Mesh_Blend{width=0.35,enabled=true})
    core.table_settle(&ids,&entities)
    core.table_settle(&parents,&entities)
    core.table_settle(&blends,&entities)
    core.hierarchy_settle(&hierarchy,&parents,&entities)

    state:Mesh_Blend_Runtime
    defer {
        mesh_blend_release(&state,{},r)
        mesh_blend_runtime_free(&state)
    }
    state.values.allocator=allocator
    state.scratch.allocator=allocator
    state.author_groups.allocator=allocator
    sync:=render.Render_Sync{}

    camera:=gpu.Camera{
        position={0,2.2,3.4},forward=linalg.normalize([3]f32{0,-0.55,-0.835}),up={0,1,0},
        fov_y=45,near=0.1,far=50,
    }
    floor_xform:=mm.transform_matrix({position={0,-0.5,0},rotation=mm.TRANSFORM_IDENTITY.rotation,scale={12,1,12}})
    sphere_xform:=mm.transform_matrix({position={0,0.35,0},rotation=mm.TRANSFORM_IDENTITY.rotation,scale={1.2,1.2,1.2}})
    float_xform:=mm.transform_matrix({position={1.3,1.2,0.5},rotation=mm.TRANSFORM_IDENTITY.rotation,scale={0.6,0.6,0.6}})

    gpu.scene_set_environment(r,{direction=linalg.normalize([3]f32{0.5,0.8,0.4}),color={1,0.98,0.95},illuminance=50000,enabled=true},{})

    submit :: proc(
        state:^Mesh_Blend_Runtime,sync:^render.Render_Sync,r:^gpu.Renderer,allocator:mem.Allocator,
        blends:^core.Table(Mesh_Blend),hierarchy:^core.Hierarchy,ids:^core.Table(core.Id),
        cfg:Mesh_Blend_Settings,
    ) {
        mesh_blend_render(state,{sync},r,allocator,core.read(blends),{hierarchy},core.read(ids),cfg)
    }

    read_tex :: proc(t:^testing.T,device:^rhi.Device,texture:rhi.Texture,bytes:[]u8) -> bool {
        return testing.expect(t,rhi.texture_read(device,texture,bytes),rhi.device_error(device))
    }

    capture_dir,_:=os.lookup_env("MIRAI_MESHBLEND_CAPTURE",context.temp_allocator)
    write_png :: proc(dir,name:string,r:^gpu.Renderer) {
        if dir=="" {return}
        pixels,w,h,ok:=gpu.renderer_capture(r,context.temp_allocator)
        if !ok {return}
        path:=strings.clone_to_cstring(fmt.tprintf("%s/%s.png",dir,name),context.temp_allocator)
        stbi.write_png(path,c.int(w),c.int(h),4,raw_data(pixels),c.int(w*4))
    }

    // Test both plain G-buffer (no Reflectance target) and Reflectance target (custom specular_factor).
    for custom_specular in ([2]bool{false,true}) {
        mat_floor:=gpu.DEFAULT_MATERIAL
        mat_floor.base_color={0.12,0.45,0.12}
        mat_floor.roughness=0.85
        mat_floor.metallic=0.0
        if custom_specular {mat_floor.specular_factor=0.3}

        mat_sphere:=gpu.DEFAULT_MATERIAL
        mat_sphere.base_color={0.85,0.22,0.12}
        mat_sphere.roughness=0.25
        mat_sphere.metallic=1.0

        mat_float:=gpu.DEFAULT_MATERIAL
        mat_float.base_color={0.1,0.15,0.85}
        mat_float.roughness=0.5
        mat_float.metallic=0.0

        gpu.materials_set(r,{mat_floor,mat_sphere,mat_float})
        gpu.scene_update(r,0,{
            {object_id=u64(e_floor),mesh=cube.handle,transform=floor_xform,material=1,tint=1},
            {object_id=u64(e_sphere),mesh=sphere.handle,transform=sphere_xform,material=2,tint=1},
            {object_id=u64(e_float),mesh=cube.handle,transform=float_xform,material=3,tint=1},
        },true)

        // First render WITHOUT MeshBlend as baseline.
        if !testing.expect(t,gpu.renderer_frame(r,{camera=camera}),r.error) {return}
        rhi.device_wait_idle(device)
        if !custom_specular {write_png(capture_dir,"before",r)}

        w,h:=int(r.main.targets.width),int(r.main.targets.height)
        albedo_off:=make([][4]u8,w*h,context.temp_allocator)
        normal_off:=make([][4]f16,w*h,context.temp_allocator)
        if !read_tex(t,device,r.main.targets.albedo,mem.slice_data_cast([]u8,albedo_off)) ||
           !read_tex(t,device,r.main.targets.normal,mem.slice_data_cast([]u8,normal_off)) {return}

        // Now render WITH MeshBlend.
        submit(&state,&sync,r,allocator,&blends,&hierarchy,&ids,MESH_BLEND_SETTINGS_DEFAULT)
        if !testing.expect(t,gpu.renderer_frame(r,{camera=camera}),r.error) {return}
        rhi.device_wait_idle(device)
        if !custom_specular {write_png(capture_dir,"after",r)}

        albedo_on:=make([][4]u8,w*h,context.temp_allocator)
        normal_on:=make([][4]f16,w*h,context.temp_allocator)
        if !read_tex(t,device,r.main.targets.albedo,mem.slice_data_cast([]u8,albedo_on)) ||
           !read_tex(t,device,r.main.targets.normal,mem.slice_data_cast([]u8,normal_on)) {return}

        // Verify:
        //  1. Seam pixels between floor (green, metallic=0, rough=0.85) and sphere (red, metallic=1, rough=0.25)
        //     actually blended base color, metallic, and roughness.
        //  2. Floating blue box (depth gap above floor) did NOT blend with floor!
        blended_metallic:=0
        blended_rough:=0
        false_float_blend:=0
        for i in 0..<w*h {
            a0,a1:=albedo_off[i],albedo_on[i]
            r0,r1:=f32(normal_off[i].z),f32(normal_on[i].z)
            // Floating box pixels are distinctly blue (b > 150, r < 100, g < 120).
            if a0.b>150 && a0.r<100 && a0.g<120 {
                if abs(int(a1.g)-int(a0.g))>8 || abs(int(a1.b)-int(a0.b))>8 {false_float_blend+=1}
                continue
            }
            if a0.a==0 && a1.a>10 && a1.a<245 {blended_metallic+=1}
            if a0.a==255 && a1.a>10 && a1.a<245 {blended_metallic+=1}
            if abs(r1-r0)>0.05 {blended_rough+=1}
        }
        testing.expectf(t,blended_metallic>50,"expected metallic seam blending on floor/sphere (custom_specular=%v), got %d pixels",custom_specular,blended_metallic)
        testing.expectf(t,blended_rough>50,"expected roughness seam blending on floor/sphere (custom_specular=%v), got %d pixels",custom_specular,blended_rough)
        testing.expectf(t,false_float_blend==0,"floating box with depth gap must not blend with floor, got %d pixels",false_float_blend)
        testing.expect_value(t,r.main.targets.surface!=0,custom_specular)

        if custom_specular {
            refl_on:=make([][4]u8,w*h,context.temp_allocator)
            if !read_tex(t,device,r.main.targets.surface,mem.slice_data_cast([]u8,refl_on)) {return}
            // Check reflectance target has intermediate F0 values along the seam.
            refl_blended:=0
            for i in 0..<w*h {
                if albedo_off[i].a==0 && albedo_on[i].a>20 && refl_on[i].r>80 {refl_blended+=1}
            }
            testing.expectf(t,refl_blended>20,"expected reflectance F0 blending on seam, got %d pixels",refl_blended)
        }
    }

    // Zero-seam frame (only floor): runs cleanly with 0 active tiles.
    gpu.scene_update(r,0,{{object_id=u64(e_floor),mesh=cube.handle,transform=floor_xform,material=1,tint=1}},true)
    submit(&state,&sync,r,allocator,&blends,&hierarchy,&ids,MESH_BLEND_SETTINGS_DEFAULT)
    if !testing.expect(t,gpu.renderer_frame(r,{camera=camera}),r.error) {return}
    rhi.device_wait_idle(device)

    // Restore intersecting scene and verify stable frames allocate 0 CPU memory and 0 GPU resources.
    gpu.scene_update(r,0,{
        {object_id=u64(e_floor),mesh=cube.handle,transform=floor_xform,material=1,tint=1},
        {object_id=u64(e_sphere),mesh=sphere.handle,transform=sphere_xform,material=2,tint=1},
    },true)
    for _ in 0..<3 {
        submit(&state,&sync,r,allocator,&blends,&hierarchy,&ids,MESH_BLEND_SETTINGS_DEFAULT)
        if !testing.expect(t,gpu.renderer_frame(r,{camera=camera}),r.error) {return}
    }
    rhi.device_wait_idle(device)
    before:=rhi.device_stats(device)
    allocs:=tracker.total_allocation_count
    {
        context.allocator=allocator
        submit(&state,&sync,r,allocator,&blends,&hierarchy,&ids,MESH_BLEND_SETTINGS_DEFAULT)
        if !testing.expect(t,gpu.renderer_frame(r,{camera=camera}),r.error) {return}
    }
    rhi.device_wait_idle(device)
    testing.expect_value(t,tracker.total_allocation_count,allocs)
    testing.expect_value(t,rhi.device_stats(device),before)
}
