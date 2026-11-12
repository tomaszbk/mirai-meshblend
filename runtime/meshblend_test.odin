package meshblend

import "core:math"
import "core:testing"
import "mirai:core"

@(test)
an_entity_takes_the_width_of_its_owner_and_the_group_of_its_author :: proc(t:^testing.T) {
    group:=core.Entity(0x0003_0000_0001_2345)
    value:=value_of({width=0.2,enabled=true},group)
    testing.expect_value(t,value,Blend_Value{0.2,u32(core.entity_index(group))})
    testing.expect_value(t,value_of({width=0.2,enabled=false},group).width,0)
    testing.expect_value(t,value_of({width=0,enabled=true},group).width,0)
    testing.expect_value(t,value_of({width=f32(math.inf_f32(1)),enabled=true},group).width,0)
    testing.expect(t,value_of({width=0.2,enabled=false},group).group==value.group,"a left out one keeps its group: its neighbors still see where it ends")
}

@(test)
an_entity_under_a_mesh_blend_takes_its_width_and_the_group_of_its_author :: proc(t:^testing.T) {
    entities:core.Entities;core.entities_init(&entities);defer core.entities_destroy(&entities)
    blends:core.Table(Mesh_Blend);core.table_init(&blends,"Mesh_Blend");defer core.table_destroy(&blends)
    parents:core.Table(core.Parent);core.table_init(&parents,"Parent");defer core.table_destroy(&parents)
    ids:core.Table(core.Id);core.table_init(&ids,"Id");defer core.table_destroy(&ids)
    island:=core.create(&entities)
    meadow:=core.create(&entities)
    lone:=core.create(&entities)
    core.add(&ids,island,core.Id{value=1});core.add(&ids,meadow,core.Id{value=2});core.add(&ids,lone,core.Id{value=3})
    core.add(&parents,meadow,core.Parent{entity=island})
    core.add(&blends,island,Mesh_Blend{width=0.3,enabled=true})
    core.table_settle(&ids,&entities);core.table_settle(&parents,&entities);core.table_settle(&blends,&entities)
    value,under:=entity_value(&blends,&parents,&ids,meadow)
    index:=u32(core.entity_index(meadow))
    testing.expect(t,under)
    testing.expect_value(t,value,Blend_Value{0.3,index})
    _,under=entity_value(&blends,&parents,&ids,lone)
    testing.expect(t,!under,"an entity under no Mesh_Blend takes no part")
}

@(test)
the_search_takes_the_settings_within_their_ranges :: proc(t:^testing.T) {
    d:=MESH_BLEND_SETTINGS_DEFAULT
    testing.expect_value(t,params_of(d),Gpu_Params{noise=d.noise,noise_scale=d.noise_scale,min_pixels=d.min_pixels,max_widen=d.max_widen})
    nan:=math.nan_f32()
    testing.expect_value(t,params_of({nan,nan,nan,nan}),Gpu_Params{noise=d.noise,noise_scale=d.noise_scale,min_pixels=d.min_pixels,max_widen=d.max_widen})
    testing.expect_value(t,params_of({noise=2,noise_scale=0,min_pixels=-1,max_widen=0}),Gpu_Params{noise=1,noise_scale=0.1,min_pixels=0,max_widen=1})
    testing.expect_value(t,params_of({noise=0.5,noise_scale=40,min_pixels=100,max_widen=20}),Gpu_Params{noise=0.5,noise_scale=16,min_pixels=64,max_widen=8})
}
