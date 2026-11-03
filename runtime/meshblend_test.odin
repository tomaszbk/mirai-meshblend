package meshblend

import "core:math"
import "core:testing"
import "mirai:core"

@(test)
an_instance_takes_the_width_of_its_owner_and_the_group_of_its_author :: proc(t:^testing.T) {
    group:=core.Entity(0x0003_0000_0001_2345)
    value:=value_of({width=0.2,enabled=true},group)
    index:=u32(core.entity_index(group))
    testing.expect_value(t,value,[4]f32{0.2,f32(index&0xffff),f32(index>>16),1})
    testing.expect_value(t,value_of({width=0.2,enabled=false},group).w,0)
    testing.expect_value(t,value_of({width=0,enabled=true},group).w,0)
    testing.expect_value(t,value_of({width=f32(math.inf_f32(1)),enabled=true},group).w,0)
    testing.expect(t,value_of({width=0.2,enabled=false},group).yz==value.yz,"a left out one keeps its group: its neighbors still see where it ends")
}

@(test)
the_batches_of_an_entity_under_a_mesh_blend_take_its_width_and_the_group_of_the_entity :: proc(t:^testing.T) {
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
    value,under:=batch_value(core.read(&blends),core.read(&parents),core.read(&ids),meadow)
    index:=u32(core.entity_index(meadow))
    testing.expect(t,under)
    testing.expect_value(t,value,[4]f32{0.3,f32(index&0xffff),f32(index>>16),1})
    _,under=batch_value(core.read(&blends),core.read(&parents),core.read(&ids),lone)
    testing.expect(t,!under,"a batch not under a Mesh_Blend keeps what its owner gave it")
}

@(test)
the_search_takes_the_settings_within_their_ranges :: proc(t:^testing.T) {
    d:=MESH_BLEND_SETTINGS_DEFAULT
    testing.expect_value(t,params_of(d),Gpu_Params{d.noise,d.noise_scale,d.min_pixels,d.max_widen})
    nan:=math.nan_f32()
    testing.expect_value(t,params_of({nan,nan,nan,nan}),Gpu_Params{d.noise,d.noise_scale,d.min_pixels,d.max_widen})
    testing.expect_value(t,params_of({noise=2,noise_scale=0,min_pixels=-1,max_widen=0}),Gpu_Params{1,0.1,0,1})
    testing.expect_value(t,params_of({noise=0.5,noise_scale=40,min_pixels=100,max_widen=20}),Gpu_Params{0.5,16,64,8})
}
