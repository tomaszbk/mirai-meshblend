package meshblend

import "core:math"
import "core:testing"
import "mirai:core"

@(test)
blend_value_packs_width_and_group_into_four_bytes :: proc(t:^testing.T) {
    testing.expect_value(t,size_of(Blend_Value),4)
    v:=pack_blend_value(0.25,42)
    testing.expect_value(t,blend_value_group(v),u16(42))
    testing.expect(t,abs(blend_value_width(v)-0.25)<1e-3,"f16 round-trips width")
    testing.expect_value(t,pack_blend_value(0,42),Blend_Value(0))
    testing.expect_value(t,pack_blend_value(-0.5,42),Blend_Value(0))
    testing.expect_value(t,pack_blend_value(math.nan_f32(),42),Blend_Value(0))
    testing.expect_value(t,pack_blend_value(f32(math.inf_f32(1)),42),Blend_Value(0))
    testing.expect_value(t,pack_blend_value(0.25,0),Blend_Value(0))
}

@(test)
values_build_walks_blends_subtrees_with_overrides_and_author_groups :: proc(t:^testing.T) {
    entities:core.Entities;core.entities_init(&entities);defer core.entities_destroy(&entities)
    blends:core.Table(Mesh_Blend);core.table_init(&blends,"Mesh_Blend");defer core.table_destroy(&blends)
    parents:core.Table(core.Parent);core.table_init(&parents,"Parent");defer core.table_destroy(&parents)
    ids:core.Table(core.Id);core.table_init(&ids,"Id");defer core.table_destroy(&ids)
    hierarchy:core.Hierarchy;defer {
        delete(hierarchy.parent);delete(hierarchy.first_child);delete(hierarchy.last_child)
        delete(hierarchy.next_sibling);delete(hierarchy.previous_sibling);delete(hierarchy.cut)
        delete(hierarchy.moved.items)
    }

    // Scene:
    //   island (Id, Mesh_Blend 0.3)
    //     ├── meadow (Id)
    //     │     └── grass_chunk (no Id -> inherits meadow's group)
    //     ├── rock (Id, Mesh_Blend 0.1 -> overrides width for its subtree)
    //     │     └── rock_part (no Id -> inherits rock's group & 0.1 width)
    //     └── excluded (Id, Mesh_Blend enabled=false -> leaves subtree out)
    //           └── excluded_child (no Id -> also excluded)
    //   lone (Id, outside any Mesh_Blend)
    island:=core.create(&entities)
    meadow:=core.create(&entities)
    grass_chunk:=core.create(&entities)
    rock:=core.create(&entities)
    rock_part:=core.create(&entities)
    excluded:=core.create(&entities)
    excluded_child:=core.create(&entities)
    lone:=core.create(&entities)

    core.add(&ids,island,core.Id{value=1})
    core.add(&ids,meadow,core.Id{value=2})
    core.add(&ids,rock,core.Id{value=3})
    core.add(&ids,excluded,core.Id{value=4})
    core.add(&ids,lone,core.Id{value=5})

    core.add(&parents,meadow,core.Parent{entity=island})
    core.add(&parents,grass_chunk,core.Parent{entity=meadow})
    core.add(&parents,rock,core.Parent{entity=island})
    core.add(&parents,rock_part,core.Parent{entity=rock})
    core.add(&parents,excluded,core.Parent{entity=island})
    core.add(&parents,excluded_child,core.Parent{entity=excluded})

    core.add(&blends,island,Mesh_Blend{width=0.3,enabled=true})
    core.add(&blends,rock,Mesh_Blend{width=0.1,enabled=true})
    core.add(&blends,excluded,Mesh_Blend{width=0.5,enabled=false})

    core.table_settle(&ids,&entities)
    core.table_settle(&parents,&entities)
    core.table_settle(&blends,&entities)
    core.hierarchy_settle(&hierarchy,&parents,&entities)

    state:Mesh_Blend_Runtime
    defer mesh_blend_runtime_free(&state)
    active:=values_build(&state,&blends,{&hierarchy},&ids)
    testing.expect_value(t,active,5) // island, meadow, grass_chunk, rock, rock_part

    at :: proc(s:^Mesh_Blend_Runtime,e:core.Entity) -> Blend_Value {
        i:=core.entity_index(e)
        return s.values[i] if i<len(s.values) else 0
    }
    v_island:=at(&state,island)
    v_meadow:=at(&state,meadow)
    v_chunk:=at(&state,grass_chunk)
    v_rock:=at(&state,rock)
    v_rock_part:=at(&state,rock_part)

    testing.expect(t,v_island!=0 && abs(blend_value_width(v_island)-0.3)<1e-3)
    testing.expect(t,v_meadow!=0 && blend_value_group(v_meadow)!=blend_value_group(v_island))
    testing.expect_value(t,v_chunk,v_meadow) // piece without Id shares author's group & width
    testing.expect(t,v_rock!=0 && abs(blend_value_width(v_rock)-0.1)<1e-3)
    testing.expect_value(t,v_rock_part,v_rock)
    testing.expect(t,blend_value_group(v_rock)!=blend_value_group(v_island))
    testing.expect_value(t,at(&state,excluded),Blend_Value(0))
    testing.expect_value(t,at(&state,excluded_child),Blend_Value(0))
    testing.expect_value(t,at(&state,lone),Blend_Value(0))

    // Second rebuild with no changes produces an empty dirty range.
    state.dirty_lo,state.dirty_hi=0,0
    values_build(&state,&blends,{&hierarchy},&ids)
    testing.expect_value(t,state.dirty_lo,state.dirty_hi)

    // Changing only rock's width dirties only rock and rock_part.
    if r:=core.edit(&blends,rock);r!=nil {r.width=0.4}
    core.table_settle(&blends,&entities)
    values_build(&state,&blends,{&hierarchy},&ids)
    testing.expect_value(t,state.dirty_lo,core.entity_index(rock))
    testing.expect_value(t,state.dirty_hi,core.entity_index(rock_part)+1)
    testing.expect(t,abs(blend_value_width(at(&state,rock_part))-0.4)<1e-3)
}

@(test)
the_search_takes_the_settings_within_their_ranges :: proc(t:^testing.T) {
    d:=MESH_BLEND_SETTINGS_DEFAULT
    testing.expect_value(t,params_of(d),Gpu_Params{noise=d.noise,noise_scale=d.noise_scale,min_pixels=d.min_pixels,max_widen=d.max_widen,slope=d.slope})
    nan:=math.nan_f32()
    testing.expect_value(t,params_of({nan,nan,nan,nan,nan}),Gpu_Params{noise=d.noise,noise_scale=d.noise_scale,min_pixels=d.min_pixels,max_widen=d.max_widen,slope=d.slope})
    testing.expect_value(t,params_of({noise=2,noise_scale=0,min_pixels=-1,max_widen=0,slope=0}),Gpu_Params{noise=1,noise_scale=0.1,min_pixels=0,max_widen=1,slope=1})
    testing.expect_value(t,params_of({noise=0.5,noise_scale=40,min_pixels=100,max_widen=20,slope=50}),Gpu_Params{noise=0.5,noise_scale=16,min_pixels=64,max_widen=8,slope=8})
}
