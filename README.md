# Mirai MeshBlend

A render plugin for the Mirai engine that blends the surfaces of meshes where they meet, before
they are lit: near the seam each surface takes the base color, the normal and the roughness of the
other, so light, shadows, global illumination and reflections are computed on the blended surface.
A rock sunk into the ground, a wall standing on a terrain or grass growing out of soil lose the hard
line where they cross.

It extends the renderer through `engine/render` (a compute stage and a pass of the `Surface`
stage, instance ids and per-instance values); it imports neither `render` nor `rhi`.

## Install

From the catalog, in the folder of a Mirai project:

```sh
mirai plugin add meshblend
```

or from the Catalog of the Plugins page in the project settings of the editor. It has no plugin
dependencies.

## Component

`Mesh_Blend` on an entity blends the surfaces of its meshes and those of its descendants with the
meshes they meet:

- `width` (m): how far from the seam the surfaces blend. The seam takes the smaller width of its
  two sides.
- `enabled`: false leaves the entity and its subtree out.

A descendant with its own `Mesh_Blend` decides for its subtree. The group of an instance is the
entity that created it (the one with an `Id`): the pieces of one model or the chunks of one
terrain do not blend with each other, only with other groups.

An [instance batch](#batches) (foliage) whose entity is under a `Mesh_Blend` takes its width and
that entity as its group, so its instances blend with what they touch and not with each other,
and the plugin that owns the batch does not need to know about MeshBlend.

## Settings

`Mesh_Blend_Settings`, a project setting:

- `noise`: how much the blend breaks up along the seam, with a world-space noise; `noise_scale`
  is its size as a part of the width, so the edge does not slide when the camera moves away.
- `min_pixels` and `max_widen` (advanced): far away the seam widens in the world so it is not
  narrower than `min_pixels` on screen (in pixels of a 1080-high image), up to `max_widen` times
  its own width.

## How it works

- `mesh_blend_render` gives each instance its values (width, group, whether it takes part) with
  `instance_values_set`, only when the tables, the hierarchy or the scene of the renderer change;
  with no participants it asks for no passes.
- `mesh_blend.comp` (compute of the `Surface` stage): every pixel that takes part searches around
  it, up to the projected width, for a surface of another group, with directions and distances
  that vary per pixel and per frame and that TAA averages. It finds the seam by bisection along
  the direction that found it and along another turned towards it, and measures the distance to
  the line through both points, so the width does not depend on the search direction. The blend
  fades as the two surfaces separate in depth (two that only overlap on screen do not blend) and
  mixes the color in OKLab, the normal and the roughness of the other side. A thin object blends
  over no more than half of itself.
- `mesh_blend_apply.frag` writes albedo and normal with `Alpha_Color` blending: metallic and the
  layer bits stay those of each surface.

### Batches

Batches of instances published through `engine/render` (`batch_set`) take their values with
`batch_values_set`, without rebuilding the renderer tables.

## Limits

- What does not enter the opaque G-buffer (glass, translucent surfaces, particles) does not blend.
- The neighbour has to be visible on screen.
- The F0, the layers and the ray hits see each material unblended.

## Verify

- CPU: in a project with the plugin, `mirai test <project>/plugins/meshblend/runtime`
  (`the_batches_of_an_entity_under_a_mesh_blend_take_its_width_and_the_group_of_the_entity`).
- In the editor of a project with the plugin, with GPU validation: a sphere sunk into a floor,
  both with `Mesh_Blend`, captured with and without `enabled`: the base of the sphere takes the
  color of the floor and the contact shadow stays. A scene with mirrors and layered materials
  draws without validation errors. A `Foliage` under a `Mesh_Blend` on a terrain with its own:
  the base of the grass takes the color of the ground and the tufts do not blend with each other.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
