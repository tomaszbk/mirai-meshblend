# Mirai MeshBlend

A render plugin for the Mirai engine that blends the surfaces of meshes where they meet, before
they are lit: near the seam each surface takes the base color, metallic, normal, roughness and
reflectance of the other, so light, shadows, global illumination and reflections are computed on
the blended surface. A rock sunk into the ground, a metal post in concrete, a wall standing on a
terrain or grass growing out of soil lose the hard line where they cross.

It extends the renderer through `engine/render` (compute stages and an indirect tile pass of the
`Surface` stage, instance ids and world buffers); it imports neither `render` nor `rhi`.

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

- `noise`: how much the blend breaks up along the seam, with a world-space noise anchored to the
  rest surface (`pixel_rest`, so swaying foliage does not slide through it); `noise_scale` is its
  size as a part of the width, so the edge does not slide when the camera moves away.
- `min_pixels` and `max_widen` (advanced): far away the seam widens in the world so it is not
  narrower than `min_pixels` on screen (in pixels of a 1080-high image), up to `max_widen` times
  its own width.
- `slope` (advanced): narrows steep (~90°) intersections by up to this factor compared to coplanar
  joins, avoiding wide halos where vertical geometry meets flat ground.

## How it works

- `mesh_blend_render` walks only the subtrees rooted at `Mesh_Blend` components via `core.Hierarchy`
  (independent of total scene instance/batch count), packs each participating entity into 4 bytes
  (`f16` width + dense 16-bit group id), and uploads only changed dirty ranges to a per-world
  `pass_buffer`. With no active `Mesh_Blend` components it asks for no passes.
- `mesh_blend_seed.comp`: resolves each pixel's instance id once per wave (`WaveReadLaneFirst`)
  into a compact 4-byte `pixel_info` buffer, checks 4-neighbors in a 10×10 `groupshared` tile for
  exact 3D-contact seams (rejecting depth occlusions), seeds half-res JFA, and tracks active seam
  count and maximum screen radius.
- `mesh_blend_prepare.comp` + `mesh_blend_jfa.comp`: GPU-driven bounded half-res Jump Flooding
  (0 dispatches when no seams are visible; 1 to 7 steps scaled to the widest seam on screen).
  Completely deterministic with no temporal jitter (works with TAA off, FSR, and fast motion),
  propagating up to 2 distinct seam pairs per cell for exact 3-way corners. The final step emits
  a compact list of active 8×8 screen tiles.
- `mesh_blend_resolve.comp`: indirect compute dispatch over active 8×8 tiles only. Computes exact
  3D tangent-plane seam contact, thin-object half-width capping, rest-space breakup noise,
  anti-streaked multi-tap neighbor sampling, OKLab base color blending, metallic, facing-safe
  normals, roughness, and reflectance (`F0`/`F90` when `FRAME_FLAG_REFLECTANCE` is active),
  including inside mirrors and portals (`Nested_View`), packed losslessly into `RGBA32_Uint`.
- `mesh_blend_apply.vert` + `mesh_blend_apply.frag`: indirect quad draw over active 8×8 tiles
  only, writing `Albedo`, `Normal` (preserving layer bits) and `Reflectance` (when active).

### Batches

Batches of instances published through `engine/render` (`batch_set`) take the values of their
entity: the seed pass reads the object id of the batch, so nothing is given to the batch itself.

## Limits

- What does not enter the opaque G-buffer (glass, translucent surfaces, particles) does not blend.
- The neighbour has to be visible on screen.
- Material layers (coat, fuzz, anisotropy, diffuse transmission) and ray hits see each material
  unblended.

## Verify

- CPU: in a project with the plugin, `mirai test <project>/plugins/meshblend/runtime`.
- GPU with validation: a metallic or dielectric mesh sunk into a floor (with and without custom
  specular reflectance, mirrors/portals, and swaying foliage under `Mesh_Blend`), verifying zero
  validation errors, zero stable-frame allocations, smooth PBR seam blending, and zero false
  blends across depth gaps.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
