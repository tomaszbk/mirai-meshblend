#include "native.slang"
// Pass 1: resolves each pixel's instance id into compact 4-byte pixel_info
// (deduplicated per wave so flat tiles touch instances[] and values[] once),
// detects exact 3D-contact seams across 4-neighbors in a 10x10 shared tile,
// seeds half-res jfa_a and tracks seam_count / max_half_radius for GPU-driven JFA.
#include "scene.slang"
#include "pixel.slang"
#include "mesh_blend_common.slang"

[[vk::binding(SLOT_PARAMS, 0)]] ConstantBuffer<Mesh_Blend_Params> params : register(MIRAI_REGISTER(b, SLOT_PARAMS), space0);
[[vk::binding(SLOT_PASS0, 0)]] Texture2D<float4> linear_depth : register(MIRAI_REGISTER(t, SLOT_PASS0), space0);
[[vk::binding(SLOT_PASS1, 0)]] Texture2D<uint> instance_ids : register(MIRAI_REGISTER(t, SLOT_PASS1), space0);
[[vk::binding(SLOT_PASS2, 0)]] StructuredBuffer<uint> values : register(MIRAI_REGISTER(t, SLOT_PASS2), space0);
[[vk::binding(SLOT_PASS3, 0)]] RWStructuredBuffer<uint> args : register(MIRAI_REGISTER(u, SLOT_PASS3), space0);
[[vk::binding(SLOT_PASS4, 0)]] RWStructuredBuffer<uint> work : register(MIRAI_REGISTER(u, SLOT_PASS4), space0);
[[vk::binding(SLOT_PASS5, 0)]] [[vk::image_format("rgba32ui")]] RWTexture2D<uint4> jfa_out : register(MIRAI_REGISTER(u, SLOT_PASS5), space0);

groupshared uint s_info[BLEND_TILE + 2u][BLEND_TILE + 2u];
groupshared float s_depth[BLEND_TILE + 2u][BLEND_TILE + 2u];
groupshared uint s_seed[BLEND_TILE][BLEND_TILE];
groupshared uint s_meta[BLEND_TILE][BLEND_TILE];
groupshared uint s_seam_any;
groupshared uint s_max_half_r;

bool inside_view(int2 p) { return all(p >= int2(0)) && all(p < int2(frame.resolution)); }

// Wave-deduplicated entity lookup: when all active lanes share an instance id,
// instances[] and values[] are loaded once for the whole wave.
uint lookup_instance_value(uint id) {
    if (id == 0u) return 0u;
    for (;;) {
        uint first = WaveReadLaneFirst(id);
        if (first != id) continue;
        uint index = first - 1u;
        uint object = instance_batched(index) ? batch_row(batch_of(index)).meta.z : instances[index].meta.z;
        return object < params.values ? values[object] : 0u;
    }
}

uint load_pixel(int2 q, out float depth) {
    depth = 0.0;
    if (!inside_view(q)) return 0u;
    depth = linear_depth.Load(int3(q, 0)).r;
    if (depth_empty(depth)) return 0u;
    uint id = instance_ids.Load(int3(q, 0));
    uint raw = lookup_instance_value(id);
    if (raw == 0u) return 0u;
    return pack_pixel_info(raw, pixel_view(q));
}

float3 pos_at(int2 p, float depth) {
    float2 uv = (float2(p) + 0.5) * frame.inv_resolution;
    return pixel_at(p, uv, depth).position;
}

// Fast exact 3D-contact check between two adjacent pixels p and q of different groups.
// Rejects screen-space occlusions where one surface merely overlaps another in depth.
bool touching_seam(int2 p, int2 q, int2 step_dir, uint info_p, uint info_q, float depth_p, float depth_q, float width) {
    float dp = depth_distance(depth_p);
    float dq = depth_distance(depth_q);
    float footprint = frame.footprint.x + frame.footprint.y * max(dp, dq);
    float3 pos_p = pos_at(p, depth_p);
    float3 pos_q = pos_at(q, depth_q);
    float3 boundary = pos_q - pos_p;

    // Account for grazing angles using 1-pixel tangents on each side when available.
    float surface_footprint = footprint;
    int2 src_p = p - step_dir;
    int2 dst_p = q + step_dir;
    float3 src_tan = float3(0.0), dst_tan = float3(0.0);
    bool src_ok = false, dst_ok = false;

    if (inside_view(src_p)) {
        float ds = linear_depth.Load(int3(src_p, 0)).r;
        if (!depth_empty(ds) && depth_same_side(depth_p, ds) && pixel_view(src_p) == info_view(info_p)) {
            uint sid = instance_ids.Load(int3(src_p, 0));
            uint sraw = sid != 0u ? (instance_batched(sid - 1u) ? batch_row(batch_of(sid - 1u)).meta.z : instances[sid - 1u].meta.z) : 0xffffffffu;
            uint sinfo = sraw < params.values ? pack_pixel_info(values[sraw], info_view(info_p)) : 0u;
            if (info_same_group(sinfo, info_p)) {
                src_tan = pos_p - pos_at(src_p, ds);
                surface_footprint = max(surface_footprint, length(src_tan));
                src_ok = true;
            }
        }
    }
    if (inside_view(dst_p)) {
        float dt = linear_depth.Load(int3(dst_p, 0)).r;
        if (!depth_empty(dt) && depth_same_side(depth_q, dt) && pixel_view(dst_p) == info_view(info_q)) {
            uint tid = instance_ids.Load(int3(dst_p, 0));
            uint traw = tid != 0u ? (instance_batched(tid - 1u) ? batch_row(batch_of(tid - 1u)).meta.z : instances[tid - 1u].meta.z) : 0xffffffffu;
            uint tinfo = traw < params.values ? pack_pixel_info(values[traw], info_view(info_q)) : 0u;
            if (info_same_group(tinfo, info_q)) {
                dst_tan = pos_at(dst_p, dt) - pos_q;
                surface_footprint = max(surface_footprint, length(dst_tan));
                dst_ok = true;
            }
        }
    }

    float max_gap = min(max(surface_footprint * 3.0, width * 0.1), width);
    if (length(boundary) >= max_gap) return false;
    if (src_ok && dst_ok) {
        float3 diff = src_tan - dst_tan;
        float diff2 = dot(diff, diff);
        float amount = diff2 > footprint * footprint * 1e-4 ?
                       saturate(dot(boundary - dst_tan, diff) / max(diff2, 1e-12)) : 0.5;
        float sep = length(boundary - lerp(dst_tan, src_tan, amount));
        if (sep >= max(footprint * 0.5, width * 0.1)) return false;
    }
    return true;
}

[shader("compute")]
[numthreads(BLEND_TILE, BLEND_TILE, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 local_id : SV_GroupThreadID, uint lane : SV_GroupIndex) {
    if (lane == 0u) {
        s_seam_any = 0u;
        s_max_half_r = 0u;
    }

    // Load 10x10 tile (8x8 + 1px halo) into shared memory.
    int2 tile_origin = int2(group_id.xy) * int(BLEND_TILE) - int2(1, 1);
    const uint TILE_SIDE = BLEND_TILE + 2u;
    const uint TILE_CELLS = TILE_SIDE * TILE_SIDE;
    for (uint idx = lane; idx < TILE_CELLS; idx += BLEND_TILE * BLEND_TILE) {
        uint sy = idx / TILE_SIDE;
        uint sx = idx % TILE_SIDE;
        float d;
        s_info[sy][sx] = load_pixel(tile_origin + int2(int(sx), int(sy)), d);
        s_depth[sy][sx] = d;
    }
    GroupMemoryBarrierWithGroupSync();

    int2 p = int2(dispatch_id.xy);
    uint lx = local_id.x;
    uint ly = local_id.y;
    uint info = s_info[ly + 1u][lx + 1u];
    float depth = s_depth[ly + 1u][lx + 1u];

    if (inside_view(p)) {
        work[params.info_offset + uint(p.y) * uint(frame.resolution.x) + uint(p.x)] = info;
    }

    uint best_seed = SEED_EMPTY;
    uint best_meta = 0u;
    uint local_half_r = 0u;

    if (info_active(info)) {
        float self_w = info_width(info);
        float self_d = depth_distance(depth);
        const int2 dirs[4] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1)};
        [unroll]
        for (uint d = 0u; d < 4u; d++) {
            int2 off = dirs[d];
            uint nb_info = s_info[uint(int(ly + 1u) + off.y)][uint(int(lx + 1u) + off.x)];
            if (!info_other_group(info, nb_info)) continue;
            float nb_depth = s_depth[uint(int(ly + 1u) + off.y)][uint(int(lx + 1u) + off.x)];
            float own_w = min(self_w, info_width(nb_info));
            float w = mb_seam_width(own_w, self_d, frame.footprint, frame.resolution.y, params.min_pixels, params.max_widen);
            if (!touching_seam(p, p + off, off, info, nb_info, depth, nb_depth, w)) continue;
            float r = mb_seam_radius(w, self_d, frame.footprint, frame.resolution.y);
            local_half_r = max(local_half_r, uint(ceil(r * 0.5)));
            if (best_seed == SEED_EMPTY) {
                best_seed = pack_seed(p, d);
                best_meta = pack_pair(info_group(info), info_group(nb_info));
            }
        }
    }

    s_seed[ly][lx] = best_seed;
    s_meta[ly][lx] = best_meta;
    if (best_seed != SEED_EMPTY) {
        InterlockedOr(s_seam_any, 1u);
        InterlockedMax(s_max_half_r, max(local_half_r, 1u));
    }
    GroupMemoryBarrierWithGroupSync();

    // Reduce each 2x2 full-res quad into one half-res jfa_a texel (up to 2 distinct seam pairs).
    if ((lx & 1u) == 0u && (ly & 1u) == 0u) {
        int2 h = p >> 1;
        int2 half_res = (int2(frame.resolution) + int2(1, 1)) >> 1;
        if (all(h < half_res)) {
            uint seed0 = SEED_EMPTY, meta0 = 0u;
            uint seed1 = SEED_EMPTY, meta1 = 0u;
            [unroll]
            for (uint dy = 0u; dy < 2u; dy++) {
                [unroll]
                for (uint dx = 0u; dx < 2u; dx++) {
                    uint s = s_seed[ly + dy][lx + dx];
                    if (s == SEED_EMPTY) continue;
                    uint m = s_meta[ly + dy][lx + dx];
                    if (seed0 == SEED_EMPTY) {
                        seed0 = s;
                        meta0 = m;
                    } else if (m != meta0 && seed1 == SEED_EMPTY) {
                        seed1 = s;
                        meta1 = m;
                    }
                }
            }
            jfa_out[h] = uint4(seed0, meta0, seed1, meta1);
        }
    }

    // One atomic update per workgroup that found a seam.
    if (lane == 0u && s_seam_any != 0u) {
        InterlockedAdd(args[0], 1u);
        InterlockedMax(args[1], s_max_half_r);
    }
}
