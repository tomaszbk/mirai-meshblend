#include "native.slang"
// Pass 3: bounded half-res Jump Flooding over exact full-res seam seeds.
// Each texel propagates the 2 closest seam seeds (with distinct canonical group
// pairs) whose pair matches a participating group in its 2x2 pixel cell, so
// foreground objects never pollute background seams and 3-way corners stay exact.
// On the final step (step == 1), emits active 8x8 full-res tiles into work[].
#include "scene.slang"
#include "depth.slang"
#include "mesh_blend_common.slang"

[[vk::binding(SLOT_PARAMS, 0)]] ConstantBuffer<Mesh_Blend_Params> params : register(MIRAI_REGISTER(b, SLOT_PARAMS), space0);
[[vk::binding(SLOT_PASS0, 0)]] Texture2D<float4> linear_depth : register(MIRAI_REGISTER(t, SLOT_PASS0), space0);
[[vk::binding(SLOT_PASS1, 0)]] Texture2D<uint4> jfa_in : register(MIRAI_REGISTER(t, SLOT_PASS1), space0);
[[vk::binding(SLOT_PASS2, 0)]] [[vk::image_format("rgba32ui")]] RWTexture2D<uint4> jfa_out : register(MIRAI_REGISTER(u, SLOT_PASS2), space0);
[[vk::binding(SLOT_PASS3, 0)]] RWStructuredBuffer<uint> work : register(MIRAI_REGISTER(u, SLOT_PASS3), space0);

// An 8x8 half-res workgroup covers 2x2 full-res 8x8 tiles (= 4 sub-tiles).
groupshared uint s_tile_active[4];

uint read_info(int2 p, int2 res) {
    if (any(p >= res)) return 0u;
    return work[params.info_offset + uint(p.y) * uint(res.x) + uint(p.x)];
}

bool cell_matches(uint pair, uint4 groups) {
    uint lo = pair & 0xffffu;
    uint hi = pair >> 16u;
    return any(groups == uint4(lo)) || any(groups == uint4(hi));
}

void consider_seed(
    float2 center, uint4 groups, uint seed, uint meta,
    inout uint s0, inout uint m0, inout float d0,
    inout uint s1, inout uint m1, inout float d1
) {
    if (seed == SEED_EMPTY || !cell_matches(meta, groups)) return;
    float2 delta = float2(seed_pos(seed)) + 0.5 - center;
    float dist2 = dot(delta, delta);
    if (s0 != SEED_EMPTY && meta == m0) {
        if (dist2 < d0) { s0 = seed; d0 = dist2; }
        return;
    }
    if (s1 != SEED_EMPTY && meta == m1) {
        if (dist2 < d1) { s1 = seed; d1 = dist2; }
        if (d1 < d0) {
            uint ts = s0, tm = m0; float td = d0;
            s0 = s1; m0 = m1; d0 = d1;
            s1 = ts; m1 = tm; d1 = td;
        }
        return;
    }
    if (s0 == SEED_EMPTY || dist2 < d0) {
        s1 = s0; m1 = m0; d1 = d0;
        s0 = seed; m0 = meta; d0 = dist2;
    } else if (s1 == SEED_EMPTY || dist2 < d1) {
        s1 = seed; m1 = meta; d1 = dist2;
    }
}

[shader("compute")]
[numthreads(BLEND_TILE, BLEND_TILE, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 local_id : SV_GroupThreadID, uint lane : SV_GroupIndex) {
    if (params.step == 1u && lane < 4u) s_tile_active[lane] = 0u;
    if (params.step == 1u) GroupMemoryBarrierWithGroupSync();

    int2 res = int2(frame.resolution);
    int2 half_res = (res + int2(1, 1)) >> 1;
    int2 h = int2(dispatch_id.xy);
    bool in_bounds = all(h < half_res);

    uint best_s0 = SEED_EMPTY, best_m0 = 0u;
    uint best_s1 = SEED_EMPTY, best_m1 = 0u;
    float best_d0 = 1e30, best_d1 = 1e30;
    bool needs_blend = false;

    if (in_bounds) {
        int2 base_p = h << 1;
        uint i00 = read_info(base_p + int2(0, 0), res);
        uint i10 = read_info(base_p + int2(1, 0), res);
        uint i01 = read_info(base_p + int2(0, 1), res);
        uint i11 = read_info(base_p + int2(1, 1), res);
        uint4 groups = uint4(info_group(i00), info_group(i10), info_group(i01), info_group(i11));

        if (any(groups != uint4(0u))) {
            float2 center = float2(base_p) + 1.0;
            int step = int(params.step);
            [unroll]
            for (int dy = -1; dy <= 1; dy++) {
                [unroll]
                for (int dx = -1; dx <= 1; dx++) {
                    int2 q = h + int2(dx, dy) * step;
                    if (any(q < int2(0)) || any(q >= half_res)) continue;
                    uint4 v = jfa_in.Load(int3(q, 0));
                    consider_seed(center, groups, v.x, v.y, best_s0, best_m0, best_d0, best_s1, best_m1, best_d1);
                    consider_seed(center, groups, v.z, v.w, best_s0, best_m0, best_d0, best_s1, best_m1, best_d1);
                }
            }

            if (params.step == 1u && best_s0 != SEED_EMPTY) {
                float max_w = max(max(info_width(i00), info_width(i10)), max(info_width(i01), info_width(i11)));
                float min_d = 1e30;
                [unroll]
                for (int cy = 0; cy < 2; cy++) {
                    [unroll]
                    for (int cx = 0; cx < 2; cx++) {
                        int2 qp = base_p + int2(cx, cy);
                        if (all(qp < res)) {
                            float ld = linear_depth.Load(int3(qp, 0)).r;
                            if (!depth_empty(ld)) min_d = min(min_d, depth_distance(ld));
                        }
                    }
                }
                if (min_d < 1e29) {
                    float w = mb_seam_width(max_w, min_d, frame.footprint, frame.resolution.y, params.min_pixels, params.max_widen);
                    float r = mb_seam_radius(w, min_d, frame.footprint, frame.resolution.y) + 2.5;
                    needs_blend = best_d0 <= r * r;
                }
            }
        }
        jfa_out[h] = uint4(best_s0, best_m0, best_s1, best_m1);
    }

    if (params.step == 1u) {
        if (needs_blend) {
            uint sub = (local_id.y >> 2u) * 2u + (local_id.x >> 2u);
            InterlockedOr(s_tile_active[sub], 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane < 4u && s_tile_active[lane] != 0u) {
            uint tx = group_id.x * 2u + (lane & 1u);
            uint ty = group_id.y * 2u + (lane >> 1u);
            uint tiles_x = (uint(res.x) + BLEND_TILE - 1u) / BLEND_TILE;
            uint tiles_y = (uint(res.y) + BLEND_TILE - 1u) / BLEND_TILE;
            if (tx < tiles_x && ty < tiles_y) {
                uint slot;
                InterlockedAdd(work[0], 1u, slot);
                if (slot < params.tiles_max) {
                    work[WORK_HEADER_WORDS + slot] = tx | (ty << 16u);
                    InterlockedMax(work[5], slot + 1u);
                } else {
                    work[0] = params.tiles_max;
                }
            }
        }
    }
}
