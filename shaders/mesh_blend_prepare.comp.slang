#include "native.slang"
// Pass 2 (1 thread): populates GPU-driven JFA indirect dispatch arguments from
// seam_count and max_half_radius. Always starts on an even JFA step index
// (6, 4, 2, or 0) so the final step (k = 6, stride = 1) always writes jfa_b.
// When seam_count == 0, every JFA dispatch, resolve dispatch and apply draw is 0.
#include "scene.slang"
#include "mesh_blend_common.slang"

[[vk::binding(SLOT_PASS0, 0)]] RWStructuredBuffer<uint> args : register(MIRAI_REGISTER(u, SLOT_PASS0), space0);
[[vk::binding(SLOT_PASS1, 0)]] RWStructuredBuffer<uint> work : register(MIRAI_REGISTER(u, SLOT_PASS1), space0);

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint seams = args[0];
    uint max_r = args[1];

    // Initialize work header: resolve dispatch (0, 1, 1), apply draw (6, 0, 0, 0).
    work[0] = 0u;
    work[1] = 1u;
    work[2] = 1u;
    work[3] = 0u;
    work[4] = 6u;
    work[5] = 0u;
    work[6] = 0u;
    work[7] = 0u;

    uint hw = (uint(frame.resolution.x) + 1u) >> 1u;
    uint hh = (uint(frame.resolution.y) + 1u) >> 1u;
    uint gx = (hw + BLEND_TILE - 1u) / BLEND_TILE;
    uint gy = (hh + BLEND_TILE - 1u) / BLEND_TILE;

    // Pick even start index so ping-pong starts at jfa_a and finishes in jfa_b:
    //   k_start = 6 -> [1]                       (reach 1 half-res cell  =   2 px)
    //   k_start = 4 -> [4, 2, 1]                 (reach 7 half-res cells =  14 px)
    //   k_start = 2 -> [16, 8, 4, 2, 1]          (reach 31 half-res cells = 62 px)
    //   k_start = 0 -> [64, 32, 16, 8, 4, 2, 1]  (reach 127 half-res cells = 254 px)
    uint k_start = JFA_STEPS; // disables all JFA steps when seams == 0
    if (seams != 0u) {
        if (max_r <= 1u) k_start = 6u;
        else if (max_r <= 7u) k_start = 4u;
        else if (max_r <= 31u) k_start = 2u;
        else k_start = 0u;
    }

    [unroll]
    for (uint k = 0u; k < JFA_STEPS; k++) {
        uint base = 4u + k * 4u;
        bool run = k >= k_start;
        args[base + 0u] = run ? gx : 0u;
        args[base + 1u] = run ? gy : 0u;
        args[base + 2u] = run ? 1u : 0u;
        args[base + 3u] = 0u;
    }
}
