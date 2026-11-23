#include "native.slang"
// Expands each active 8x8 tile into a screen-space quad so mesh_blend_apply.frag
// only shades tiles that actually contain blended seams (0 instances when none).
#include "scene.slang"
#include "mesh_blend_common.slang"

[[vk::binding(SLOT_PASS1, 0)]] StructuredBuffer<uint> work : register(MIRAI_REGISTER(t, SLOT_PASS1), space0);

static const float2 TILE_CORNERS[6] = {
    float2(0.0, 0.0), float2(1.0, 0.0), float2(1.0, 1.0),
    float2(0.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0),
};

struct Stage_Output {
    float4 position : SV_Position;
};

[shader("vertex")]
Stage_Output main(uint vertex_id : SV_VulkanVertexID, uint instance_id : SV_VulkanInstanceID) {
    uint tile_packed = work[WORK_HEADER_WORDS + instance_id];
    float2 origin = float2(float(tile_packed & 0xffffu), float(tile_packed >> 16u)) * float(BLEND_TILE);
    float2 corner = TILE_CORNERS[vertex_id % 6u];
    float2 px = min(origin + corner * float(BLEND_TILE), frame.resolution);
    float2 uv = px * frame.inv_resolution;
    Stage_Output output;
    output.position = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return output;
}
