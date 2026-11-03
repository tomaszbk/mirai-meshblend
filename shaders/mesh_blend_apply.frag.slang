#include "native.slang"
// Writes the surfaces mesh_blend.comp blended over the G-buffer: the base
// color and the normal with its roughness. The alpha of each target stays:
// the metallic of the surface and the bits of its layers.
#include "common.slang"

[[vk::binding(SLOT_PASS0, 0)]] Texture2D<uint2> blended : register(MIRAI_REGISTER(t, SLOT_PASS0), space0);

struct Stage_Output {
    [[vk::location(0)]] float4 out_albedo : SV_Target0;
    [[vk::location(1)]] float4 out_normal : SV_Target1;
};

float unpack_unorm(uint packed, uint shift, uint bits) { return float((packed >> shift) & ((1u << bits) - 1u)) / float((1u << bits) - 1u); }

[shader("fragment")]
Stage_Output main(float4 pixel : SV_Position) {
    uint2 v = blended.Load(int3(int2(pixel.xy), 0));
    if ((v.x >> 24u) == 0u) discard;
    float3 coded = float3(unpack_unorm(v.x, 0u, 8u), unpack_unorm(v.x, 8u, 8u), unpack_unorm(v.x, 16u, 8u));
    float2 oct = float2(unpack_unorm(v.y, 0u, 12u), unpack_unorm(v.y, 12u, 12u)) * 2.0 - 1.0;
    Stage_Output output;
    output.out_albedo = float4(coded * coded, 1.0);
    output.out_normal = float4(oct, unpack_unorm(v.y, 24u, 8u), 1.0);
    return output;
}
