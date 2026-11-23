#include "native.slang"
//! variant reflectance MIRAI_MESH_BLEND_REFLECTANCE=1
// Writes the full PBR blended surface (albedo + metallic, normal + roughness +
// preserved layer bits, and reflectance when active) into the G-buffer on active tiles.
#include "common.slang"
#include "mesh_blend_common.slang"

[[vk::binding(SLOT_PASS0, 0)]] Texture2D<uint4> surface_in : register(MIRAI_REGISTER(t, SLOT_PASS0), space0);

struct Stage_Output {
    [[vk::location(0)]] float4 out_albedo : SV_Target0;
    [[vk::location(1)]] float4 out_normal : SV_Target1;
#ifdef MIRAI_MESH_BLEND_REFLECTANCE
    [[vk::location(2)]] float4 out_reflectance : SV_Target2;
#endif
};

[shader("fragment")]
Stage_Output main(float4 pixel : SV_Position) {
    uint4 v = surface_in.Load(int3(int2(pixel.xy), 0));
    if ((v.z & 0x80000000u) == 0u) discard;
    Stage_Output output;
    output.out_albedo = unpack_srgb_rgba8(v.x);
    float2 oct = float2(unpack_unorm16(v.y, 0u), unpack_unorm16(v.y, 16u)) * 2.0 - 1.0;
    float roughness = unpack_unorm16(v.z, 0u);
    uint bits = (v.z >> 16u) & 0x7fffu;
    output.out_normal = float4(oct, roughness, float(bits));
#ifdef MIRAI_MESH_BLEND_REFLECTANCE
    output.out_reflectance = unpack_srgb_rgba8(v.w);
#endif
    return output;
}
