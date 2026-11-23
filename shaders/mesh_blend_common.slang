//! mirai needs instance_ids
//! mirai constant BLEND_TILE uint ../runtime/meshblend.odin
//! mirai constant MIN_PIXELS_HEIGHT float ../runtime/meshblend.odin
//! mirai constant MAX_RADIUS float ../runtime/meshblend.odin
//! mirai constant JFA_STEPS uint ../runtime/meshblend.odin
//! mirai constant WORK_HEADER_WORDS uint ../runtime/meshblend.odin
#ifndef MIRAI_MESH_BLEND_COMMON_SLANG
#define MIRAI_MESH_BLEND_COMMON_SLANG
#include <constants/plugins/meshblend.slang>

// Shared parameters (meshblend.Gpu_Params, 48 bytes).
struct Mesh_Blend_Params {
    float noise;
    float noise_scale;
    float min_pixels;
    float max_widen;
    uint values;
    uint info_offset;
    uint tiles_max;
    uint step;
    float slope;
    uint3 pad;
};

static const uint SEED_EMPTY = 0xffffffffu;

// ---------------------------------------------------------------- packed pixel_info
// Per-pixel metadata stored in work[info_offset + y * width + x]:
//   bits  0..15: dense group id (1..65535; 0 = non-participant)
//   bits 16..27: top 12 bits of positive IEEE-754 f16 width
//   bits 28..31: view index (0 = main view, 1..8 = nested view)
bool info_active(uint info) { return (info & 0xffffu) != 0u; }
uint info_group(uint info) { return info & 0xffffu; }
uint info_view(uint info) { return info >> 28u; }
// Positive half-float decode without 16-bit SPIR-V capabilities (as batch_instance.slang).
float info_width(uint info) { return asfloat(((info >> 16u) & 0xfffu) << 16u) * asfloat(0x77800000u); }
uint pack_pixel_info(uint raw_value, uint view_id) {
    uint group = raw_value & 0xffffu;
    uint w12 = (raw_value >> 19u) & 0xfffu;
    if (group == 0u || w12 == 0u) return 0u;
    return group | (w12 << 16u) | ((view_id & 0xfu) << 28u);
}
bool info_same_group(uint a, uint b) { return info_group(a) == info_group(b) && info_view(a) == info_view(b); }
bool info_other_group(uint a, uint b) {
    return info_active(a) && info_active(b) && info_view(a) == info_view(b) && info_group(a) != info_group(b);
}

// ---------------------------------------------------------------- packed seam seed
// One 32-bit seam seed:
//   bits  0..14: full-res pixel x (0..32767)
//   bits 15..29: full-res pixel y (0..32767)
//   bits 30..31: 4-neighbor direction to touching partner (0=+X, 1=-X, 2=+Y, 3=-Y)
int2 seed_dir_offset(uint dir) {
    return dir == 0u ? int2(1, 0) : dir == 1u ? int2(-1, 0) : dir == 2u ? int2(0, 1) : int2(0, -1);
}
uint pack_seed(int2 p, uint dir) {
    return (uint(p.x) & 0x7fffu) | ((uint(p.y) & 0x7fffu) << 15u) | ((dir & 3u) << 30u);
}
int2 seed_pos(uint seed) { return int2(int(seed & 0x7fffu), int((seed >> 15u) & 0x7fffu)); }
uint seed_dir(uint seed) { return seed >> 30u; }
int2 seed_partner(uint seed) { return seed_pos(seed) + seed_dir_offset(seed_dir(seed)); }

// Canonical unordered pair of the two seam groups (order-independent so A|B == B|A).
uint pack_pair(uint g0, uint g1) { return min(g0, g1) | (max(g0, g1) << 16u); }
bool pair_has_group(uint pair, uint g) { return g != 0u && ((pair & 0xffffu) == g || (pair >> 16u) == g); }

// ---------------------------------------------------------------- surface packing (RGBA32_Uint)
// Lossless G-buffer storage without 16-bit SPIR-V capabilities:
//   x: sRGB8 albedo + unorm8 metallic
//   y: unorm16x2 octahedral normal
//   z: unorm16 roughness | ((bits | 0x8000u) << 16u) (bit 31 = valid pixel flag)
//   w: sRGB8 reflectance F0 + unorm8 F90
float3 mb_linear_to_srgb(float3 c) {
    c = saturate(c);
    return lerp(c * 12.92, 1.055 * pow(max(c, float3(1e-7)), float3(1.0 / 2.4)) - 0.055, step(float3(0.0031308), c));
}
float3 mb_srgb_to_linear(float3 c) {
    c = saturate(c);
    return lerp(c * (1.0 / 12.92), pow((c + 0.055) * (1.0 / 1.055), float3(2.4)), step(float3(0.04045), c));
}
uint pack_unorm8(float v) { return uint(round(saturate(v) * 255.0)); }
uint pack_unorm16(float v) { return uint(round(saturate(v) * 65535.0)); }
float unpack_unorm8(uint v, uint shift) { return float((v >> shift) & 255u) * (1.0 / 255.0); }
float unpack_unorm16(uint v, uint shift) { return float((v >> shift) & 65535u) * (1.0 / 65535.0); }

uint pack_srgb_rgba8(float3 linear_rgb, float alpha) {
    float3 s = mb_linear_to_srgb(linear_rgb);
    return pack_unorm8(s.r) | (pack_unorm8(s.g) << 8u) | (pack_unorm8(s.b) << 16u) | (pack_unorm8(alpha) << 24u);
}
float4 unpack_srgb_rgba8(uint packed) {
    float3 s = float3(unpack_unorm8(packed, 0u), unpack_unorm8(packed, 8u), unpack_unorm8(packed, 16u));
    return float4(mb_srgb_to_linear(s), unpack_unorm8(packed, 24u));
}

// ---------------------------------------------------------------- screen projection helpers
float mb_projected(float width, float depth, float2 footprint) {
    return width / max(footprint.x + footprint.y * depth, 1e-6);
}
float mb_seam_width(float width, float depth, float2 footprint, float resolution_y, float min_pixels, float max_widen) {
    float wanted = min_pixels * resolution_y / MIN_PIXELS_HEIGHT;
    return width * clamp(wanted / max(mb_projected(width, depth, footprint), 1e-6), 1.0, max_widen);
}
float mb_seam_radius(float width, float depth, float2 footprint, float resolution_y) {
    return min(mb_projected(width, depth, footprint), min(resolution_y * 0.2, MAX_RADIUS));
}

#endif
