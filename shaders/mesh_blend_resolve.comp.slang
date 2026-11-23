#include "native.slang"
// Pass 4 (sparse indirect dispatch over active 8x8 tiles only):
// Computes exact deterministic 3D seam blending for full PBR G-buffer surfaces:
// OKLab base color, metallic, facing-safe normalized normal, roughness,
// reflectance (F0/F90 when FRAME_FLAG_REFLECTANCE), rest-space breakup noise
// for swaying foliage, thin-object half-width cap, and nested view isolation.
#include "scene.slang"
#define GBUFFER_ALBEDO SLOT_PASS1
#define GBUFFER_NORMAL SLOT_PASS2
#include "gbuffer.slang"
#include "pixel.slang"
#include "mesh_blend_common.slang"

[[vk::binding(SLOT_PARAMS, 0)]] ConstantBuffer<Mesh_Blend_Params> params : register(MIRAI_REGISTER(b, SLOT_PARAMS), space0);
[[vk::binding(SLOT_PASS0, 0)]] Texture2D<float4> linear_depth : register(MIRAI_REGISTER(t, SLOT_PASS0), space0);
[[vk::binding(SLOT_PASS3, 0)]] Texture2D<uint4> jfa_in : register(MIRAI_REGISTER(t, SLOT_PASS3), space0);
[[vk::binding(SLOT_PASS4, 0)]] StructuredBuffer<uint> work : register(MIRAI_REGISTER(t, SLOT_PASS4), space0);
[[vk::binding(SLOT_PASS5, 0)]] [[vk::image_format("rgba32ui")]] RWTexture2D<uint4> surface_out : register(MIRAI_REGISTER(u, SLOT_PASS5), space0);

bool inside_view(int2 p) { return all(p >= int2(0)) && all(p < int2(frame.resolution)); }

uint info_at(int2 p) {
    if (!inside_view(p)) return 0u;
    return work[params.info_offset + uint(p.y) * uint(frame.resolution.x) + uint(p.x)];
}

float3 pos_at(int2 p, float depth) {
    float2 uv = (float2(p) + 0.5) * frame.inv_resolution;
    return pixel_at(p, uv, depth).position;
}

uint hash_u(uint v) {
    v = (v ^ (v >> 16)) * 0x7feb352du;
    v = (v ^ (v >> 15)) * 0x846ca68bu;
    return v ^ (v >> 16);
}
float unit_u(uint v) { return float(hash_u(v) >> 8) * (1.0 / 16777216.0); }
float cell_noise(int3 p) { return unit_u(uint(p.x) * 1973u ^ uint(p.y) * 9277u ^ uint(p.z) * 26699u); }
float value_noise(int3 base, float3 p) {
    int3 cell = base + int3(floor(p));
    float3 t = frac(p);
    t = t * t * (3.0 - 2.0 * t);
    float a = lerp(cell_noise(cell), cell_noise(cell + int3(1, 0, 0)), t.x);
    float b = lerp(cell_noise(cell + int3(0, 1, 0)), cell_noise(cell + int3(1, 1, 0)), t.x);
    float c = lerp(cell_noise(cell + int3(0, 0, 1)), cell_noise(cell + int3(1, 0, 1)), t.x);
    float d = lerp(cell_noise(cell + int3(0, 1, 1)), cell_noise(cell + int3(1, 1, 1)), t.x);
    return lerp(lerp(a, b, t.y), lerp(c, d, t.y), t.z);
}

float3 to_oklab(float3 c) {
    float3 lms = float3(dot(c, float3(0.4122214708, 0.5363325363, 0.0514459929)),
                        dot(c, float3(0.2119034982, 0.6806995451, 0.1073969566)),
                        dot(c, float3(0.0883024619, 0.2817188376, 0.6299787005)));
    lms = pow(max(lms, float3(0.0)), float3(1.0 / 3.0));
    return float3(dot(lms, float3(0.2104542553, 0.7936177850, -0.0040720468)),
                  dot(lms, float3(1.9779984951, -2.4285922050, 0.4505937099)),
                  dot(lms, float3(0.0259040371, 0.7827717662, -0.8086757660)));
}
float3 from_oklab(float3 c) {
    float3 lms = float3(c.x + 0.3963377774 * c.y + 0.2158037573 * c.z,
                        c.x - 0.1055613458 * c.y - 0.0638541728 * c.z,
                        c.x - 0.0894841775 * c.y - 1.2914855480 * c.z);
    lms = lms * lms * lms;
    return saturate(float3(dot(lms, float3(4.0767416621, -3.3077115913, 0.2309699292)),
                           dot(lms, float3(-1.2684380046, 2.6097574011, -0.3413193965)),
                           dot(lms, float3(-0.0041960863, -0.7034186147, 1.7076147010))));
}

// Brings a blended shading normal that turned behind the viewer back to the silhouette (as raster.slang).
float3 facing_normal(float3 normal, float3 to_camera) {
    float facing = dot(normal, to_camera);
    return facing < 0.0 ? normalize(normal - to_camera * (facing - 1e-3)) : normal;
}

struct Seed_Candidate {
    int2 own_p;
    int2 other_p;
    uint other_info;
    float dist2;
};

struct Side_Sample {
    float3 albedo;
    float metallic;
    float3 normal;
    float roughness;
    float3 f0;
    float f90;
    float weight;
};

void consider_candidate(
    int2 p, uint self_info, uint seed,
    inout Seed_Candidate first, inout Seed_Candidate second
) {
    if (seed == SEED_EMPTY) return;
    int2 p0 = seed_pos(seed);
    int2 p1 = seed_partner(seed);
    uint i0 = info_at(p0);
    uint i1 = info_at(p1);
    int2 own_p, other_p;
    uint other_info;
    if (info_same_group(i0, self_info) && info_other_group(self_info, i1)) {
        own_p = p0; other_p = p1; other_info = i1;
    } else if (info_same_group(i1, self_info) && info_other_group(self_info, i0)) {
        own_p = p1; other_p = p0; other_info = i0;
    } else {
        return;
    }
    float2 delta = float2(own_p - p);
    float d2 = dot(delta, delta);
    uint other_group = info_group(other_info);
    if (first.other_info != 0u && info_group(first.other_info) == other_group) {
        if (d2 < first.dist2) { first.own_p = own_p; first.other_p = other_p; first.other_info = other_info; first.dist2 = d2; }
        return;
    }
    if (second.other_info != 0u && info_group(second.other_info) == other_group) {
        if (d2 < second.dist2) { second.own_p = own_p; second.other_p = other_p; second.other_info = other_info; second.dist2 = d2; }
        if (second.dist2 < first.dist2) {
            Seed_Candidate tmp = first; first = second; second = tmp;
        }
        return;
    }
    if (first.other_info == 0u || d2 < first.dist2) {
        second = first;
        first.own_p = own_p; first.other_p = other_p; first.other_info = other_info; first.dist2 = d2;
    } else if (second.other_info == 0u || d2 < second.dist2) {
        second.own_p = own_p; second.other_p = other_p; second.other_info = other_info; second.dist2 = d2;
    }
}

// Thin-object radius along the direction away from the seam using the 4-byte pixel_info buffer
// (zero reads of instances[] or values[]). A thin object blends over at most half of itself.
float object_radius(int2 p, uint self_info, float2 away, float dist_to_seam, float radius) {
    float previous = 0.0;
    for (float d = 2.0; d <= radius; d *= 2.0) {
        int2 q = int2(floor(float2(p) + 0.5 + away * d));
        if (!inside_view(q)) return radius;
        if (!info_same_group(info_at(q), self_info)) {
            float lo = previous, hi = d;
            [unroll]
            for (uint s = 0u; s < 3u; s++) {
                float mid = (lo + hi) * 0.5;
                int2 mq = int2(floor(float2(p) + 0.5 + away * mid));
                if (info_same_group(info_at(mq), self_info)) lo = mid;
                else hi = mid;
            }
            return min(radius, (dist_to_seam + hi) * 0.5);
        }
        previous = d;
    }
    return radius;
}

float fade_at(float dist, float width, float contact) {
    float t = dist / max(width, 1e-4);
    if (t >= 0.9) return 0.0;
    float u = saturate(1.0 - t * (1.0 / 0.9));
    // Firm near the seam for seamless contact, tighter quadratic falloff outward.
    return 0.5 * lerp(u * u, u, saturate(u - 0.75)) * contact;
}

// 3D angle factor between self (A), seam (B) and opposite surface (C):
// 0 for steep corners (<= 120 deg), rising smoothly to 1 for coplanar joins (~180 deg).
float coplanar_factor(float3 a, float3 b, float3 c) {
    float3 ba = a - b;
    float3 bc = c - b;
    float la = dot(ba, ba);
    float lc = dot(bc, bc);
    if (la < 1e-10 || lc < 1e-10) return 1.0;
    float cos_theta = dot(ba, bc) * rsqrt(la * lc);
    float x = 1.0 - saturate((-cos_theta - 0.5) * 2.0);
    float x2 = x * x;
    return 1.0 - x2 * x2 * x;
}

// Picks a texture-preserving sample pixel mirrored across the seam onto the other
// surface, falling back smoothly toward edge_p if the mirrored point leaves the group
// or crosses a depth discontinuity.
int2 mirrored_other_pixel(
    int2 p, int2 edge_p, int2 step_dir, uint target_info, float3 edge_pos, float max_jump
) {
    int2 delta = edge_p - p;
    int2 candidates[3] = {
        edge_p + delta,
        edge_p + (delta / 2),
        edge_p + step_dir * max(1, min(int(length(float2(delta)) * 0.25), 3)),
    };
    [unroll]
    for (int i = 0; i < 3; i++) {
        int2 q = candidates[i];
        if (!info_same_group(info_at(q), target_info)) continue;
        float dq = linear_depth.Load(int3(q, 0)).r;
        if (depth_empty(dq)) continue;
        if (length(pos_at(q, dq) - edge_pos) <= max_jump) return q;
    }
    return edge_p;
}

// Reads full PBR G-buffer properties at a valid pixel of the target group with weight w.
void accumulate_surface(
    int2 q, uint target_info, bool has_reflectance, float w,
    inout float3 albedo, inout float metallic, inout float3 normal, inout float roughness,
    inout float3 f0, inout float f90, inout float weight_sum
) {
    if (!info_same_group(info_at(q), target_info)) return;
    float4 a = gbuffer_albedo_target.Load(int3(q, 0));
    float4 n = gbuffer_normal_roughness(q);
    albedo += a.rgb * w;
    metallic += a.a * w;
    normal += n.xyz * w;
    roughness += n.w * w;
    if (has_reflectance) {
        float4 spec = pixel_reflectance(q, a, true);
        f0 += spec.rgb * w;
        f90 += spec.a * w;
    }
    weight_sum += w;
}

Side_Sample evaluate_side(
    int2 p, uint self_info, float self_depth, float3 self_pos, float3 rest_pos,
    Seed_Candidate cand, bool has_reflectance
) {
    Side_Sample res;
    res.weight = 0.0;
    res.albedo = float3(0.0);
    res.metallic = 0.0;
    res.normal = float3(0.0);
    res.roughness = 0.0;
    res.f0 = float3(0.0);
    res.f90 = 0.0;
    if (cand.other_info == 0u) return res;

    float own_w = min(info_width(self_info), info_width(cand.other_info));
    float self_d = depth_distance(self_depth);
    float width = mb_seam_width(own_w, self_d, frame.footprint, frame.resolution.y, params.min_pixels, params.max_widen);
    float radius = mb_seam_radius(width, self_d, frame.footprint, frame.resolution.y);
    float dist_px = sqrt(cand.dist2);
    if (dist_px > radius + 1.5) return res;

    // Smooth 3D seam contact & position using G-buffer surface normals (rotationally
    // invariant along curved staircase edges — eliminates radial Voronoi streaks).
    int2 near_p = cand.own_p;
    int2 edge_p = cand.other_p;
    float near_d = linear_depth.Load(int3(near_p, 0)).r;
    float edge_d = linear_depth.Load(int3(edge_p, 0)).r;
    if (depth_empty(near_d) || depth_empty(edge_d)) return res;

    float3 near_pos = pos_at(near_p, near_d);
    float3 edge_pos = pos_at(edge_p, edge_d);
    float3 boundary = edge_pos - near_pos;
    float footprint = frame.footprint.x + frame.footprint.y * max(depth_distance(near_d), depth_distance(edge_d));
    float3 near_n = gbuffer_normal(near_p);
    float3 edge_n = gbuffer_normal(edge_p);
    // At grazing angles a pixel covers footprint / |dot(N, V)| along the surface.
    float2 near_uv = (float2(near_p) + 0.5) * frame.inv_resolution;
    float3 near_view = pixel_at(near_p, near_uv, near_d).view;
    float grazing = max(max(abs(dot(near_n, near_view)), abs(dot(edge_n, near_view))), 0.15);
    float surface_footprint = footprint / grazing;

    // Distance along each surface normal (plane-to-point) is ~0 at true intersections
    // and large across depth occlusions, independent of horizontal vs vertical pixel steps.
    float plane_sep = min(abs(dot(boundary, near_n)), abs(dot(boundary, edge_n)));
    float max_gap = min(max(surface_footprint * 3.0, width * 0.15), width);
    float contact = saturate(2.0 * (1.0 - length(boundary) / max_gap)) *
                    saturate(2.0 * (1.0 - plane_sep / max(footprint * 2.5, width * 0.12)));
    if (contact <= 1e-4) return res;

    float3 seam = (near_pos + edge_pos) * 0.5;
    // Project onto local seam tangent segment when a neighboring seam pixel exists,
    // turning discrete staircase seeds into a smooth continuous curve.
    int2 step_dir = edge_p - near_p;
    int2 tangent = int2(-step_dir.y, step_dir.x);
    float dist3d = length(self_pos - seam);
    [unroll]
    for (int side = -1; side <= 1; side += 2) {
        int2 n2 = near_p + tangent * side;
        int2 e2 = edge_p + tangent * side;
        if (!info_same_group(info_at(n2), self_info) || !info_same_group(info_at(e2), cand.other_info)) continue;
        float nd2 = linear_depth.Load(int3(n2, 0)).r;
        float ed2 = linear_depth.Load(int3(e2, 0)).r;
        if (depth_empty(nd2) || depth_empty(ed2)) continue;
        float3 s2 = (pos_at(n2, nd2) + pos_at(e2, ed2)) * 0.5;
        float3 seg = s2 - seam;
        float seg2 = dot(seg, seg);
        if (seg2 > footprint * footprint * 0.25 && seg2 < width * width) {
            float t = clamp(dot(self_pos - seam, seg) / seg2, -1.0, 1.0);
            dist3d = min(dist3d, length(self_pos - (seam + seg * t)));
        }
    }
    if (fade_at(dist3d, width, contact) <= 1e-4) return res;

    // Mirrored point on the other side of the seam for both 3D slope-angle narrowing
    // and crisp texture-preserving surface sampling.
    float max_jump = width * 1.5 + surface_footprint * 4.0;
    int2 sample_p = mirrored_other_pixel(p, edge_p, step_dir, cand.other_info, edge_pos, max_jump);
    float sample_d = linear_depth.Load(int3(sample_p, 0)).r;
    float3 sample_pos = depth_empty(sample_d) ? edge_pos : pos_at(sample_p, sample_d);

    // Narrow steep (~90 deg) intersections (e.g. vertical rock meeting flat ground)
    // while keeping coplanar joins at full width.
    float adj_slope = lerp(1.0, params.slope, saturate((own_w - 0.05) * 20.0));
    float flat = coplanar_factor(self_pos, seam, sample_pos);
    width /= lerp(adj_slope, 1.0, flat);
    if (fade_at(dist3d, width, contact) <= 1e-4) return res;

    // Thin-object half-width cap along the direction away from the seam.
    float2 from_other = float2(p - edge_p);
    float from_len = length(from_other);
    float2 away = from_len > 1e-3 ? from_other / from_len : float2(-step_dir);
    width *= saturate(object_radius(p, self_info, away, max(dist_px, 1.0), radius) / max(radius, 1.0));
    float fade = fade_at(dist3d, width, contact);
    if (fade <= 1e-4) return res;

    // Subtractive 2-octave world-space FBM breakup anchored to rest position (stable on
    // wind-swaying foliage): preserves solid contact right at the seam while breaking
    // the outer boundary into organic tongues instead of a uniform soft halo.
    if (params.noise > 0.0) {
        float scale = 2.0 / max(width * params.noise_scale, 1e-3);
        int3 whole0, whole1;
        float3 part0, part1;
        origin_lattice(frame.origin.xyz, float3(scale), whole0, part0);
        origin_lattice(frame.origin.xyz, float3(scale * 2.13), whole1, part1);
        float n = 0.65 * value_noise(whole0, part0 + rest_pos * scale) +
                  0.35 * value_noise(whole1, part1 + rest_pos * (scale * 2.13));
        float center_guard = saturate(fade * 2.0 - 0.5);
        float n_cut = saturate(n - center_guard);
        float edge_ramp = smoothstep(0.0, 0.25, dist3d / max(width, 1e-4));
        fade = lerp(fade, max(fade - n_cut * 0.65, 0.0), params.noise * edge_ramp);
        if (fade <= 1e-4) return res;
    }

    // Texture-preserving sampling: primarily the mirrored pixel on the other surface
    // so grain and normal-map detail carry across cleanly without smearing edge_p.
    float3 sum_albedo = float3(0.0), sum_normal = float3(0.0), sum_f0 = float3(0.0);
    float sum_metallic = 0.0, sum_rough = 0.0, sum_f90 = 0.0, weight_sum = 0.0;
    accumulate_surface(sample_p, cand.other_info, has_reflectance, 1.0, sum_albedo, sum_metallic, sum_normal, sum_rough, sum_f0, sum_f90, weight_sum);
    if (any(sample_p != edge_p) && dist_px < 3.0) {
        float edge_w = 0.35 * (1.0 - dist_px / 3.0);
        accumulate_surface(edge_p, cand.other_info, has_reflectance, edge_w, sum_albedo, sum_metallic, sum_normal, sum_rough, sum_f0, sum_f90, weight_sum);
    }
    if (weight_sum <= 0.0) return res;

    float inv = 1.0 / weight_sum;
    res.albedo = sum_albedo * inv;
    res.metallic = sum_metallic * inv;
    res.normal = dot(sum_normal, sum_normal) > 1e-8 ? normalize(sum_normal) : float3(0.0, 1.0, 0.0);
    res.roughness = sum_rough * inv;
    res.f0 = sum_f0 * inv;
    res.f90 = sum_f90 * inv;
    res.weight = clamp(fade, 0.0, 0.5);
    return res;
}

[shader("compute")]
[numthreads(BLEND_TILE, BLEND_TILE, 1)]
void main(uint3 group_id : SV_GroupID, uint3 local_id : SV_GroupThreadID) {
    uint tile_packed = work[WORK_HEADER_WORDS + group_id.x];
    int2 tile = int2(int(tile_packed & 0xffffu), int(tile_packed >> 16u));
    int2 p = tile * int(BLEND_TILE) + int2(local_id.xy);
    if (!inside_view(p)) return;

    uint self_info = info_at(p);
    if (!info_active(self_info)) {
        surface_out[p] = uint4(0u);
        return;
    }

    // Gather up to 2 distinct-group closest full-res seam seeds from the 2x2 half-res JFA cells
    // around p plus immediate 4-neighbors (guarantees 1px precision on thin blades & 3-way corners).
    Seed_Candidate first, second;
    first.other_info = second.other_info = 0u;
    first.dist2 = second.dist2 = 1e30;

    const int2 dirs[4] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1)};
    [unroll]
    for (uint d = 0u; d < 4u; d++) {
        uint nb_info = info_at(p + dirs[d]);
        if (info_other_group(self_info, nb_info)) {
            consider_candidate(p, self_info, pack_seed(p, d), first, second);
        }
    }

    int2 half_res = (int2(frame.resolution) + int2(1, 1)) >> 1;
    int2 h0 = p >> 1;
    int2 step_h = int2((p.x & 1) == 0 ? -1 : 1, (p.y & 1) == 0 ? -1 : 1);
    [unroll]
    for (int hy = 0; hy < 2; hy++) {
        [unroll]
        for (int hx = 0; hx < 2; hx++) {
            int2 h = h0 + int2(hx == 0 ? 0 : step_h.x, hy == 0 ? 0 : step_h.y);
            if (any(h < int2(0)) || any(h >= half_res)) continue;
            uint4 v = jfa_in.Load(int3(h, 0));
            consider_candidate(p, self_info, v.x, first, second);
            consider_candidate(p, self_info, v.z, first, second);
        }
    }

    if (first.other_info == 0u) {
        surface_out[p] = uint4(0u);
        return;
    }

    float self_depth = linear_depth.Load(int3(p, 0)).r;
    if (depth_empty(self_depth)) {
        surface_out[p] = uint4(0u);
        return;
    }

    float2 uv = (float2(p) + 0.5) * frame.inv_resolution;
    Pixel self_px = pixel_at(p, uv, self_depth);
    float3 rest_pos = self_px.position - pixel_rest(p, self_depth);
    bool has_reflectance = (frame.flags & FRAME_FLAG_REFLECTANCE) != 0u;

    Side_Sample a = evaluate_side(p, self_info, self_depth, self_px.position, rest_pos, first, has_reflectance);
    Side_Sample b = evaluate_side(p, self_info, self_depth, self_px.position, rest_pos, second, has_reflectance);
    if (a.weight + b.weight <= 1e-4) {
        surface_out[p] = uint4(0u);
        return;
    }

    float wa = a.weight / (1.0 - a.weight);
    float wb = b.weight / (1.0 - b.weight);
    float inv_w = 1.0 / (1.0 + wa + wb);

    // Right on the 1-pixel rasterized seam edge, pull self one pixel back inside the
    // same surface (when continuous) to avoid rasterization aliasing on the seam line.
    int2 self_sample_p = p;
    if (first.dist2 <= 1.5 && a.weight > 0.35) {
        int2 back_p = p - (first.other_p - first.own_p);
        if (info_same_group(info_at(back_p), self_info)) {
            float bd = linear_depth.Load(int3(back_p, 0)).r;
            if (!depth_empty(bd) && abs(depth_distance(bd) - depth_distance(self_depth)) < info_width(self_info) * 0.5) {
                self_sample_p = back_p;
            }
        }
    }

    float4 self_albedo = gbuffer_albedo_target.Load(int3(self_sample_p, 0));
    float4 self_normal_texel = gbuffer_normal_target.Load(int3(self_sample_p, 0));
    float3 self_normal = gbuffer_normal_unpack(self_normal_texel);
    float self_rough = self_normal_texel.z;
    uint self_bits = gbuffer_bits_unpack(gbuffer_normal_target.Load(int3(p, 0)));

    float3 oklab = to_oklab(self_albedo.rgb) + to_oklab(a.albedo) * wa;
    if (wb > 0.0) oklab += to_oklab(b.albedo) * wb;
    float3 mixed_albedo = from_oklab(oklab * inv_w);
    float mixed_metallic = saturate((self_albedo.a + a.metallic * wa + b.metallic * wb) * inv_w);

    float3 raw_normal = self_normal + a.normal * wa + b.normal * wb;
    float3 bent = dot(raw_normal, raw_normal) > 1e-8 ? normalize(raw_normal) : self_normal;
    bent = facing_normal(bent, -self_px.view);
    float mixed_rough = saturate((self_rough + a.roughness * wa + b.roughness * wb) * inv_w);

    uint packed_refl = 0u;
    if (has_reflectance) {
        float4 self_spec = pixel_reflectance(self_sample_p, self_albedo, true);
        float3 mixed_f0 = (self_spec.rgb + a.f0 * wa + b.f0 * wb) * inv_w;
        float mixed_f90 = (self_spec.a + a.f90 * wa + b.f90 * wb) * inv_w;
        float4 spec_out = specular_pack(mixed_f0, mixed_f90);
        packed_refl = pack_srgb_rgba8(spec_out.rgb, spec_out.a);
    }

    float2 oct = oct_encode(bent) * 0.5 + 0.5;
    uint packed_albedo = pack_srgb_rgba8(mixed_albedo, mixed_metallic);
    uint packed_oct = pack_unorm16(oct.x) | (pack_unorm16(oct.y) << 16u);
    uint packed_rough_bits = pack_unorm16(mixed_rough) | (((self_bits & 0x7fffu) | 0x8000u) << 16u);
    surface_out[p] = uint4(packed_albedo, packed_oct, packed_rough_bits, packed_refl);
}
