//! mirai needs instance_ids
#include "native.slang"
// Where the surfaces of two groups of meshes meet (meshblend.mesh_blend_record),
// before they are lit: each pixel of a participating surface looks around it
// for a surface of another group, finds the seam between them by bisection
// along two directions, checks from the depth that both surfaces touch there
// and, within the width of the seam from the line through both seam points,
// takes the base color, the normal and the roughness of the other side. It
// writes the blended surface for mesh_blend_apply.frag, or nothing where the
// pixel keeps its own.
#include "scene.slang"
#define GBUFFER_ALBEDO SLOT_PASS1
#define GBUFFER_NORMAL SLOT_PASS2
#include "gbuffer.slang"
#include "pixel.slang"

struct Mesh_Blend_Params {
    float noise;
    float noise_scale;
    float min_pixels; // Mesh_Blend_Settings.min_pixels, in pixels of an image MIN_PIXELS_HEIGHT high
    float max_widen;  // at least 1
};
[[vk::binding(SLOT_PARAMS, 0)]] ConstantBuffer<Mesh_Blend_Params> params : register(MIRAI_REGISTER(b, SLOT_PARAMS), space0);
[[vk::binding(SLOT_PASS0, 0)]] Texture2D<float4> linear_depth : register(MIRAI_REGISTER(t, SLOT_PASS0), space0);
[[vk::binding(SLOT_PASS3, 0)]] Texture2D<uint> instance_ids : register(MIRAI_REGISTER(t, SLOT_PASS3), space0);
[[vk::binding(SLOT_PASS4, 0)]] [[vk::image_format("rg32ui")]] RWTexture2D<uint2> blended : register(MIRAI_REGISTER(u, SLOT_PASS4), space0);

static const float MIN_PIXELS_HEIGHT = 1080.0;

// A surface of the G-buffer as the blend reads it.
struct Sample {
    uint id;      // the instance + 1; 0: none
    float4 data;  // its values: width, group (low and high bits), participates
    float depth;  // linear
    float3 position;
};

// The values of an instance (meshblend.value_of).
float4 instance_data(uint id) { return id == 0u ? float4(0.0) : instance_values(id - 1u); }

bool participant(uint id, float4 data) { return id != 0u && data.w > 0.5 && data.x > 0.0; }
bool same_group(float4 a, float4 b) { return all(a.yz == b.yz); }

bool inside(int2 p) { return all(p >= int2(0)) && all(p < int2(frame.resolution)); }

int2 offset_pixel(int2 p, float2 offset) { return int2(floor(float2(p) + 0.5 + offset)); }

// The instance the id target holds at a pixel, before the depth says whether
// this view drew it: the sky clears it, but a nested view leaves its own.
uint id_drawn(int2 p) { return inside(p) ? instance_ids.Load(int3(p, 0)) : 0u; }

bool drawn_here(float depth) { return !(depth_empty(depth) || depth_nested(depth)); }
bool drawn_here_at(int2 p) { return drawn_here(linear_depth.Load(int3(p, 0)).r); }

Sample sample_at(int2 p) {
    Sample s;
    s.depth = linear_depth.Load(int3(p, 0)).r;
    uint id = id_drawn(p);
    s.id = id != 0u && drawn_here(s.depth) ? id : 0u;
    s.data = instance_data(s.id);
    float2 uv = (float2(p) + 0.5) * frame.inv_resolution;
    s.position = pixel_at(p, uv, s.depth).position;
    return s;
}

// Pixels a length in metres spans at a depth.
float projected(float width, float depth) { return width / max(frame.footprint.x + frame.footprint.y * depth, 1e-6); }

// The width in metres a seam blends over at a depth: far away it widens to
// stay min_pixels wide on the screen, up to max_widen times its own.
float seam_width(float width, float depth) {
    float wanted = params.min_pixels * frame.resolution.y / MIN_PIXELS_HEIGHT;
    return width * clamp(wanted / max(projected(width, depth), 1e-6), 1.0, params.max_widen);
}

// Pixels the search reaches for a width: at most a fifth of the screen.
float seam_radius(float width, float depth) { return min(projected(width, depth), min(frame.resolution.y * 0.2, 256.0)); }

struct Candidate {
    float2 direction;
    float distance;
    uint id;
    float4 data;
};

struct Side {
    float3 albedo;
    float3 normal;
    float roughness;
    float weight;
};

uint hash(uint v) {
    v = (v ^ (v >> 16)) * 0x7feb352du;
    v = (v ^ (v >> 15)) * 0x846ca68bu;
    return v ^ (v >> 16);
}
float unit(uint v) { return float(hash(v) >> 8) * (1.0 / 16777216.0); }
float cell_noise(int3 p) { return unit(uint(p.x) * 1973u ^ uint(p.y) * 9277u ^ uint(p.z) * 26699u); }
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

// Two numbers per pixel and frame for the search: the R2 sequence over the
// screen gives neighbors values far apart, as blue noise does, and each frame
// moves them along.
float2 search_noise(int2 p) {
    float2 place = float2(p);
    float2 spread = float2(dot(place, float2(0.7548776662, 0.5698402910)), dot(place, float2(0.5698402910, 0.7548776662)));
    return frac(spread + float(frame.frame_index & 4095u) * float2(0.6180339887, 0.4142135624));
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

void consider(int2 p, Sample self, float2 direction, float distance, inout Candidate first, inout Candidate second) {
    int2 q = offset_pixel(p, direction * distance);
    uint id = id_drawn(q);
    if (id == 0u || id == self.id) return;
    float4 data = instance_data(id);
    if (!participant(id, data) || same_group(data, self.data)) return;
    if (distance > max(projected(seam_width(min(self.data.x, data.x), self.depth), self.depth), 1.0)) return;
    if (!drawn_here_at(q)) return;
    Candidate value;
    value.direction = direction;
    value.distance = distance;
    value.id = id;
    value.data = data;
    if (first.id != 0u && same_group(data, first.data)) {
        if (distance < first.distance) first = value;
    } else if (second.id != 0u && same_group(data, second.data)) {
        if (distance < second.distance) second = value;
    } else if (first.id == 0u || distance < first.distance) {
        second = first;
        first = value;
    } else if (second.id == 0u || distance < second.distance) {
        second = value;
    }
    if (second.id != 0u && second.distance < first.distance) {
        Candidate previous = first;
        first = second;
        second = previous;
    }
}

bool group_at(int2 p, float4 group) {
    uint id = id_drawn(p);
    if (id == 0u) return false;
    float4 data = instance_data(id);
    return participant(id, data) && same_group(data, group) && drawn_here_at(p);
}

// The seam along a direction, between lo on the own group and hi on the
// other: false where the ray crosses something of neither. The samples at
// the seam check the depth of what it finds.
bool bisect(int2 p, float2 direction, float4 own, float4 other, inout float lo, inout float hi) {
    for (uint step = 0u; step < 8u; step++) {
        if (hi - lo <= 0.5) break;
        float middle = (lo + hi) * 0.5;
        uint id = id_drawn(offset_pixel(p, direction * middle));
        if (id == 0u) return false;
        float4 data = instance_data(id);
        if (!participant(id, data)) return false;
        if (same_group(data, other)) hi = middle;
        else if (same_group(data, own)) lo = middle;
        else return false;
    }
    return true;
}

// The point of the seam between the last pixel of the own group at lo and the
// first of the other at hi along a direction, and how surely the two surfaces
// touch there: 1 while they meet, down to 0 as they come apart in depth, where
// one only covers the other on the screen. middle: halfway between the two
// pixels, before the surfaces correct it.
bool seam_at(int2 p, float2 direction, float lo, float hi, Sample self, Candidate candidate, float width, out float3 seam, out float3 middle, out float contact) {
    seam = middle = float3(0.0);
    contact = 0.0;
    int2 near_p = offset_pixel(p, direction * lo);
    int2 edge_p = offset_pixel(p, direction * hi);
    Sample near_s = sample_at(near_p);
    Sample edge_s = sample_at(edge_p);
    if (near_s.id == 0u || edge_s.id == 0u) return false;
    middle = (near_s.position + edge_s.position) * 0.5;
    float footprint = frame.footprint.x + frame.footprint.y * max(near_s.depth, edge_s.depth);
    // At a grazing angle a continuous surface spans more of the world per pixel.
    float surface_footprint = footprint;
    float2 boundary_pixels = float2(edge_p - near_p);
    float pixels = max(length(boundary_pixels), 1.0);
    int2 source_p = offset_pixel(near_p, -boundary_pixels);
    int2 target_p = offset_pixel(edge_p, boundary_pixels);
    float3 source_tangent = float3(0.0), target_tangent = float3(0.0);
    bool source_known = group_at(source_p, self.data);
    bool target_known = group_at(target_p, candidate.data);
    if (source_known) {
        source_tangent = near_s.position - sample_at(source_p).position;
        surface_footprint = max(surface_footprint, length(source_tangent) / pixels);
    }
    if (target_known) {
        target_tangent = sample_at(target_p).position - edge_s.position;
        surface_footprint = max(surface_footprint, length(target_tangent) / pixels);
    }
    float3 boundary = edge_s.position - near_s.position;
    // Full up to half of how far apart they may be, none past it.
    contact = saturate(2.0 * (1.0 - length(boundary) / min(max(surface_footprint * 3.0, width * 0.1), width)));
    seam = middle;
    if (source_known && target_known) {
        float3 difference = source_tangent - target_tangent;
        float difference2 = dot(difference, difference);
        float amount = difference2 > footprint * footprint * 1e-4 ?
                       saturate(dot(boundary - target_tangent, difference) / max(difference2, 1e-12)) : 0.5;
        float separation = length(boundary - lerp(target_tangent, source_tangent, amount));
        contact *= saturate(2.0 * (1.0 - separation / max(footprint * 0.5, width * 0.1)));
        seam = (near_s.position + source_tangent * amount + edge_s.position - target_tangent * (1.0 - amount)) * 0.5;
    }
    return contact > 0.0;
}

// How far the pixel is from the seam: from the line through the middle of the
// seam and a second middle along a direction turned 30 degrees from the first,
// so the distance does not depend on the direction the search took; from the
// seam point alone where no second one is found.
float seam_distance(int2 p, Sample self, Candidate candidate, float hi, float width, float3 seam, float3 middle) {
    float distance = length(self.position - seam);
    float2 d = candidate.direction;
    // Turned toward the seam, the ray crosses it no farther than the first one did.
    float reach = hi * 1.25 + 1.0;
    for (uint k = 0u; k < 2u; k++) {
        float turn = k == 0u ? 0.5 : -0.5;
        float2 direction = float2(d.x * 0.8660254038 - d.y * turn, d.x * turn + d.y * 0.8660254038);
        if (!group_at(offset_pixel(p, direction * reach), candidate.data)) continue;
        float lo = 0.0, far = reach;
        if (!bisect(p, direction, self.data, candidate.data, lo, far)) return distance;
        Sample near_s = sample_at(offset_pixel(p, direction * lo));
        Sample edge_s = sample_at(offset_pixel(p, direction * far));
        float footprint = frame.footprint.x + frame.footprint.y * max(near_s.depth, edge_s.depth);
        // Where they do not meet in depth the second middle says nothing of the seam.
        if (near_s.id == 0u || edge_s.id == 0u || length(edge_s.position - near_s.position) > min(max(footprint * 3.0, width * 0.1), width)) return distance;
        float3 along = (near_s.position + edge_s.position) * 0.5 - middle;
        float span = length(along);
        if (span < 2.0 * footprint) return distance;
        return min(distance, length(cross(self.position - middle, along / span)));
    }
    return distance;
}

// How far the surface of the pixel reaches away from the seam: a thin object
// blends over no more than half of itself.
float object_radius(int2 p, Sample self, Candidate candidate, float radius) {
    float previous = 0.0;
    for (float distance = 2.0; distance <= radius; distance *= 2.0) {
        int2 q = offset_pixel(p, -candidate.direction * distance);
        if (!inside(q)) return radius;
        if (!group_at(q, self.data)) {
            float lo = previous, hi = distance;
            for (uint step = 0u; step < 3u; step++) {
                float middle = (lo + hi) * 0.5;
                if (group_at(offset_pixel(p, -candidate.direction * middle), self.data)) lo = middle;
                else hi = middle;
            }
            return min(radius, (candidate.distance + hi) * 0.5);
        }
        previous = distance;
    }
    return radius;
}

void surface_of(int2 p, out float3 albedo, out float3 normal, out float roughness) {
    albedo = gbuffer_albedo_target.Load(int3(p, 0)).rgb;
    float4 shading = gbuffer_normal_roughness(p);
    normal = shading.xyz;
    roughness = shading.w;
}

float fade_at(float distance, float width, float contact) { return 0.5 * pow(saturate(1.0 - distance / max(width, 1e-4)), 2.0) * contact; }

// refine: the distance to the seam is measured to its line, not to its point
// (only the nearest of the two candidates, the one that weighs most).
Side side_of(int2 p, Sample self, Candidate candidate, bool refine) {
    Side result;
    result.weight = 0.0;
    result.albedo = float3(0.0);
    result.normal = float3(0.0);
    result.roughness = 0.0;
    if (candidate.id == 0u) return result;
    float own = min(self.data.x, candidate.data.x);
    float width = seam_width(own, self.depth);
    float radius = seam_radius(width, self.depth);
    float lo = 0.0, hi = candidate.distance;
    if (!bisect(p, candidate.direction, self.data, candidate.data, lo, hi)) return result;
    float3 seam, middle;
    float contact;
    if (!seam_at(p, candidate.direction, lo, hi, self, candidate, width, seam, middle, contact)) return result;
    float distance = refine ? seam_distance(p, self, candidate, hi, width, seam, middle) : length(self.position - seam);
    // A thin object only narrows the blend: past the full width there is nothing to measure.
    if (fade_at(distance, width, contact) <= 1e-4) return result;
    candidate.distance = hi;
    width *= saturate(object_radius(p, self, candidate, radius) / max(radius, 1.0));
    float fade = fade_at(distance, width, contact);
    if (fade <= 1e-4) return result;
    if (params.noise > 0.0) {
        // On the lattice of the world, as fine as the width of the seam before it widens far away.
        float scale = 1.0 / max(own * params.noise_scale, 1e-3);
        int3 whole;
        float3 part;
        origin_lattice(frame.origin.xyz, float3(scale), whole, part);
        float n = value_noise(whole, part + self.position * scale);
        fade *= lerp(1.0, 0.35 + 0.65 * n, params.noise * smoothstep(0.0, 0.2, distance / max(width, 1e-4)));
    }
    int2 edge_p = offset_pixel(p, candidate.direction * hi);
    // The other side a little past the seam, away from what the seam itself drew.
    int2 interior_p = offset_pixel(p, candidate.direction * (hi + max(1.0, min(hi * 0.1, 4.0))));
    if (!group_at(interior_p, candidate.data)) interior_p = edge_p;
    float3 albedo_a, normal_a, albedo_b, normal_b;
    float roughness_a, roughness_b;
    surface_of(edge_p, albedo_a, normal_a, roughness_a);
    surface_of(interior_p, albedo_b, normal_b, roughness_b);
    result.albedo = (albedo_a + albedo_b) * 0.5;
    result.normal = normal_a + normal_b;
    result.roughness = (roughness_a + roughness_b) * 0.5;
    result.weight = clamp(fade, 0.0, 0.5);
    return result;
}

uint pack_unorm(float value, uint bits) { return uint(round(saturate(value) * float((1u << bits) - 1u))); }

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int2 p = int2(id.xy);
    if (!inside(p)) return;
    blended[p] = uint2(0u, 0u);
    Sample self = sample_at(p);
    if (!participant(self.id, self.data)) return;
    Candidate first, second;
    first.id = second.id = 0u;
    first.distance = second.distance = 1e30;
    float radius = seam_radius(seam_width(self.data.x, self.depth), self.depth);
    const float2 close_directions[4] = {float2(1, 0), float2(0, 1), float2(-1, 0), float2(0, -1)};
    for (uint i = 0u; i < 4u; i++) consider(p, self, close_directions[i], 1.0, first, second);
    if (radius > 1.0) {
        // Three directions and the distances of the rings turned by pixel and frame: the temporal resolve averages them.
        float2 noise = search_noise(p);
        float s, c;
        sincos(noise.x * 2.0943951024, s, c);
        float2 directions[3];
        directions[0] = float2(c, s);
        directions[1] = float2(-0.5 * c - 0.8660254038 * s, 0.8660254038 * c - 0.5 * s);
        directions[2] = float2(-0.5 * c + 0.8660254038 * s, -0.8660254038 * c - 0.5 * s);
        for (uint ring = 1u; ring <= 3u; ring++) {
            float t = (float(ring) - noise.y) / 3.0;
            float distance = max(1.0, t * t * t * radius);
            for (uint d = 0u; d < 3u; d++) consider(p, self, directions[d], distance, first, second);
        }
    }
    if (first.id == 0u) return;
    Side a = side_of(p, self, first, true);
    Side b = side_of(p, self, second, false);
    if (a.weight + b.weight <= 1e-4) return;
    float wa = a.weight / (1.0 - a.weight);
    float wb = b.weight / (1.0 - b.weight);
    float3 albedo, normal;
    float roughness;
    surface_of(p, albedo, normal, roughness);
    float3 mixed = from_oklab((to_oklab(albedo) + to_oklab(a.albedo) * wa + to_oklab(b.albedo) * wb) / (1.0 + wa + wb));
    float3 bent = normalize(normal + a.normal * (0.5 * wa) + b.normal * (0.5 * wb));
    float rough = (roughness + a.roughness * wa + b.roughness * wb) / (1.0 + wa + wb);
    // Square roots in 8 bits keep the darks of the base color, which is stored in sRGB.
    float3 coded = sqrt(mixed);
    float2 oct = oct_encode(bent) * 0.5 + 0.5;
    blended[p] = uint2(pack_unorm(coded.r, 8u) | (pack_unorm(coded.g, 8u) << 8u) | (pack_unorm(coded.b, 8u) << 16u) | (255u << 24u),
                       pack_unorm(oct.x, 12u) | (pack_unorm(oct.y, 12u) << 12u) | (pack_unorm(rough, 8u) << 24u));
}
