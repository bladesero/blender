/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once
#include "draw_view.bsl.hh"
#include "eevee_colorspace_lib.bsl.hh"
#include "eevee_debug_shared.hh"
#include "eevee_gbuffer_read.bsl.hh"
#include "eevee_light_lib.bsl.hh"
#include "eevee_lightprobe.bsl.hh"
#include "eevee_raytrace_shared.hh"
#include "eevee_reverse_z_lib.bsl.hh"
#include "eevee_uniform.bsl.hh"
#include "gpu_shader_ray_query_lib.glsl"
#include "gpu_shader_shared_exponent_lib.glsl"

namespace eevee::hardware {

constexpr uint cache_mask = (1u << 20u) - 1u;

uint cache_slot(uint4 key)
{
  uint hash = key.x * 0x9e3779b9u ^ key.y * 0x85ebca6bu ^ key.z * 0xc2b2ae35u ^ key.w;
  hash ^= hash >> 16u;
  hash *= 0x7feb352du;
  hash ^= hash >> 15u;
  return hash & cache_mask;
}

struct Scene {
  [[acceleration_structure(0)]] accelerationStructureEXT rt_scene;
  [[storage(1, read)]] const HardwareRTInstance (&rt_instances)[];
  [[storage(2, read)]] const float4 (&rt_normals)[];

  [[storage(7, read)]] const HardwareRTTriangle (&rt_triangles)[];
  [[storage(8, read)]] const uint (&rt_material_map)[];
  [[storage(9, read)]] const HardwareRTMaterial (&rt_materials)[];
  [[storage(10, read)]] const float4 (&rt_pixels)[];

  uint4 surface_key(GPURayQueryHit hit) const
  {
    uint2 cell = min(uint2(hit.barycentric * 4.0f), uint2(3u));
    uint4 identity = rt_instances[hit.instance].identity;
    return uint4(identity.x, hit.primitive, cell.x + 4u * cell.y, identity.z);
  }
  float3 normal(GPURayQueryHit hit) const
  {
    HardwareRTInstance instance = rt_instances[hit.instance];
    float3 N = rt_normals[instance.identity.y + hit.primitive].xyz;
    return normalize(transpose(to_float3x3(instance.world_to_object)) * N);
  }
};

struct CacheRead {
  [[storage(3, read)]] const HardwareRadianceEntry (&rt_cache)[];
  [[sampler(1)]] sampler2D rt_direct_tx;
};

struct Capture {
  [[acceleration_structure(0)]] accelerationStructureEXT rt_scene;
  [[sampler(0)]] sampler2D rt_emission_tx;
  [[sampler(1)]] usampler2D rt_direct_0_tx;
  [[sampler(2)]] usampler2D rt_direct_1_tx;
  [[sampler(3)]] usampler2D rt_direct_2_tx;
  [[image(0, write, SFLOAT_16_16_16_16)]] image2D rt_direct_img;
};
[[compute, local_size(8, 8)]]
void capture([[resource_table]] Capture &srt,
             [[resource_table]] const gbuffer::Reader &reader,
             [[global_invocation_id]] const uint3 gid)
{
  int2 texel = int2(gid.xy);
  if (any(greaterThanEqual(texel, imageSize(srt.rt_direct_img)))) {
    return;
  }
  const gbuffer::Layers gbuf = reader.read_layers(texel);
  const uint3 bins = gbuf.header.bin_index_per_layer();
  float3 radiance = texelFetch(srt.rt_emission_tx, texel, 0).rgb;
  for (int i = 0; i < 3; i++) [[unroll]] {
    if (i < gbuf.header.closure_len()) {
      uint encoded_radiance = 0u;
      if (bins[i] == 0u) {
        encoded_radiance = texelFetch(srt.rt_direct_0_tx, texel, 0).r;
      }
      if (bins[i] == 1u) {
        encoded_radiance = texelFetch(srt.rt_direct_1_tx, texel, 0).r;
      }
      if (bins[i] == 2u) {
        encoded_radiance = texelFetch(srt.rt_direct_2_tx, texel, 0).r;
      }
      radiance += rgb9e5_decode(encoded_radiance) * gbuf.layer[i].color;
    }
  }
  radiance = any(isnan(radiance)) || any(isinf(radiance)) ? float3(0.0f) : radiance;
  imageStore(srt.rt_direct_img, texel, float4(clamp(radiance, 0.0f, 65000.0f), 1.0f));
}

struct Select {
  [[sampler(0)]] sampler2DDepth rt_depth_tx;
  [[storage(3, read_write)]] uint (&rt_winners)[];
  [[image(0, write, UINT_32_32_32_32)]] uimage2D rt_key_img;
};
[[compute, local_size(8, 8)]]
void select([[resource_table]] Select &srt,
            [[resource_table]] const Scene &scene,
            [[resource_table]] const gbuffer::Reader &reader,
            [[resource_table]] const draw::View &views,
            [[global_invocation_id]] const uint3 gid)
{
  int2 texel = int2(gid.xy);
  int2 extent = imageSize(srt.rt_key_img);
  if (any(greaterThanEqual(texel, extent))) {
    return;
  }
  imageStore(srt.rt_key_img, texel, uint4(0u));
  float depth = reverse_z::read(texelFetch(srt.rt_depth_tx, texel, 0).r);
  if (depth >= 1.0f) {
    return;
  }
  const ViewMatrices view = views.get(0);
  float2 uv = (float2(texel) + 0.5f) / float2(extent);
  float3 P = view.point_screen_to_world(float3(uv, depth));
  float3 origin = view.point_screen_to_world(float3(uv, 0.0f));
  float distance = length(P - origin);
  if (distance <= 0.0f) {
    return;
  }
  float3 direction = (P - origin) / distance;
  float tolerance = max(0.0001f, distance * 0.0001f);
  GPURayQueryHit hit = gpu_ray_query(
      scene.rt_scene, origin, direction, 0.0f, distance + tolerance);
  if (!hit.valid || abs(hit.distance - distance) > tolerance) {
    return;
  }
  const gbuffer::Layers gbuf = reader.read_layers(texel);
  float3 Ng = gbuf.header.geometry_normal(gbuf.surface_N());
  if (abs(dot(scene.normal(hit), Ng)) < 0.9f) {
    return;
  }
  uint4 key = scene.surface_key(hit);
  imageStore(srt.rt_key_img, texel, key);
  /* Winner selection and writes are separate dispatches. No unsynchronized HDR float writes. */
  atomicMax(srt.rt_winners[cache_slot(key)], 1u + gid.y * uint(extent.x) + gid.x);
}

struct Resolve {
  [[acceleration_structure(1)]] accelerationStructureEXT rt_scene;
  [[storage(3, read_write)]] HardwareRadianceEntry (&rt_cache)[];
  [[storage(0, read)]] const uint (&rt_winners)[];
  [[sampler(1)]] sampler2D rt_direct_tx;
  [[image(0, read, UINT_32_32_32_32)]] const uimage2D rt_key_img;
};
[[compute, local_size(8, 8)]]
void resolve([[resource_table]] Resolve &srt, [[global_invocation_id]] const uint3 gid)
{
  int2 texel = int2(gid.xy);
  int2 extent = imageSize(srt.rt_key_img);
  if (any(greaterThanEqual(texel, extent))) {
    return;
  }
  uint4 key = imageLoad(srt.rt_key_img, texel);
  if (key.x == 0u) {
    return;
  }
  uint slot = cache_slot(key);
  if (srt.rt_winners[slot] != 1u + gid.y * uint(extent.x) + gid.x) {
    return;
  }
  srt.rt_cache[slot].key = key;
  srt.rt_cache[slot].radiance = texelFetch(srt.rt_direct_tx, texel, 0);
}

struct HitMaterial {
  float3 N;
  float3 diffuse;
  float3 f0;
  float3 emission;
  float roughness;
  bool supported;
};

int wrap_texel(int v, int size, uint extension)
{
  if (extension == 0u) {
    return (v % size + size) % size;
  }
  if (extension == 3u) {
    int t = (v % (2 * size) + 2 * size) % (2 * size);
    return min(t, 2 * size - 1 - t);
  }
  return clamp(v, 0, size - 1);
}

float4 material_texel([[resource_table]] const Scene &scene, uint4 info, int2 p)
{
  uint extension = info.w & 3u;
  if (extension == 2u && (any(lessThan(p, int2(0))) || any(greaterThanEqual(p, int2(info.yz))))) {
    return float4(0.0f);
  }
  p = int2(wrap_texel(p.x, int(info.y), extension), wrap_texel(p.y, int(info.z), extension));
  return scene.rt_pixels[info.x + uint(p.y) * info.y + uint(p.x)];
}

float4 material_parameter([[resource_table]] const Scene &scene,
                          HardwareRTParameter param,
                          float2 uv)
{
  uint4 info = param.texture;
  if (info.y == 0u) {
    return param.value;
  }
  float2 p = uv * float2(info.yz) - 0.5f;
  int2 i = int2(floor(p));
  float2 f = fract(p);
  float4 value;
  if ((info.w & 4u) != 0u) {
    value = material_texel(scene, info, int2(floor(p + 0.5f)));
  }
  else {
    value = mix(
        mix(material_texel(scene, info, i), material_texel(scene, info, i + int2(1, 0)), f.x),
        mix(material_texel(scene, info, i + int2(0, 1)),
            material_texel(scene, info, i + int2(1, 1)),
            f.x),
        f.y);
  }
  if ((info.w & 8u) != 0u) {
    return float4(value.a);
  }
  if ((info.w & 16u) != 0u) {
    return float4((value.r + value.g + value.b) / 3.0f);
  }
  return value;
}

HitMaterial hit_material([[resource_table]] const Scene &scene, GPURayQueryHit hit)
{
  HardwareRTInstance instance = scene.rt_instances[hit.instance];
  HardwareRTTriangle tri = scene.rt_triangles[instance.identity.y + hit.primitive];
  HardwareRTMaterial data =
      scene.rt_materials[scene.rt_material_map[instance.identity.w + tri.material.x]];
  float3 b = float3(1.0f - hit.barycentric.x - hit.barycentric.y, hit.barycentric);
  float2 uv = float2(dot(b, float3(tri.corner0.w, tri.corner1.w, tri.corner2.w)),
                     dot(b, tri.uv_y.xyz));
  HitMaterial result;
  result.N = normalize(transpose(to_float3x3(instance.world_to_object)) *
                       (tri.corner0.xyz * b.x + tri.corner1.xyz * b.y + tri.corner2.xyz * b.z));
  float3 base = clamp(material_parameter(scene, data.base_color, uv).rgb, 0.0f, 1.0f);
  float metallic = saturate(material_parameter(scene, data.metallic, uv).x);
  float eta = max(1.0f, data.properties.x);
  float dielectric = square((eta - 1.0f) / (eta + 1.0f)) * data.properties.y * 2.0f;
  result.f0 = data.properties.w > 0.0f ? float3(0.0f) :
                                         mix(float3(saturate(dielectric)), base, metallic);
  result.diffuse = base * (1.0f - metallic) * (1.0f - result.f0);
  result.roughness = clamp(material_parameter(scene, data.roughness, uv).x, 0.02f, 1.0f);
  result.emission = max(float3(0.0f),
                        material_parameter(scene, data.emission, uv).rgb *
                            material_parameter(scene, data.emission_strength, uv).x);
  result.supported = data.properties.z > 0.0f;
  return result;
}

struct Lighting {
  [[uniform(7)]] const LightData (&rt_sunlight)[2];
  [[storage(11, read)]] const LightData (&rt_lights)[];
  [[push_constant]] const int rt_light_count;
};

bool set_member(uint2 membership, uint set)
{
  return (membership[set >> 5u] & (1u << (set & 31u))) != 0u;
}

float ray_bias(float3 P)
{
  return max(0.0001f, reduce_max(abs(P)) * 0.000001f);
}

float shadow_visibility(
    [[resource_table]] const Scene &scene, float3 P, float3 N, LightData light, LightVector lv)
{
  if (light.tilemap_index == LIGHT_NO_SHADOW) {
    return 1.0f;
  }
  float bias = ray_bias(P);
  float3 origin = P + N * bias;
  float distance = is_sun_light(light.type) ? 1e6f : max(bias, lv.dist - bias);
  /* Closest opaque queries let shadow linking skip blockers without an any-hit pipeline. */
  float tmin = bias;
  for (int i = 0; i < 64; i++) {
    GPURayQueryHit blocker = gpu_ray_query(scene.rt_scene, origin, lv.L, tmin, distance);
    if (!blocker.valid) {
      return 1.0f;
    }
    uint4 linking = scene.rt_instances[blocker.instance].linking;
    if (linking.z == 0u && set_member(light.shadow_set_membership, linking.y)) {
      return 0.0f;
    }
    tmin = blocker.distance + bias;
    if (tmin >= distance) {
      return 1.0f;
    }
  }
  return 0.0f;
}

float3 direct_radiance([[resource_table]] const Scene &scene,
                       [[resource_table]] const Lighting &lighting,
                       [[resource_table]] const UtilityTexture &utility,
                       GPURayQueryHit hit,
                       HitMaterial mat,
                       float3 P,
                       float3 V,
                       bool diffuse_only)
{
  float3 N = dot(mat.N, V) >= 0.0f ? mat.N : -mat.N;
  float3 Ng = scene.normal(hit);
  Ng = dot(Ng, V) >= 0.0f ? Ng : -Ng;
  float4 spec_ltc = lut::ltc::sample_utility_tx(utility, saturate(dot(N, V)), mat.roughness);
  float3 fresnel = mat.f0 + (1.0f - mat.f0) * pow(1.0f - saturate(dot(N, V)), 5.0f);
  float3 radiance = mat.emission;
  uint receiver = scene.rt_instances[hit.instance].linking.x;
  for (int i = 0; i < lighting.rt_light_count; i++) {
    LightData light = lighting.rt_lights[i];
    if (is_sun_light(light.type) && light.color.r < 0.0f) {
      light.color = lighting.rt_sunlight[i].color;
      LightSunData sun = light.sun();
      sun.direction = lighting.rt_sunlight[i].object_to_world.z_axis();
      light.sun() = sun;
    }
    if (!set_member(light.light_set_membership, receiver)) {
      continue;
    }
    LightVector lv = light_vector_get(light, is_sun_light(light.type), P);
    float attenuation = light_attenuation_surface(light, is_sun_light(light.type), lv) *
                        light_attenuation_facing(light, lv.L, lv.dist, Ng, false);
    if (attenuation <= 1e-6f) {
      continue;
    }
    attenuation *= shadow_visibility(scene, P, Ng, light, lv);
    if (attenuation == 0.0f) {
      continue;
    }
    LightVertices vertices = light_shape_corners(light, lv);
    float diffuse = light_ltc(utility.utility_tx, light, N, V, lv, lut::ltc::identity(), vertices);
    float3 reflected = mat.diffuse * diffuse * light.power[0];
    if (!diffuse_only && any(greaterThan(mat.f0, float3(0.0f)))) {
      reflected += fresnel * light.power[1] *
                   light_ltc(utility.utility_tx, light, N, V, lv, spec_ltc, vertices);
    }
    radiance += light.color * attenuation * reflected;
  }
  return radiance;
}

constexpr uint irradiance_count = 1u << 15u;
uint irradiance_slot(int4 key)
{
  uint h = uint(key.x) * 0x9e3779b9u ^ uint(key.y) * 0x85ebca6bu ^ uint(key.z) * 0xc2b2ae35u ^
           uint(key.w) * 0x27d4eb2du;
  h ^= h >> 16u;
  return h & (irradiance_count - 1u);
}
int normal_octant(float3 N)
{
  return int(N.x >= 0.0f) + 2 * int(N.y >= 0.0f) + 4 * int(N.z >= 0.0f);
}
struct IrradianceRead {
  [[storage(12, read)]] const HardwareIrradianceProbe (&rt_irradiance_probes)[];
  [[storage(13, read)]] const float4 (&rt_irradiance)[];
};

float4 irradiance_lookup([[resource_table]] const Scene &scene,
                         [[resource_table]] const IrradianceRead &cache,
                         float3 P,
                         float3 N)
{
  float spacing = cache.rt_irradiance_probes[0].position.w;
  if (spacing <= 0.0f) {
    return float4(0.0f);
  }
  int3 cell = int3(floor(P / spacing));
  int octant = normal_octant(N);
  float best_distance = spacing * spacing * 3.0f;
  uint best = irradiance_count;
  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        int4 key = int4(cell + int3(x, y, z), octant);
        uint slot = irradiance_slot(key);
        HardwareIrradianceProbe probe = cache.rt_irradiance_probes[slot];
        float3 delta = probe.position.xyz - P;
        float d2 = dot(delta, delta);
        if (probe.normal.w > 0.0f && cache.rt_irradiance[slot].w > 0.0f &&
            all(equal(probe.key, key)) && dot(probe.normal.xyz, N) > 0.95f &&
            abs(dot(delta, N)) < spacing * 0.15f && d2 < best_distance)
        {
          best_distance = d2;
          best = slot;
        }
      }
    }
  }
  if (best == irradiance_count) {
    return float4(0.0f);
  }
  /* Normal/plane tests alone cannot prevent leaking through a thin wall. */
  HardwareIrradianceProbe probe = cache.rt_irradiance_probes[best];
  float bias = max(ray_bias(P), ray_bias(probe.position.xyz));
  float3 delta = probe.position.xyz + probe.normal.xyz * bias - (P + N * bias);
  float distance = length(delta);
  if (distance > 2.0f * bias &&
      gpu_ray_query(scene.rt_scene, P + N * bias, delta / distance, bias, distance - bias).valid)
  {
    return float4(0.0f);
  }
  return cache.rt_irradiance[best];
}

struct IrradianceUpdate {
  [[storage(14, write)]] float4 (&rt_irradiance_next)[];
  [[push_constant]] const int rt_irradiance_frame;
};
uint random_hash(uint x)
{
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  return x ^ (x >> 16u);
}

[[compute, local_size(64)]]
void irradiance_update([[resource_table]] IrradianceUpdate &srt,
                       [[resource_table]] const Scene &scene,
                       [[resource_table]] const Lighting &lighting,
                       [[resource_table]] const UtilityTexture &utility,
                       [[resource_table]] const IrradianceRead &cache,
                       [[resource_table]] const CacheRead &surface_cache,
                       [[resource_table]] const LightprobeRenderData &lightprobes,
                       [[resource_table]] const Uniform &uni,
                       [[global_invocation_id]] const uint3 gid)
{
  uint index = gid.x;
  if (index >= irradiance_count) {
    return;
  }
  float4 old = cache.rt_irradiance[index];
  HardwareIrradianceProbe probe = cache.rt_irradiance_probes[index];
  if (probe.normal.w == 0.0f || (index & 3u) != (uint(srt.rt_irradiance_frame) & 3u)) {
    srt.rt_irradiance_next[index] = old;
    return;
  }
  float3 N = probe.normal.xyz;
  float3 P = probe.position.xyz;
  float3 T, B;
  make_orthonormal_basis(N, T, B);
  float3 sum = float3(0.0f);
  for (uint sample_index = 0u; sample_index < 8u; sample_index++) {
    uint seed = random_hash(index ^
                            random_hash(uint(srt.rt_irradiance_frame) * 8u + sample_index));
    float u = (float(seed & 65535u) + 0.5f) / 65536.0f;
    float phi = (float(seed >> 16u) + 0.5f) * (6.28318530718f / 65536.0f);
    float3 direction = T * (sqrt(u) * cos(phi)) + B * (sqrt(u) * sin(phi)) + N * sqrt(1.0f - u);
    float bias = ray_bias(P);
    float3 origin = P + N * bias;
    GPURayQueryHit hit = gpu_ray_query(scene.rt_scene, origin, direction, bias, 1e6f);
    float3 radiance = float3(0.0f);
    if (hit.valid) {
      float3 hit_P = origin + hit.distance * direction;
      HitMaterial mat = hit_material(scene, hit);
      float3 hit_N = dot(mat.N, direction) < 0.0f ? mat.N : -mat.N;
      if (mat.supported) {
        radiance = direct_radiance(scene, lighting, utility, hit, mat, hit_P, -direction, true);
        /* Read only the previous iteration. E/pi is outgoing Lambertian radiance before albedo. */
        radiance += mat.diffuse * irradiance_lookup(scene, cache, hit_P, hit_N).rgb *
                    (1.0f / 3.14159265359f);
      }
      else {
        uint4 key = scene.surface_key(hit);
        HardwareRadianceEntry entry = surface_cache.rt_cache[cache_slot(key)];
        if (all(equal(entry.key, key)) && entry.radiance.w > 0.0f) {
          radiance = entry.radiance.rgb;
        }
      }
    }
    else {
      LightProbeSample samp = lightprobes.load(float2(index, sample_index), P, N, N);
      radiance = lightprobes.eval_direction(samp, P, direction, 0.0f);
    }
    radiance = colorspace::brightness_clamp_max(radiance, uni.uniform_buf.clamp.surface_indirect);
    sum += any(isnan(radiance)) || any(isinf(radiance)) ? float3(0.0f) :
                                                          max(radiance, float3(0.0f));
  }
  float samples = min(old.w + 1.0f, 32.0f);
  float3 estimate = sum * (3.14159265359f / 8.0f);
  srt.rt_irradiance_next[index] = float4(mix(old.rgb, estimate, 1.0f / samples), samples);
}

struct Trace {
  [[specialization_constant(0)]] int closure_index;
  [[storage(5, read)]] const uint (&tiles_coord_buf)[];
  [[sampler(0)]] sampler2DDepth depth_tx;
  [[image(0, read, SFLOAT_16_16_16_16)]] const image2D ray_data_img;
  [[image(1, write, RAYTRACE_RAYTIME_FORMAT)]] image2D ray_time_img;
  [[image(2, write, RAYTRACE_RADIANCE_FORMAT)]] image2D ray_radiance_img;
};
[[compute, local_size(RAYTRACE_GROUP_SIZE, RAYTRACE_GROUP_SIZE)]]
void trace([[resource_table]] Trace &srt,
           [[resource_table]] const Lighting &lighting,
           [[resource_table]] const UtilityTexture &utility,
           [[resource_table]] const IrradianceRead &irradiance,
           [[resource_table]] const Scene &scene,
           [[resource_table]] const CacheRead &cache,
           [[resource_table]] const gbuffer::Reader &reader,
           [[resource_table]] const LightprobeRenderData &lightprobes,
           [[resource_table]] const Uniform &uni,
           [[resource_table]] const draw::View &views,
           [[work_group_id]] const uint3 group_id,
           [[local_invocation_id]] const uint3 local_id)
{
  uint2 tile = unpackUvec2x16(srt.tiles_coord_buf[group_id.x]);
  int2 texel = int2(tile * RAYTRACE_GROUP_SIZE + local_id.xy);
  if (any(greaterThanEqual(texel, imageSize(srt.ray_data_img)))) {
    return;
  }
  float4 ray = imageLoad(srt.ray_data_img, texel);
  /* Screen pass already resolved invalid rays, transmission and explicit planar probes. */
  if (ray.w <= 0.0f) {
    return;
  }
  int2 fulltexel = texel * uni.raytrace_buf.trace_pixel_scale +
                   uni.raytrace_buf.trace_pixel_offset;
  ClosureUndetermined cl = reader.read_bin(fulltexel, srt.closure_index);
  if (closure_has_transmission(cl.type)) {
    return;
  }
  const ViewMatrices view = views.get(0);
  float depth = reverse_z::read(texelFetch(srt.depth_tx, fulltexel, 0).r);
  float2 uv = (float2(fulltexel) + 0.5f) * uni.raytrace_buf.full_resolution_inv;
  float3 P = view.point_screen_to_world(float3(uv, depth));
  float3 V = view.world_incident_vector(P);
  float3 direction = normalize(ray.xyz);
  float bias = max(0.0001f, length(P - view.position()) * 0.00001f);
  float3 origin = P + direction * bias;
  float max_distance = abs(view.far() - view.near());
  GPURayQueryHit hit = gpu_ray_query(scene.rt_scene, origin, direction, bias, max_distance);
  float roughness = closure_apparent_roughness_get(cl);
  bool found = false;
  float3 radiance = float3(0.0f);
  float hit_time = 10000.0f;
  if (hit.valid && (hit.front_face || uni.raytrace_buf.use_backface_hit)) {
    float3 hit_P = origin + hit.distance * direction;
    hit_time = length(hit_P - P);
    HitMaterial material = hit_material(scene, hit);
    if (material.supported) {
      float3 hit_N = dot(material.N, direction) < 0.0f ? material.N : -material.N;
      radiance = direct_radiance(
          scene, lighting, utility, hit, material, hit_P, -direction, false);
      float4 E = irradiance_lookup(scene, irradiance, hit_P, hit_N);
      if (E.w > 0.0f) {
        radiance += material.diffuse * E.rgb * (1.0f / 3.14159265359f);
      }
      else {
        /* An unallocated, occluded or not-yet-updated cache cell has explicit probe fallback. */
        LightProbeSample samp = lightprobes.load(float2(texel), hit_P, hit_N, -direction);
        radiance += material.diffuse * lightprobes.eval_direction(samp, hit_P, hit_N, 1.0f);
      }
      found = true;
    }
    float3 screen = view.point_world_to_screen(hit_P);
    if (!found && screen.z >= 0.0f && screen.z <= 1.0f &&
        all(greaterThanEqual(screen.xy, float2(0.0f))) && all(lessThan(screen.xy, float2(1.0f))))
    {
      int2 hit_texel = int2(screen.xy * float2(uni.raytrace_buf.full_resolution));
      float raster_depth = reverse_z::read(texelFetch(srt.depth_tx, hit_texel, 0).r);
      float3 raster_P = view.point_screen_to_world(float3(screen.xy, raster_depth));
      float tolerance = max(0.001f, length(hit_P - view.position()) * 0.0001f);
      const gbuffer::Layers hit_gbuf = reader.read_layers(hit_texel);
      float3 Ng = hit_gbuf.header.geometry_normal(hit_gbuf.surface_N());
      if (raster_depth < 1.0f && length(raster_P - hit_P) < tolerance &&
          abs(dot(scene.normal(hit), Ng)) > 0.9f)
      {
        radiance = texelFetch(cache.rt_direct_tx, hit_texel, 0).rgb;
        found = true;
      }
    }
    if (!found) {
      uint4 key = scene.surface_key(hit);
      HardwareRadianceEntry entry = cache.rt_cache[cache_slot(key)];
      if (all(equal(entry.key, key)) && entry.radiance.a > 0.0f) {
        radiance = entry.radiance.rgb;
        found = true;
      }
    }
    if (found && !hit.front_face) {
      radiance *= uni.raytrace_buf.backface_hit_scale;
    }
  }
  if (!found) {
    LightProbeSample samp = lightprobes.load(float2(texel), P, direction, V);
    samp.volume_irradiance = spherical_harmonics::clamp_energy(
        samp.volume_irradiance, uni.uniform_buf.clamp.surface_indirect);
    radiance = lightprobes.eval_direction(samp, P, direction, roughness);
  }
  radiance = colorspace::brightness_clamp_max(radiance, uni.uniform_buf.clamp.surface_indirect);
  radiance = any(isnan(radiance)) || any(isinf(radiance)) ? float3(0.0f) : radiance;
  imageStore(srt.ray_time_img, texel, float4(hit_time));
  imageStore(srt.ray_radiance_img, texel, float4(radiance, 0.0f));
}
struct Debug {
  [[push_constant]] const int rt_debug_mode;
  [[sampler(0)]] sampler2DDepth rt_depth_tx;
  [[image(0, write, SFLOAT_16_16_16_16)]] image2D rt_debug_img;
};
[[compute, local_size(8, 8)]]
void debug_view([[resource_table]] Debug &srt,
                [[resource_table]] const IrradianceRead &irradiance,
                [[resource_table]] const Scene &scene,
                [[resource_table]] const CacheRead &cache,
                [[resource_table]] const gbuffer::Reader &reader,
                [[resource_table]] const draw::View &views,
                [[global_invocation_id]] const uint3 gid)
{
  int2 texel = int2(gid.xy);
  int2 extent = imageSize(srt.rt_debug_img);
  if (any(greaterThanEqual(texel, extent))) {
    return;
  }
  float depth = reverse_z::read(texelFetch(srt.rt_depth_tx, texel, 0).r);
  float3 color = float3(0.0f);
  if (depth < 1.0f) {
    color = float3(1.0f, 0.0f, 0.0f);
    const ViewMatrices view = views.get(0);
    float2 uv = (float2(texel) + 0.5f) / float2(extent);
    float3 P = view.point_screen_to_world(float3(uv, depth));
    float3 origin = view.point_screen_to_world(float3(uv, 0.0f));
    float distance = length(P - origin);
    float tolerance = max(0.0001f, distance * 0.0001f);
    GPURayQueryHit hit = gpu_ray_query(
        scene.rt_scene, origin, normalize(P - origin), 0.0f, distance + tolerance);
    if (hit.valid && abs(hit.distance - distance) < tolerance) {
      if (srt.rt_debug_mode == int(DEBUG_HARDWARE_RT_IRRADIANCE)) {
        float3 N = scene.normal(hit);
        N = dot(N, P - origin) < 0.0f ? N : -N;
        color = irradiance_lookup(scene, irradiance, P, N).rgb * (1.0f / 3.14159265359f);
      }
      else if (srt.rt_debug_mode == int(DEBUG_HARDWARE_RT_NORMAL)) {
        color = scene.normal(hit) * 0.5f + 0.5f;
      }
      else if (srt.rt_debug_mode == int(DEBUG_HARDWARE_RT_DISTANCE)) {
        color = float3(hit.distance / (1.0f + hit.distance));
      }
      else if (srt.rt_debug_mode == int(DEBUG_HARDWARE_RT_INSTANCE)) {
        uint hash = scene.rt_instances[hit.instance].identity.x * 0x9e3779b9u;
        color = float3(
                    float(hash & 255u), float((hash >> 8u) & 255u), float((hash >> 16u) & 255u)) /
                255.0f;
      }
      else {
        uint4 key = scene.surface_key(hit);
        HardwareRadianceEntry entry = cache.rt_cache[cache_slot(key)];
        color = all(equal(entry.key, key)) && entry.radiance.a > 0.0f ? float3(0.0f, 1.0f, 0.0f) :
                                                                        float3(1.0f, 1.0f, 0.0f);
      }
    }
  }
  imageStore(srt.rt_debug_img, texel, float4(color, 1.0f));
}
}  // namespace eevee::hardware

PipelineCompute eevee_hardware_radiance_capture(eevee::hardware::capture);
PipelineCompute eevee_hardware_cache_select(eevee::hardware::select);
PipelineCompute eevee_hardware_cache_resolve(eevee::hardware::resolve);
PipelineCompute eevee_hardware_trace(eevee::hardware::trace);
PipelineCompute eevee_hardware_debug(eevee::hardware::debug_view);

PipelineCompute eevee_hardware_irradiance_update(eevee::hardware::irradiance_update);
