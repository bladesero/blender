/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once
#include "draw_view.bsl.hh"
#include "eevee_bxdf_types.bsl.hh"
#include "eevee_colorspace_lib.bsl.hh"
#include "eevee_debug_shared.hh"
#include "eevee_gbuffer_read.bsl.hh"
#include "eevee_light_lib.bsl.hh"
#include "eevee_lightprobe.bsl.hh"
#include "eevee_ray_trace_screen_lib.bsl.hh"
#include "eevee_raytrace_shared.hh"
#include "eevee_reverse_z_lib.bsl.hh"
#include "eevee_sampling_lib.bsl.hh"
#include "eevee_uniform.bsl.hh"
#include "gpu_shader_common_math.glsl"
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
  [[storage(16, read)]] const HardwareRTUVAttribute (&rt_uv_attributes)[];
  [[storage(17, read)]] const HardwareRTMaterial (&rt_closures)[];
  [[sampler(20)]] sampler2DArray rt_images0;
  [[sampler(21)]] sampler2DArray rt_images1;
  [[sampler(22)]] sampler2DArray rt_images2;
  [[sampler(23)]] sampler2DArray rt_images3;
  [[sampler(24)]] sampler2DArray rt_images4;
  [[sampler(25)]] sampler2DArray rt_images5;
  [[sampler(26)]] sampler2DArray rt_images6;
  [[sampler(27)]] sampler2DArray rt_images7;

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
  float3 transmission;
  float roughness;
  float weight;
  bool supported;
};

struct HitContext {
  float3 P, V, N, Ng;
  float4x4 world_to_object;
  float2 uv;
  uint attribute_offset;
  float3 barycentric;
  bool front_face;
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
  int3 coordinate = int3(p, int(info.x & 0xffffffu));
  switch (info.x >> 24u) {
    case 0u:
      return texelFetch(scene.rt_images0, coordinate, 0);
    case 1u:
      return texelFetch(scene.rt_images1, coordinate, 0);
    case 2u:
      return texelFetch(scene.rt_images2, coordinate, 0);
    case 3u:
      return texelFetch(scene.rt_images3, coordinate, 0);
    case 4u:
      return texelFetch(scene.rt_images4, coordinate, 0);
    case 5u:
      return texelFetch(scene.rt_images5, coordinate, 0);
    case 6u:
      return texelFetch(scene.rt_images6, coordinate, 0);
    case 7u:
      return texelFetch(scene.rt_images7, coordinate, 0);
  }
  return float4(0);
}

float4 material_parameter_leaf([[resource_table]] const Scene &scene,
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
    value.rgb *= value.a;
  }
  if ((info.w & 32u) != 0u && value.a != 0.0f && value.a != 1.0f) {
    value.rgb /= value.a;
  }
  if ((info.w & 64u) != 0u) {
    value.a = 1.0f;
  }
  return value;
}

float4 material_parameter([[resource_table]] const Scene &scene,
                          HardwareRTParameter param,
                          HitContext context)
{
  if (param.texture.y == 0u) {
    return param.value;
  }
  float2 uv = float2(0.0f);
  if (param.mapping.z == 0u) {
    HardwareRTUVAttribute attr =
        scene.rt_uv_attributes[context.attribute_offset + param.mapping.x];
    float3 b = context.barycentric;
    uv = attr.uv01.xy * b.x + attr.uv01.zw * b.y + attr.uv2.xy * b.z;
  }
  float4 coordinate = float4(uv, 0.0f, 1.0f);
  uv = float2(dot(param.uv_row0, coordinate), dot(param.uv_row1, coordinate));
  float4 value = param.color_transform * material_parameter_leaf(scene, param, uv) + param.value;
  return param.mapping.y != 0u ? saturate(value) : value;
}

HardwareRTMaterial hit_material_data([[resource_table]] const Scene &scene, GPURayQueryHit hit)
{
  HardwareRTInstance instance = scene.rt_instances[hit.instance];
  HardwareRTTriangle tri = scene.rt_triangles[instance.identity.y + hit.primitive];
  HardwareRTMaterial material =
      scene.rt_materials[scene.rt_material_map[instance.identity.w + tri.material.x]];
  if (tri.material.y != 0u) {
    material.properties.z = 0.0f;
  }
  return material;
}

HitMaterial hit_closure([[resource_table]] const Scene &scene,
                        GPURayQueryHit hit,
                        HardwareRTMaterial data,
                        float3 P,
                        float3 V)
{
  HardwareRTInstance instance = scene.rt_instances[hit.instance];
  HardwareRTTriangle tri = scene.rt_triangles[instance.identity.y + hit.primitive];
  float3 b = float3(1.0f - hit.barycentric.x - hit.barycentric.y, hit.barycentric);
  float2 uv = float2(dot(b, float3(tri.corner0.w, tri.corner1.w, tri.corner2.w)),
                     dot(b, tri.uv_y.xyz));
  HitMaterial result;
  float3x3 normal_matrix = transpose(to_float3x3(instance.world_to_object));
  float face_sign = hit.front_face ? 1.0f : -1.0f;
  float3 Ni = (normalize(normal_matrix * tri.corner0.xyz) * b.x +
               normalize(normal_matrix * tri.corner1.xyz) * b.y +
               normalize(normal_matrix * tri.corner2.xyz) * b.z) *
              face_sign;
  result.N = normalize(Ni);
  HitContext context;
  context.P = P;
  context.V = V;
  context.N = result.N;
  context.Ng = scene.normal(hit) * face_sign;
  context.world_to_object = instance.world_to_object;
  context.uv = uv;
  context.attribute_offset = tri.attributes.x;
  context.barycentric = b;
  context.front_face = hit.front_face;
  /* Fixed-size descriptors share the same bounded texture sampling path. */
  float4 parameters[9];
  for (uint parameter_index = 0u; parameter_index < 9u; parameter_index++) {
    HardwareRTParameter parameter;
    switch (parameter_index) {
      case 0u:
        parameter = data.weight;
        break;
      case 1u:
        parameter = data.base_color;
        break;
      case 2u:
        parameter = data.roughness;
        break;
      case 3u:
        parameter = data.metallic;
        break;
      case 4u:
        parameter = data.emission;
        break;
      case 5u:
        parameter = data.emission_strength;
        break;
      case 6u:
        parameter = data.normal_color;
        break;
      case 7u:
        parameter = data.normal_strength;
        break;
      default:
        parameter = data.alpha;
        break;
    }
    parameters[parameter_index] = material_parameter(scene, parameter, context);
  }
  if (data.normal_info.x != 0u) {
    float3 encoded = parameters[6].rgb;
    float3 mapped = 2.0f * encoded - 1.0f;
    /* Match Blender's legacy object/world normal-map convention exactly. */
    if (data.normal_info.y == 3u || data.normal_info.y == 4u) {
      mapped = float3(2.0f, -2.0f, -2.0f) * encoded - 1.0f;
    }
    if (data.normal_info.z != 0u) {
      mapped.y = -mapped.y;
    }
    float strength = parameters[7].x;
    if (data.normal_info.y == 0u) {
      HardwareRTUVAttribute attr = scene.rt_uv_attributes[tri.attributes.x + data.normal_info.w];
      float4 tangent = attr.tangent0 * b.x + attr.tangent1 * b.y + attr.tangent2 * b.z;
      /* Raster attributes normalize each corner before interpolation, including nonuniform scale.
       */
      float3 T = normalize(normal_matrix * attr.tangent0.xyz) * b.x +
                 normalize(normal_matrix * attr.tangent1.xyz) * b.y +
                 normalize(normal_matrix * attr.tangent2.xyz) * b.z;
      T *= face_sign;
      float3 B = tangent.w * face_sign * cross(Ni, T);
      B *= determinant(to_float3x3(instance.world_to_object)) < 0.0f ? -1.0f : 1.0f;
      mapped.xy *= strength;
      mapped.z = mix(1.0f, mapped.z, saturate(strength));
      mapped = mapped.x * T + mapped.y * B + mapped.z * Ni;
    }
    else {
      if (data.normal_info.y == 1u || data.normal_info.y == 3u) {
        mapped = transpose(to_float3x3(instance.world_to_object)) * mapped;
      }
      mapped = mix(result.N, mapped, max(0.0f, strength));
    }
    if (dot(mapped, mapped) > 1e-12f && !any(isnan(mapped)) && !any(isinf(mapped))) {
      result.N = normalize(mapped);
    }
  }
  float3 base = clamp(parameters[1].rgb, 0.0f, 1.0f);
  float metallic = saturate(parameters[3].x);
  float eta = max(1.0f, data.properties.x);
  float dielectric = square((eta - 1.0f) / (eta + 1.0f)) * data.properties.y * 2.0f;
  result.f0 = data.properties.w > 0.0f ? float3(0.0f) :
                                         mix(float3(saturate(dielectric)), base, metallic);
  result.diffuse = base * (1.0f - metallic) * (1.0f - result.f0);
  result.roughness = clamp(parameters[2].x, 0.02f, 1.0f);
  result.emission = max(float3(0.0f), parameters[4].rgb * parameters[5].x);
  result.supported = data.properties.z > 0.0f;
  const float alpha = saturate(parameters[8].x);
  const float weight = saturate(parameters[0].x);
  result.transmission = weight * (data.properties.w == 2.0f ? base : float3(1.0f - alpha));
  result.weight = data.properties.w == 2.0f ? 0.0f : weight * alpha;
  return result;
}

HitMaterial hit_material([[resource_table]] const Scene &scene,
                         GPURayQueryHit hit,
                         float3 P,
                         float3 V)
{
  HardwareRTMaterial root = hit_material_data(scene, hit);
  HitMaterial result = hit_closure(scene, hit, root, P, V);
  result.diffuse *= result.weight;
  result.emission *= result.weight;
  for (uint i = 1u; i < root.closures.y; i++) {
    HitMaterial leaf = hit_closure(scene, hit, scene.rt_closures[root.closures.x + i], P, V);
    result.diffuse += leaf.weight * leaf.diffuse;
    result.emission += leaf.weight * leaf.emission;
    result.transmission += leaf.transmission;
  }
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

/* Negative visibility requests native shading for this query, not an opaque blocker. */
float3 shadow_visibility(
    [[resource_table]] const Scene &scene, float3 P, float3 N, LightData light, LightVector lv)
{
  if (light.tilemap_index == LIGHT_NO_SHADOW) {
    return float3(1.0f);
  }
  float bias = ray_bias(P);
  float3 origin = P + N * bias;
  float distance = is_sun_light(light.type) ? 1e6f : max(bias, lv.dist - bias);
  float tmin = bias;
  float3 transmission = float3(1.0f);
  for (int i = 0; i < 64; i++) {
    GPURayQueryHit blocker = gpu_ray_query(scene.rt_scene, origin, lv.L, tmin, distance);
    if (!blocker.valid) {
      return transmission;
    }
    uint4 linking = scene.rt_instances[blocker.instance].linking;
    if (linking.z == 0u && set_member(light.shadow_set_membership, linking.y)) {
      if (hit_material_data(scene, blocker).properties.z == 0.0f) {
        return float3(-1.0f);
      }
      HitMaterial material = hit_material(scene, blocker, origin + blocker.distance * lv.L, -lv.L);
      transmission *= saturate(material.transmission);
      if (reduce_max(transmission) < 1e-5f) {
        return float3(0.0f);
      }
    }
    tmin = blocker.distance + bias;
    if (tmin >= distance) {
      return transmission;
    }
  }
  return float3(-1.0f);
}

float3 closure_direct_radiance([[resource_table]] const Scene &scene,
                               [[resource_table]] const Lighting &lighting,
                               [[resource_table]] const UtilityTexture &utility,
                               GPURayQueryHit hit,
                               HitMaterial mat,
                               float3 P,
                               float3 V,
                               bool diffuse_only)
{
  float3 N = mat.N;
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
    float3 visibility = shadow_visibility(scene, P, Ng, light, lv);
    if (any(lessThan(visibility, float3(0.0f)))) {
      return float3(-1.0f);
    }
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
    radiance += light.color * attenuation * visibility * reflected;
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
  float3 transmission = float3(1.0f);
  if (distance > 2.0f * bias) {
    float tmin = bias;
    bool complete = false;
    for (int layer = 0; layer < 64; layer++) {
      GPURayQueryHit hit = gpu_ray_query(
          scene.rt_scene, P + N * bias, delta / distance, tmin, distance - bias);
      if (!hit.valid) {
        complete = true;
        break;
      }
      if (hit_material_data(scene, hit).properties.z == 0.0f) {
        return float4(0.0f); /* Unknown visibility uses the native probe fallback. */
      }
      HitMaterial material = hit_material(
          scene, hit, P + N * bias + delta * (hit.distance / distance), -delta / distance);
      transmission *= saturate(material.transmission);
      if (reduce_max(transmission) < 1e-5f) {
        return float4(0.0f);
      }
      tmin = hit.distance + bias;
      if (tmin >= distance - bias) {
        complete = true;
        break;
      }
    }
    if (!complete) {
      return float4(0.0f);
    }
  }
  return float4(cache.rt_irradiance[best].rgb * transmission, cache.rt_irradiance[best].w);
}

float3 material_radiance([[resource_table]] const Scene &scene,
                         [[resource_table]] const Lighting &lighting,
                         [[resource_table]] const UtilityTexture &utility,
                         [[resource_table]] const IrradianceRead &cache,
                         [[resource_table]] const LightprobeRenderData &lightprobes,
                         GPURayQueryHit hit,
                         float3 P,
                         float3 V,
                         float2 noise,
                         bool diffuse_only,
                         bool use_probe_fallback)
{
  HardwareRTMaterial root = hit_material_data(scene, hit);
  float3 radiance = float3(0.0f);
  for (uint i = 0u; i < root.closures.y; i++) {
    HitMaterial leaf = hit_closure(scene, hit, scene.rt_closures[root.closures.x + i], P, V);
    if (leaf.weight == 0.0f) {
      continue;
    }
    float3 direct = closure_direct_radiance(
        scene, lighting, utility, hit, leaf, P, V, diffuse_only);
    if (any(lessThan(direct, float3(0.0f)))) {
      return float3(-1.0f);
    }
    radiance += leaf.weight * direct;
    float3 N = dot(leaf.N, V) >= 0.0f ? leaf.N : -leaf.N;
    float4 E = irradiance_lookup(scene, cache, P, N);
    if (E.w > 0.0f) {
      radiance += leaf.weight * leaf.diffuse * E.rgb * (1.0f / 3.14159265359f);
    }
    else if (use_probe_fallback) {
      LightProbeSample samp = lightprobes.load(noise, P, N, V);
      radiance += leaf.weight * leaf.diffuse * lightprobes.eval_direction(samp, P, N, 1.0f);
    }
  }
  return radiance;
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
    float3 radiance = float3(0.0f), throughput = float3(1.0f);
    float tmin = bias;
    bool complete = false;
    for (uint layer = 0u; layer < 64u; layer++) {
      GPURayQueryHit hit = gpu_ray_query(scene.rt_scene, origin, direction, tmin, 1e6f);
      if (!hit.valid || hit_material_data(scene, hit).properties.z == 0.0f) {
        LightProbeSample samp = lightprobes.load(float2(index, sample_index), P, N, N);
        radiance += throughput * lightprobes.eval_direction(samp, P, direction, 0.0f);
        complete = true;
        break;
      }
      float3 hit_P = origin + hit.distance * direction;
      HitMaterial material = hit_material(scene, hit, hit_P, -direction);
      float3 surface = material_radiance(scene,
                                         lighting,
                                         utility,
                                         cache,
                                         lightprobes,
                                         hit,
                                         hit_P,
                                         -direction,
                                         float2(index, sample_index),
                                         true,
                                         false);
      if (any(lessThan(surface, float3(0.0f)))) {
        LightProbeSample samp = lightprobes.load(float2(index, sample_index), P, N, N);
        radiance += throughput * lightprobes.eval_direction(samp, P, direction, 0.0f);
        complete = true;
        break;
      }
      radiance += throughput * surface;
      throughput *= saturate(material.transmission);
      if (reduce_max(throughput) < 1e-5f) {
        complete = true;
        break;
      }
      tmin = hit.distance + ray_bias(hit_P);
    }
    if (!complete) {
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
  [[sampler(28)]] sampler2D radiance_front_tx;
  [[image(0, read, SFLOAT_16_16_16_16)]] const image2D ray_data_img;
  [[image(1, write, RAYTRACE_RAYTIME_FORMAT)]] image2D ray_time_img;
  [[image(2, write, RAYTRACE_RADIANCE_FORMAT)]] image2D ray_radiance_img;
  [[image(3, write, UINT_8)]] uimage2D ray_path_img;
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
           [[resource_table]] const Sampling &sampling,
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
  /* Classify the actual raster surface. A material or coverage miss retains native tracing. */
  float3 camera_origin = view.point_screen_to_world(float3(uv, 0.0f));
  float primary_distance = length(P - camera_origin);
  GPURayQueryHit primary = gpu_ray_query(scene.rt_scene,
                                         camera_origin,
                                         normalize(P - camera_origin),
                                         0.0f,
                                         primary_distance + bias * 2.0f);
  if (!primary.valid || abs(primary.distance - primary_distance) > bias * 2.0f ||
      hit_material_data(scene, primary).properties.z == 0.0f)
  {
    return;
  }
  float roughness = closure_apparent_roughness_get(cl);
  float3 radiance = float3(0.0f), throughput = float3(1.0f);
  float hit_time = 10000.0f;
  float tmin = bias;
  bool terminated = false;
  uint resolved_path = 1u;
  for (uint layer = 0u; layer < 64u; layer++) {
    GPURayQueryHit hit = gpu_ray_query(scene.rt_scene, origin, direction, tmin, max_distance);
    if (!hit.valid) {
      LightProbeSample samp = lightprobes.load(float2(texel), P, direction, V);
      samp.volume_irradiance = spherical_harmonics::clamp_energy(
          samp.volume_irradiance, uni.uniform_buf.clamp.surface_indirect);
      radiance += throughput * lightprobes.eval_direction(samp, P, direction, roughness);
      terminated = true;
      break;
    }
    HardwareRTMaterial data = hit_material_data(scene, hit);
    float3 hit_P = origin + hit.distance * direction;
    if (data.properties.z == 0.0f) {
      /* Keep the first unsupported surface as the blocker. Re-running an unconstrained
       *
       * screen ray can see past off-screen geometry and pick up an unrelated surface. */
      float3 ndc = view.point_world_to_ndc(hit_P);
      float2 hit_uv = ndc.xy * 0.5f + 0.5f;
      float2 history_ndc = project_point(uni.raytrace_buf.history_persmat, hit_P).xy;
      bool matched = all(lessThan(abs(ndc.xy), float2(1.0f))) &&
                     all(lessThan(abs(history_ndc), float2(1.0f))) &&
                     transform_point(view.viewmat, hit_P).z < 0.0f;
      if (matched) {
        int2 pixel = int2(hit_uv * uni.raytrace_buf.full_resolution);
        float sample_depth = reverse_z::read(texelFetch(srt.depth_tx, pixel, 0).r);
        float3 sample_P = view.point_screen_to_world(float3(hit_uv, sample_depth));
        float3 near_P = view.point_screen_to_world(float3(hit_uv, 0.0f));
        float tolerance = max(
            ray_bias(hit_P) * 4.0f,
            length(view.point_screen_to_world(float3(
                       hit_uv + float2(uni.raytrace_buf.full_resolution_inv.x, 0), sample_depth)) -
                   sample_P) *
                2.0f);
        GPURayQueryHit visible = gpu_ray_query(scene.rt_scene,
                                               near_P,
                                               normalize(sample_P - near_P),
                                               0.0f,
                                               length(sample_P - near_P) + tolerance);
        matched = length(sample_P - hit_P) <= tolerance && visible.valid;
        if (matched) {
          matched = visible.instance == hit.instance &&
                    abs(visible.distance - length(sample_P - near_P)) <= tolerance &&
                    dot(scene.normal(visible), scene.normal(hit)) > 0.95f;
        }
      }
      float3 native_radiance;
      if (matched) {
        ScreenTraceHitData screen_hit;
        screen_hit.time = length(hit_P - P);
        screen_hit.v_hit_P = transform_point(view.viewmat, hit_P);
        native_radiance = raytrace_sample_screen(view,
                                                 srt.radiance_front_tx,
                                                 uni.raytrace_buf,
                                                 screen_hit,
                                                 roughness,
                                                 history_ndc * 0.5f + 0.5f);
      }
      else {
        LightProbeSample samp = lightprobes.load(float2(texel), P, direction, V);
        samp.volume_irradiance = spherical_harmonics::clamp_energy(
            samp.volume_irradiance, uni.uniform_buf.clamp.surface_indirect);
        native_radiance = lightprobes.eval_direction(samp, P, direction, roughness);
      }
      radiance += throughput * native_radiance;
      hit_time = matched ? length(hit_P - P) : 10000.0f;
      /* Native screen / off-screen probe: both keep denoising and no RR hit distance. */
      resolved_path = matched ? 2u : 3u;
      terminated = true;
      break;
    }
    HitMaterial material = hit_material(scene, hit, hit_P, -direction);
    float3 transmission = saturate(material.transmission);
    float continuation = reduce_max(transmission);
    bool covered = true;
    float surface_probability = 1.0f;
    if (data.capability.y != 0u && continuation > 0.0f) {
      float random = fract(sampling.rng_1D_get(SAMPLING_RAYTRACE_W) +
                           float(random_hash(uint(texel.x) ^ (uint(texel.y) << 16u) ^ layer)) *
                               (1.0f / 4294967296.0f));
      covered = random >= continuation;
      surface_probability = max(1e-6f, 1.0f - continuation);
    }
    if (covered) {
      float3 surface = material_radiance(scene,
                                         lighting,
                                         utility,
                                         irradiance,
                                         lightprobes,
                                         hit,
                                         hit_P,
                                         -direction,
                                         float2(texel),
                                         false,
                                         true);
      if (any(lessThan(surface, float3(0.0f)))) {
        return;
      }
      if (hit_time == 10000.0f && reduce_max(surface) > 0.0f) {
        hit_time = length(hit_P - P);
      }
      radiance += throughput * surface / surface_probability;
    }
    if (data.capability.y != 0u) {
      if (covered) {
        terminated = true;
        break;
      }
      throughput *= transmission / max(continuation, 1e-6f);
    }
    else {
      throughput *= transmission;
    }
    if (reduce_max(throughput) < 1e-5f) {
      terminated = true;
      break;
    }
    tmin = hit.distance + ray_bias(hit_P);
  }
  if (!terminated) {
    return; /* Traversal budget exhausted: screen/probe fallback for this ray. */
  }
  imageStore(srt.ray_path_img, texel, uint4(resolved_path));
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
    color = srt.rt_debug_mode == int(DEBUG_HARDWARE_RT_NORMAL_ERROR) ? float3(0) : float3(1, 0, 0);
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
      else if (srt.rt_debug_mode == int(DEBUG_HARDWARE_RT_SHADING_NORMAL)) {
        color = hit_material(scene, hit, P, normalize(origin - P)).N * 0.5f + 0.5f;
      }
      else if (srt.rt_debug_mode == int(DEBUG_HARDWARE_RT_DIFFUSE_ALBEDO)) {
        color = hit_material(scene, hit, P, normalize(origin - P)).diffuse;
      }
      else if (srt.rt_debug_mode == int(DEBUG_HARDWARE_RT_DIFFUSE_ERROR)) {
        float3 reference = float3(0.0f);
        gbuffer::Layers layers = reader.read_layers(texel);
        for (int i = 0; i < 3; i++) {
          if (layers.layer[i].type == CLOSURE_BSDF_DIFFUSE_ID) {
            reference += layers.layer[i].color;
          }
        }
        color = abs(hit_material(scene, hit, P, normalize(origin - P)).diffuse - reference);
      }
      else if (srt.rt_debug_mode == int(DEBUG_HARDWARE_RT_NORMAL_ERROR)) {
        color = abs(hit_material(scene, hit, P, normalize(origin - P)).N -
                    reader.read_layers(texel).surface_N());
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
  imageStore(srt.rt_debug_img, texel, float4(color, depth < 1.0f ? 0.0f : 1.0f));
}
}  // namespace eevee::hardware

PipelineCompute eevee_hardware_radiance_capture(eevee::hardware::capture);
PipelineCompute eevee_hardware_cache_select(eevee::hardware::select);
PipelineCompute eevee_hardware_cache_resolve(eevee::hardware::resolve);
PipelineCompute eevee_hardware_trace(eevee::hardware::trace);
PipelineCompute eevee_hardware_debug(eevee::hardware::debug_view);

PipelineCompute eevee_hardware_irradiance_update(eevee::hardware::irradiance_update);
