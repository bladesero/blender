/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once
#include "eevee_closure.bsl.hh"
#include "eevee_gbuffer_read.bsl.hh"
#include "eevee_reverse_z_lib.bsl.hh"
#include "eevee_velocity.bsl.hh"
#include "gpu_shader_fullscreen_lib.glsl"
#include "gpu_shader_shared_exponent_lib.glsl"

namespace eevee::dlss {
float reconstruction_weight(float x)
{
  x = abs(x);
  if (x < 1.0f) {
    return (1.5f * x - 2.5f) * x * x + 1.0f;
  }
  if (x < 2.0f) {
    return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
  }
  return 0.0f;
}
float4 reconstruction_color(sampler2D color, float2 coordinate)
{
  int2 size = textureSize(color, 0), p = int2(floor(coordinate));
  float2 f = fract(coordinate);
  float4 result = float4(0);
  for (int y = -1; y <= 2; y++) {
    for (int x = -1; x <= 2; x++) {
      float weight = reconstruction_weight(float(x) - f.x) * reconstruction_weight(float(y) - f.y);
      result += weight * texelFetch(color, clamp(p + int2(x, y), int2(0), size - 1), 0);
    }
  }
  return max(result, float4(0));
}
struct Integrate {
  [[sampler(0)]] sampler2D sample_color;
  [[image(0, read_write, SFLOAT_32_32_32_32)]] image2D integrated_color;
  [[push_constant]] float2 jitter;
  [[push_constant]] float weight;
};
[[compute, local_size(8, 8)]]
void integrate([[resource_table]] Integrate &srt, [[global_invocation_id]] const uint3 id)
{
  int2 p = int2(id.xy);
  if (any(greaterThanEqual(p, imageSize(srt.integrated_color)))) {
    return;
  }
  float4 value = reconstruction_color(srt.sample_color, float2(p) + srt.jitter);
  if (srt.weight < 1.0f) {
    value = mix(imageLoad(srt.integrated_color, p), value, srt.weight);
  }
  imageStore(srt.integrated_color, p, value);
}

struct Prepare {
  [[push_constant]] bool integrated;
  [[push_constant]] float2 jitter;
  [[push_constant]] bool static_frame;
  [[sampler(0)]] sampler2D input_color;
  [[sampler(1)]] sampler2DDepth input_depth;
  [[sampler(2)]] sampler2D input_vector;
  [[image(0, write, SFLOAT_16_16_16_16)]] image2D sr_color;
  [[image(1, write, SFLOAT_32)]] image2D sr_depth;
  [[image(2, write, SFLOAT_32_32)]] image2D sr_motion;
};

[[compute, local_size(8, 8)]]
void prepare([[resource_table]] Prepare &srt,
             [[resource_table]] const CameraVelocity &camera,
             [[resource_table]] const draw::View &views,
             [[global_invocation_id]] const uint3 global_id)
{
  int2 size = textureSize(srt.input_color, 0);
  int2 src = int2(global_id.xy);
  if (any(greaterThanEqual(src, size))) {
    return;
  }
  int2 dst = int2(src.x, size.y - 1 - src.y);
  float depth = reverse_z::read(texelFetch(srt.input_depth, src, 0).r);
  float2 motion = srt.static_frame ? float2(0.0f) :
                                     camera.resolve(views, srt.input_vector, src, depth).xy;
  motion *= float2(size.x, -size.y);
  if (any(isnan(motion)) || any(isinf(motion))) {
    motion = float2(0.0f);
  }
  float3 color = srt.integrated ?
                     reconstruction_color(srt.input_color, float2(src) - srt.jitter).rgb :
                     texelFetch(srt.input_color, src, 0).rgb;
  imageStore(srt.sr_color, dst, float4(color, 1.0f));
  imageStore(srt.sr_depth, dst, float4(depth));
  imageStore(srt.sr_motion, dst, float4(motion, 0.0f, 0.0f));
}

struct SubsurfaceGuide {
  [[sampler(0)]] usampler2D direct_light;
  [[sampler(1)]] sampler2D indirect_light;
  [[image(0, read_write, SFLOAT_32_32_32_32)]] image2D before_light;
  [[image(1, write, SFLOAT_16)]] image2D sss_guide;
  [[push_constant]] bool capture_before;
};
[[compute, local_size(8, 8)]]
void subsurface_guide([[resource_table]] SubsurfaceGuide &srt,
                      [[resource_table]] const gbuffer::Reader &reader,
                      [[global_invocation_id]] const uint3 id)
{
  int2 src = int2(id.xy), size = imageSize(srt.sss_guide);
  if (any(greaterThanEqual(src, size))) {
    return;
  }
  int2 dst = int2(src.x, size.y - 1 - src.y);
  const gbuffer::Layers gbuf = reader.read_layers(src);
  ClosureUndetermined cl = gbuf.layer[0];
  float3 color = float3(0);
  if (cl.type == CLOSURE_BSSRDF_BURLEY_ID) {
    color = rgb9e5_decode(texelFetch(srt.direct_light, src, 0).r) +
            texelFetch(srt.indirect_light, src, 0).rgb;
  }
  if (srt.capture_before) {
    imageStore(srt.before_light, dst, float4(color, 1));
  }
  else {
    float3 change = (color - imageLoad(srt.before_light, dst).rgb) * cl.color;
    float luminance = cl.type == CLOSURE_BSSRDF_BURLEY_ID ?
                          dot(change, float3(0.25f, 0.5f, 0.25f)) :
                          0.0f;
    imageStore(srt.sss_guide, dst, float4(luminance));
  }
}

struct Guides {
  [[image(0, write, SFLOAT_16_16_16_16)]] image2D rr_diffuse;
  [[image(1, write, SFLOAT_16_16_16_16)]] image2D rr_specular;
  [[image(2, write, SFLOAT_16_16_16_16)]] image2D rr_normals;
};
[[compute, local_size(8, 8)]]
void guides([[resource_table]] Guides &srt,
            [[resource_table]] const gbuffer::Reader &reader,
            [[global_invocation_id]] const uint3 id)
{
  int2 size = imageSize(srt.rr_diffuse), src = int2(id.xy);
  if (any(greaterThanEqual(src, size))) {
    return;
  }
  int2 dst = int2(src.x, size.y - 1 - src.y);
  gbuffer::Layers gbuf = reader.read_layers(src);
  float3 diffuse = float3(0.0f), specular = float3(0.0f), N = float3(0.0f, 0.0f, 1.0f);
  float roughness = 1.0f, dominant = -1.0f;
  for (int i = 0; i < 3; i++) {
    ClosureUndetermined cl = gbuf.layer[i];
    if (cl.type == CLOSURE_BSDF_DIFFUSE_ID || cl.type == CLOSURE_BSSRDF_BURLEY_ID ||
        cl.type == CLOSURE_BSDF_TRANSLUCENT_ID)
    {
      diffuse += cl.color;
    }
    if (cl.type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) {
      /* EEVEE's node evaluation already applied its integrated BRDF LUT and closure weights
       * to cl.color. Applying a second Fresnel approximation here would be incorrect. */
      specular += cl.color;
      float weight = dot(cl.color, float3(0.2126f, 0.7152f, 0.0722f));
      if (weight > dominant) {
        dominant = weight;
        N = cl.N;
        roughness = closure_apparent_roughness_get(cl);
      }
    }
    else if ((cl.type == CLOSURE_BSDF_DIFFUSE_ID || cl.type == CLOSURE_BSSRDF_BURLEY_ID ||
              cl.type == CLOSURE_BSDF_TRANSLUCENT_ID) &&
             dominant < 0.0f)
    {
      N = cl.N;
    }
  }
  imageStore(srt.rr_diffuse, dst, float4(max(diffuse, float3(0.0f)), 1.0f));
  imageStore(srt.rr_specular, dst, float4(max(specular, float3(0.0f)), 1.0f));
  imageStore(srt.rr_normals, dst, float4(normalize(N), clamp(roughness, 0.0f, 1.0f)));
}
struct Hit {
  [[sampler(0)]] sampler2D ray_time;
  [[sampler(1)]] usampler2D ray_path;
  [[image(0, write, SFLOAT_32)]] image2D rr_hit;
  [[push_constant]] int closure_index;
};
[[compute, local_size(8, 8)]]
void hit([[resource_table]] Hit &srt,
         [[resource_table]] const gbuffer::Reader &reader,
         [[global_invocation_id]] const uint3 id)
{
  int2 size = imageSize(srt.rr_hit), src = int2(id.xy);
  if (any(greaterThanEqual(src, size))) {
    return;
  }
  ClosureUndetermined cl = reader.read_bin(src, uchar(srt.closure_index));
  if (cl.type != CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) {
    return;
  }
  /* Match the dominant reflection selected by the normal/roughness guide. */
  float weight = dot(cl.color, float3(0.2126f, 0.7152f, 0.0722f));
  for (int i = 0; i < 3; i++) {
    ClosureUndetermined other = reader.read_bin(src, uchar(i));
    if (other.type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID &&
        dot(other.color, float3(0.2126f, 0.7152f, 0.0722f)) > weight)
    {
      return;
    }
  }
  float distance = texelFetch(srt.ray_time, src, 0).r;
  /* EEVEE's environment/probe miss sentinel is not a geometric hit. */
  if (texelFetch(srt.ray_path, src, 0).r != 1u || distance == 10000.0f || distance <= 0.0f ||
      isnan(distance) || isinf(distance))
  {
    distance = 0.0f;
  }
  imageStore(srt.rr_hit, int2(src.x, size.y - 1 - src.y), float4(distance));
}
struct Output {
  [[push_constant]] int2 crop;
  [[sampler(0)]] sampler2D sr_output;
  [[image(0, write, SFLOAT_16_16_16_16)]] image2D final_output;
};
[[compute, local_size(8, 8)]]
void resolve_output([[resource_table]] Output &srt, [[global_invocation_id]] const uint3 id)
{
  int2 size = imageSize(srt.final_output), dst = int2(id.xy);
  if (any(greaterThanEqual(dst, size))) {
    return;
  }
  imageStore(srt.final_output,
             dst,
             float4(texelFetch(srt.sr_output,
                               int2(dst.x + srt.crop.x,
                                    textureSize(srt.sr_output, 0).y - 1 - dst.y - srt.crop.y),
                               0)
                        .rgb,
                    1.0f));
}

struct Blend {
  [[sampler(0)]] sampler2D layer_r;
  [[sampler(1)]] sampler2D layer_g;
  [[sampler(2)]] sampler2D layer_b;
  [[sampler(3)]] sampler2D reconstructed;
  [[image(0, read_write, SFLOAT_32_32_32_32)]] image2D mean_radiance;
  [[image(1, read_write, SFLOAT_32_32_32_32)]] image2D mean_transmittance;
  [[image(2, write, SFLOAT_16_16_16_16)]] image2D composited;
  [[push_constant]] bool colored;
  [[push_constant]] float weight;
};
[[compute, local_size(8, 8)]]
void blend([[resource_table]] Blend &srt, [[global_invocation_id]] const uint3 id)
{
  int2 p = int2(id.xy), size = imageSize(srt.composited);
  if (any(greaterThanEqual(p, size))) {
    return;
  }
  float4 r = texelFetch(srt.layer_r, p, 0);
  float3 L = r.rgb, T = float3(r.a);
  if (srt.colored) {
    float2 g = texelFetch(srt.layer_g, p, 0).rg, b = texelFetch(srt.layer_b, p, 0).rg;
    L = float3(r.r, g.r, b.r);
    T = float3(r.g, g.g, b.g);
  }
  if (srt.weight < 1.0f) {
    L = mix(imageLoad(srt.mean_radiance, p).rgb, L, srt.weight);
    T = mix(imageLoad(srt.mean_transmittance, p).rgb, T, srt.weight);
  }
  imageStore(srt.mean_radiance, p, float4(L, 0));
  imageStore(srt.mean_transmittance, p, float4(T, 0));
  int2 dst = int2(p.x, size.y - 1 - p.y);
  float3 background = texelFetch(srt.reconstructed, dst, 0).rgb;
  imageStore(srt.composited, dst, float4(L + T * background, 1));
}

struct BeforeTransparency {
  [[sampler(0)]] sampler2D background;
  [[sampler(1)]] sampler2D layer_r;
  [[sampler(2)]] sampler2D layer_g;
  [[sampler(3)]] sampler2D layer_b;
  [[image(0, write, SFLOAT_16_16_16_16)]] image2D before_color;
  [[push_constant]] bool colored;
};
[[compute, local_size(8, 8)]]
void before_transparency([[resource_table]] BeforeTransparency &srt,
                         [[global_invocation_id]] const uint3 id)
{
  int2 p = int2(id.xy), size = imageSize(srt.before_color);
  if (any(greaterThanEqual(p, size))) {
    return;
  }
  float4 r = texelFetch(srt.layer_r, p, 0);
  float3 L = r.rgb, T = float3(r.a);
  if (srt.colored) {
    float2 g = texelFetch(srt.layer_g, p, 0).rg, b = texelFetch(srt.layer_b, p, 0).rg;
    L = float3(r.r, g.r, b.r);
    T = float3(r.g, g.g, b.g);
  }
  float3 B = texelFetch(srt.background, p, 0).rgb;
  imageStore(srt.before_color, int2(p.x, size.y - 1 - p.y), float4(L + T * B, 1.0f));
}

struct Display {
  [[push_constant]] int2 crop;
  [[sampler(0)]] sampler2D sr_output;
  [[sampler(1)]] sampler2D sr_depth;
  [[push_constant]] float2 sr_jitter;
};
struct FragmentOut {
  [[frag_color(0)]] float4 color;
};
[[vertex]] void fullscreen([[vertex_id]] const int id, [[position]] float4 &position)
{
  fullscreen_vertex(id, position);
}
[[fragment]]
void display([[resource_table]] const Display &srt,
             [[out]] FragmentOut &out,
             [[frag_coord]] const float4 frag,
             [[frag_depth(any)]] float depth)
{
  int2 size = textureSize(srt.sr_output, 0);
  int2 dst = int2(frag.xy);
  int2 src = int2(dst.x + srt.crop.x, size.y - 1 - dst.y - srt.crop.y);
  out.color = float4(texelFetch(srt.sr_output, src, 0).rgb, 1.0f);
  int2 depth_size = textureSize(srt.sr_depth, 0);
  float2 uv = (float2(src) + 0.5f) / float2(size);
  int2 depth_src = int2(uv * float2(depth_size) + float2(srt.sr_jitter.x, -srt.sr_jitter.y));
  depth_src = clamp(depth_src, int2(0), depth_size - 1);
  /* The viewport overlay framebuffer uses conventional depth, unlike EEVEE's render buffers. */
  depth = texelFetch(srt.sr_depth, depth_src, 0).r;
  depth = clamp(depth + gpu_fwidth(depth) + 2.4e-7f * 4.0f, 0.0f, 1.0f);
}
}  // namespace eevee::dlss

PipelineCompute eevee_dlss_prepare(eevee::dlss::prepare);
PipelineGraphic eevee_dlss_display(eevee::dlss::fullscreen, eevee::dlss::display);

PipelineCompute eevee_dlss_guides(eevee::dlss::guides);
PipelineCompute eevee_dlss_hit(eevee::dlss::hit);
PipelineCompute eevee_dlss_output(eevee::dlss::resolve_output);

PipelineCompute eevee_dlss_blend(eevee::dlss::blend);

PipelineCompute eevee_dlss_integrate(eevee::dlss::integrate);

PipelineCompute eevee_dlss_subsurface(eevee::dlss::subsurface_guide);
PipelineCompute eevee_dlss_before_transparency(eevee::dlss::before_transparency);
