/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once
#include "eevee_reverse_z_lib.bsl.hh"
#include "eevee_velocity.bsl.hh"
#include "eevee_gbuffer_read.bsl.hh"
#include "eevee_closure.bsl.hh"
#include "gpu_shader_fullscreen_lib.glsl"

namespace eevee::dlss {
struct Prepare {
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
  float2 motion = srt.static_frame ? float2(0.0f) : camera.resolve(views, srt.input_vector, src, depth).xy;
  motion *= float2(size.x, -size.y);
  if (any(isnan(motion)) || any(isinf(motion))) {
    motion = float2(0.0f);
  }
  imageStore(srt.sr_color, dst, float4(texelFetch(srt.input_color, src, 0).rgb, 1.0f));
  imageStore(srt.sr_depth, dst, float4(depth));
  imageStore(srt.sr_motion, dst, float4(motion, 0.0f, 0.0f));
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
  if (any(greaterThanEqual(src, size))) { return; }
  int2 dst = int2(src.x, size.y - 1 - src.y);
  gbuffer::Layers gbuf = reader.read_layers(src);
  float3 diffuse = float3(0.0f), specular = float3(0.0f), N = float3(0.0f, 0.0f, 1.0f);
  float roughness = 1.0f, dominant = -1.0f;
  for (int i = 0; i < 3; i++) {
    ClosureUndetermined cl = gbuf.layer[i];
    if (cl.type == CLOSURE_BSDF_DIFFUSE_ID) { diffuse += cl.color; }
    if (cl.type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) {
      /* EEVEE's node evaluation already applied its integrated BRDF LUT and closure weights
       * to cl.color. Applying a second Fresnel approximation here would be incorrect. */
      specular += cl.color;
      float weight = dot(cl.color, float3(0.2126f, 0.7152f, 0.0722f));
      if (weight > dominant) {
        dominant = weight; N = cl.N; roughness = closure_apparent_roughness_get(cl);
      }
    }
    else if (cl.type == CLOSURE_BSDF_DIFFUSE_ID && dominant < 0.0f) { N = cl.N; }
  }
  imageStore(srt.rr_diffuse, dst, float4(max(diffuse, float3(0.0f)), 1.0f));
  imageStore(srt.rr_specular, dst, float4(max(specular, float3(0.0f)), 1.0f));
  imageStore(srt.rr_normals, dst, float4(normalize(N), clamp(roughness, 0.0f, 1.0f)));
}
struct Hit {
  [[sampler(0)]] sampler2D ray_time;
  [[image(0, write, SFLOAT_32)]] image2D rr_hit;
  [[push_constant]] int closure_index;
};
[[compute, local_size(8, 8)]]
void hit([[resource_table]] Hit &srt,
         [[resource_table]] const gbuffer::Reader &reader,
         [[global_invocation_id]] const uint3 id)
{
  int2 size = imageSize(srt.rr_hit), src = int2(id.xy);
  if (any(greaterThanEqual(src, size))) { return; }
  ClosureUndetermined cl = reader.read_bin(src, uchar(srt.closure_index));
  if (cl.type != CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) { return; }
  /* Match the dominant reflection selected by the normal/roughness guide. */
  float weight = dot(cl.color, float3(0.2126f, 0.7152f, 0.0722f));
  for (int i = 0; i < 3; i++) {
    ClosureUndetermined other = reader.read_bin(src, uchar(i));
    if (other.type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID &&
        dot(other.color, float3(0.2126f, 0.7152f, 0.0722f)) > weight) { return; }
  }
  float distance = texelFetch(srt.ray_time, src, 0).r;
  /* EEVEE's environment/probe miss sentinel is not a geometric hit. */
  if (distance == 10000.0f || distance <= 0.0f || isnan(distance) || isinf(distance)) { distance = 0.0f; }
  imageStore(srt.rr_hit, int2(src.x, size.y - 1 - src.y), float4(distance));
}
struct Output {
  [[sampler(0)]] sampler2D sr_output;
  [[image(0, write, SFLOAT_16_16_16_16)]] image2D final_output;
};
[[compute, local_size(8, 8)]]
void resolve_output([[resource_table]] Output &srt, [[global_invocation_id]] const uint3 id)
{
  int2 size = imageSize(srt.final_output), dst = int2(id.xy);
  if (any(greaterThanEqual(dst, size))) { return; }
  imageStore(srt.final_output, dst, float4(texelFetch(srt.sr_output, int2(dst.x, size.y - 1 - dst.y), 0).rgb, 1.0f));
}

struct Display {
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
  int2 src = int2(dst.x, size.y - 1 - dst.y);
  out.color = float4(texelFetch(srt.sr_output, src, 0).rgb, 1.0f);
  int2 depth_size = textureSize(srt.sr_depth, 0);
  float2 uv = (float2(src) + 0.5f) / float2(size);
  int2 depth_src = int2(uv * float2(depth_size) +
                         float2(srt.sr_jitter.x, -srt.sr_jitter.y));
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
