/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Shared code between host and client code-bases.
 */

#pragma once

#include "GPU_shader_shared_utils.hh"

#ifndef GPU_SHADER
namespace blender::eevee {
#endif

struct [[host_shared]] HardwareRTInstance {
  float4x4 world_to_object;
  /* stable instance ID, triangle offset, geometry generation, material-map offset. */
  uint4 identity;
  /* Receiver light set, blocker shadow set. */
  uint4 linking;
};

struct [[host_shared]] HardwareRTTriangle {
  /* Corner shading normals and UV x; UV y is stored separately. */
  float4 corner0;
  float4 corner1;
  float4 corner2;
  float4 uv_y;
  uint4 material;
};

struct [[host_shared]] HardwareRTParameter {
  float4 value;
  /* Pixel offset, width, height, extension/interpolation/channel flags; width=0: constant. */
  uint4 texture;
};

struct [[host_shared]] HardwareRTMaterial {
  struct HardwareRTParameter base_color;
  struct HardwareRTParameter roughness;
  struct HardwareRTParameter metallic;
  struct HardwareRTParameter emission;
  struct HardwareRTParameter emission_strength;
  /* IOR, specular level, supported, pure diffuse/emission. */
  float4 properties;
};

struct [[host_shared]] HardwareIrradianceProbe {
  /* World position and cell size. */
  float4 position;
  /* World normal and validity. */
  float4 normal;
  /* Exact signed cell coordinate and normal octant. */
  int4 key;
};

struct [[host_shared]] HardwareRadianceEntry {
  uint4 key;
  float4 radiance;
};

struct [[host_shared]] ScreenThicknessParameters {
  float thickness_ndc_scale;
  float thickness_ndc_bias;
  float thickness_vs_scale;
  float thickness_vs_bias;

  /* Return the depth buffer Z thickness of a pixel at a given view space Z depth. */
  static ScreenThicknessParameters build(float4x4 winmat,
                                         float avg_pixel_radius_unit,
                                         float min_pixel_thickness,
                                         float min_constant_thickness)
  {
    ScreenThicknessParameters params;
    avg_pixel_radius_unit *= min_pixel_thickness;
    if (winmat[3][3] == 0.0f) {
      /* Perspective pixels increase footprint with the distance. */
      params.thickness_vs_scale = avg_pixel_radius_unit;
      params.thickness_vs_bias = -min_constant_thickness;
      params.thickness_ndc_scale = winmat[3][2];
      params.thickness_ndc_bias = 0.0f;
      /* Convert from NDC to screen range. */
      params.thickness_ndc_scale *= 0.5f;
    }
    else {
      /* Orthographic pixels have fixed footprint. */
      params.thickness_vs_scale = 0.0f;
      params.thickness_vs_bias = 1.0f; /* Avoid NaN. */
      params.thickness_ndc_scale = 0.0f;
      params.thickness_ndc_bias = -(avg_pixel_radius_unit + min_constant_thickness) * winmat[2][2];
      /* Convert from NDC to screen range. */
      params.thickness_ndc_bias *= 0.5f;
    }
    return params;
  }

  /* Return the depth buffer Z thickness of a pixel at a given view space Z depth. */
  float pixel_depth_thickness_at(float vs_z) const
  {
    float vs_thickness = vs_z * thickness_vs_scale + thickness_vs_bias;
    /* NDC offset from view space offset.
     * From http://terathon.com/gdc07_lengyel.pdf (slide 24) */
    float ndc_thickness = vs_thickness / (vs_z * (vs_z + vs_thickness));
    return ndc_thickness * thickness_ndc_scale + thickness_ndc_bias;
  }
};

struct [[host_shared]] RayTraceData {
  /** ViewProjection matrix used to render the previous frame. */
  float4x4 history_persmat;
  /** ViewProjection matrix used to denoise the previous frame. */
  float4x4 denoise_history_persmat;
  /** Input resolution. */
  int2 full_resolution;
  /** Inverse of input resolution to get screen UVs. */
  float2 full_resolution_inv;
  /** Scale and bias to go from ray-trace resolution to input resolution. */
  int2 trace_pixel_offset;
  int trace_pixel_scale;
  /** View space thickness the objects. */
  float thickness;
  /** Scale and bias to go from fast GI resolution to input resolution. */
  int2 fast_gi_resolution_bias;
  int fast_gi_resolution_scale;
  /** Bias to the full-screen buffer LOD to account for radiance buffer top down-scaling factor. */
  float fast_gi_lod_bias;
  /** Scale to apply to full-screen UVs to remove padding. */
  float2 fast_gi_uv_scale;
  /** Determine how fast the sample steps are getting bigger. */
  float quality;
  /** Maximum roughness for which we will trace a ray. */
  float roughness_mask_scale;
  float roughness_mask_bias;
  /** If set to true will bypass spatial denoising. */
  bool32_t skip_denoise;
  /** If set to false will bypass tracing for refractive closures. */
  bool32_t trace_refraction;
  /** Closure being ray-traced. */
  int closure_index;
  /** If true, consider backface hit as valid. Otherwise, use ray miss pipeline. */
  bool32_t use_backface_hit;
  /** Amount of frontface lighting to use for backface hits. */
  float backface_hit_scale;
  uint _pad0;
  uint _pad1;

  struct ScreenThicknessParameters fast_gi_thickness;
  struct ScreenThicknessParameters ray_thickness;
};

struct [[host_shared]] AOData {
  float2 pixel_size;
  float distance;
  float lod_factor;

  float thickness_near;
  float thickness_far;
  float angle_bias;
  float gi_distance;

  float lod_factor_ao;
  float _pad0;
  float _pad1;
  float _pad2;
};

#ifndef GPU_SHADER
}  // namespace blender::eevee
#endif
