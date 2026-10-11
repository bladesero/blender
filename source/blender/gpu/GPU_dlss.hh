/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <memory>
#include <string>

#include "BLI_math_matrix_types.hh"
#include "BLI_math_vector_types.hh"

namespace blender::gpu {
class Texture;

enum class DLSSQuality { Quality, Balanced, Performance };
enum class DLSSMode { SuperResolution, RayReconstruction };

struct DLSSFrame {
  Texture *color = nullptr, *depth = nullptr, *motion = nullptr, *output = nullptr;
  Texture *diffuse_albedo = nullptr, *specular_albedo = nullptr;
  Texture *normal_roughness = nullptr, *specular_hit_distance = nullptr;
  float4x4 world_to_view = float4x4::identity();
  float4x4 view_to_clip = float4x4::identity();
  float2 jitter = float2(0);
  bool reset = false;
};

/** One instance per viewport or output frame. Destruction requires an active GPU context. */
class DLSS {
 public:
  virtual ~DLSS() = default;
  static std::unique_ptr<DLSS> create(std::string &error,
                                      DLSSMode mode = DLSSMode::SuperResolution);
  virtual bool configure(int2 output_size, DLSSQuality quality, int2 &input_size) = 0;
  /** Scene-linear HDR, conventional [0,1] depth, unjittered current-to-previous motion
   * in input pixels. All textures have top-left origin. Output is RGB only. */
  virtual bool evaluate(const DLSSFrame &frame) = 0;
  virtual bool failed() const = 0;
  virtual std::string error() const = 0;
};
}  // namespace blender::gpu
