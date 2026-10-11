/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <memory>

#include "BLI_math_matrix_types.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

namespace blender::gpu {
class AccelerationStructure;
using AccelerationStructurePtr = std::shared_ptr<AccelerationStructure>;

struct RayTracingInstance {
  AccelerationStructurePtr geometry;
  float4x4 transform;
  uint32_t id = 0;
  uint8_t mask = 0xff;
};

/** Triangle BLAS or instance TLAS. An active GPU context is required for all operations.
 * TLAS retains its BLAS references. Destruction defers GPU resources until submitted work
 * completes.
 */
class AccelerationStructure {
 public:
  virtual ~AccelerationStructure() = default;
  virtual bool build_mesh(Span<float3> positions,
                          Span<uint3> triangles,
                          bool allow_update,
                          bool opaque = true) = 0;
  virtual bool build_instances(Span<RayTracingInstance> instances) = 0;
  virtual void bind(int slot) = 0;
  virtual bool is_valid() const = 0;
  virtual size_t memory_usage() const = 0;
};

/** Returns null on backends without ray query support. */
AccelerationStructurePtr acceleration_structure_create();
}  // namespace blender::gpu
