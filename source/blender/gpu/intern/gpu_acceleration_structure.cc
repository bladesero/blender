/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GPU_acceleration_structure.hh"
#include "GPU_capabilities.hh"
#include "gpu_backend.hh"

namespace blender::gpu {
AccelerationStructurePtr acceleration_structure_create()
{
  return GPU_ray_query_support() ? GPUBackend::get()->acceleration_structure_alloc() : nullptr;
}
}  // namespace blender::gpu
