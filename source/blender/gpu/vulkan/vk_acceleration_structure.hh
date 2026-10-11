/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once
#include "GPU_acceleration_structure.hh"
#include "render_graph/vk_resource_access_info.hh"
#include "vk_buffer.hh"

namespace blender::gpu {
class VKAccelerationStructure : public AccelerationStructure {
  VkAccelerationStructureKHR handle_ = VK_NULL_HANDLE;
  VkDeviceAddress address_ = 0;
  VKBuffer storage_;
  VKBuffer scratch_;
  VKBuffer vertices_;
  VKBuffer indices_;
  VKBuffer instances_;
  Vector<AccelerationStructurePtr> bottom_levels_;
  Vector<uint3> triangles_;
  uint32_t vertex_count_ = 0;
  uint32_t primitive_count_ = 0;
  VkAccelerationStructureTypeKHR type_ = VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR;
  bool allow_update_ = false;
  bool valid_ = false;

  bool build(VkAccelerationStructureGeometryKHR geometry,
             uint32_t primitive_count,
             Span<VkBuffer> inputs,
             bool update);
  void discard_handle();

 public:
  ~VKAccelerationStructure() override;
  bool build_mesh(Span<float3> positions, Span<uint3> triangles, bool allow_update) override;
  bool build_instances(Span<RayTracingInstance> instances) override;
  void bind(int slot) override;
  bool is_valid() const override
  {
    return valid_;
  }
  size_t memory_usage() const override;
  VkAccelerationStructureKHR vk_handle() const
  {
    return handle_;
  }
  void add_dependencies(render_graph::VKResourceAccessInfo &access) const;
};
}  // namespace blender::gpu
