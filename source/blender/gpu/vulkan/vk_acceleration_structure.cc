/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <cmath>

#include "BLI_math_matrix.hh"
#include "BLI_set.hh"
#include "GPU_debug.hh"
#include "vk_acceleration_structure.hh"
#include "vk_backend.hh"
#include "vk_context.hh"
#include "vk_state_manager.hh"

namespace blender::gpu {
namespace {
bool upload(VKBuffer &buffer, const void *data, size_t size)
{
  /* New storage avoids overwriting inputs consumed by an earlier queued build. */
  if (buffer.is_allocated()) {
    buffer.free();
  }
  if (!buffer.create(std::max(size, size_t(16)),
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                     1.0f))
  {
    return false;
  }
  if (size) {
    buffer.update_sub_immediately(0, size, data);
    buffer.flush_mapped_memory();
  }
  return true;
}
VkPhysicalDeviceAccelerationStructurePropertiesKHR properties()
{
  VkPhysicalDeviceAccelerationStructurePropertiesKHR as = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &as};
  vkGetPhysicalDeviceProperties2(VKBackend::get().device.physical_device_get(), &props);
  return as;
}
}  // namespace

void VKAccelerationStructure::discard_handle()
{
  if (handle_) {
    VKDiscardPool::discard_pool_get().discard_acceleration_structure(handle_);
    handle_ = VK_NULL_HANDLE;
    address_ = 0;
  }
}
VKAccelerationStructure::~VKAccelerationStructure()
{
  if (VKContext *context = VKContext::get()) {
    context->state_manager_get().storage_buffer_unbind(this);
  }
  discard_handle();
}

bool VKAccelerationStructure::build_mesh(Span<float3> positions,
                                         Span<uint3> triangles,
                                         bool allow_update)
{
  const bool was_valid = valid_;
  valid_ = false;
  const auto limits = properties();
  if (positions.is_empty() || triangles.is_empty() || uint64_t(positions.size()) > UINT32_MAX ||
      uint64_t(triangles.size()) > limits.maxPrimitiveCount)
  {
    return false;
  }
  for (const uint3 &triangle : triangles) {
    if (triangle.x >= positions.size() || triangle.y >= positions.size() ||
        triangle.z >= positions.size())
    {
      return false;
    }
  }
  const bool update = was_valid && handle_ &&
                      type_ == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR && allow_update_ &&
                      allow_update && vertex_count_ == positions.size() &&
                      primitive_count_ == triangles.size() && triangles_.as_span() == triangles;
  if (!upload(vertices_, positions.data(), positions.size_in_bytes()) ||
      !upload(indices_, triangles.data(), triangles.size_in_bytes()))
  {
    return false;
  }
  type_ = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  allow_update_ = allow_update;
  vertex_count_ = uint32_t(positions.size());
  triangles_ = triangles;
  VkAccelerationStructureGeometryKHR geometry = {
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  auto &tri = geometry.geometry.triangles;
  tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  tri.vertexData.deviceAddress = vertices_.device_address_get();
  tri.vertexStride = sizeof(float3);
  tri.maxVertex = vertex_count_ - 1;
  tri.indexType = VK_INDEX_TYPE_UINT32;
  tri.indexData.deviceAddress = indices_.device_address_get();
  const VkBuffer inputs[] = {vertices_.vk_handle(), indices_.vk_handle()};
  return build(
      geometry, uint32_t(triangles.size()), Span<VkBuffer>(inputs, ARRAY_SIZE(inputs)), update);
}

bool VKAccelerationStructure::build_instances(Span<RayTracingInstance> instances)
{
  const bool was_valid = valid_;
  valid_ = false;
  if (uint64_t(instances.size()) > properties().maxInstanceCount) {
    return false;
  }
  Vector<VkAccelerationStructureInstanceKHR> data;
  Vector<AccelerationStructurePtr> bottom_levels;
  Set<AccelerationStructure *> retained;
  for (const RayTracingInstance &instance : instances) {
    if (!instance.geometry || !instance.geometry->is_valid() || instance.id > 0x00ffffff) {
      return false;
    }
    /* Vulkan requires an invertible instance transform. Reject it before submitting work. */
    const float determinant = math::determinant(float3x3(instance.transform));
    if (!std::isfinite(determinant) || determinant == 0.0f) {
      return false;
    }
    for (int i = 0; i < 3; i++) {
      if (!std::isfinite(instance.transform[3][i])) {
        return false;
      }
    }
    const auto &blas = static_cast<const VKAccelerationStructure &>(*instance.geometry);
    if (blas.type_ != VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR) {
      return false;
    }
    VkAccelerationStructureInstanceKHR item = {};
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 4; col++) {
        item.transform.matrix[row][col] = instance.transform[col][row];
      }
    }
    item.instanceCustomIndex = instance.id;
    item.mask = instance.mask;
    item.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    item.accelerationStructureReference = blas.address_;
    data.append(item);
    if (retained.add(instance.geometry.get())) {
      bottom_levels.append(instance.geometry);
    }
  }
  if (!upload(instances_, data.data(), data.size() * sizeof(VkAccelerationStructureInstanceKHR))) {
    return false;
  }
  const bool update = was_valid && handle_ &&
                      type_ == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR &&
                      primitive_count_ == instances.size();
  type_ = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  allow_update_ = true;
  bottom_levels_ = std::move(bottom_levels);
  VkAccelerationStructureGeometryKHR geometry = {
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  geometry.geometry.instances.data.deviceAddress = instances_.device_address_get();
  const VkBuffer inputs[] = {instances_.vk_handle()};
  return build(
      geometry, uint32_t(instances.size()), Span<VkBuffer>(inputs, ARRAY_SIZE(inputs)), update);
}

bool VKAccelerationStructure::build(VkAccelerationStructureGeometryKHR geometry,
                                    uint32_t primitive_count,
                                    Span<VkBuffer> inputs,
                                    bool update)
{
  VKDevice &device = VKBackend::get().device;
  render_graph::VKBuildAccelerationStructureData node;
  node.geometry = geometry;
  node.info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  node.info.type = type_;
  node.info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  if (allow_update_) {
    node.info.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  }
  node.info.geometryCount = 1;
  node.info.pGeometries = &node.geometry;
  VkAccelerationStructureBuildSizesInfoKHR sizes = {
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  device.functions.vkGetAccelerationStructureBuildSizesKHR(
      device.vk_handle(),
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &node.info,
      &primitive_count,
      &sizes);
  if (!update || sizes.accelerationStructureSize > uint64_t(storage_.size_in_bytes())) {
    update = false;
    discard_handle();
    if (storage_.is_allocated()) {
      storage_.free();
    }
    if (!storage_.create(sizes.accelerationStructureSize,
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                         0,
                         1.0f))
    {
      return false;
    }
    VkAccelerationStructureCreateInfoKHR create = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    create.buffer = storage_.vk_handle();
    create.size = sizes.accelerationStructureSize;
    create.type = type_;
    if (device.functions.vkCreateAccelerationStructureKHR(
            device.vk_handle(), &create, nullptr, &handle_) != VK_SUCCESS)
    {
      return false;
    }
    VkAccelerationStructureDeviceAddressInfoKHR address = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    address.accelerationStructure = handle_;
    address_ = device.functions.vkGetAccelerationStructureDeviceAddressKHR(device.vk_handle(),
                                                                           &address);
  }
  const VkDeviceSize alignment = properties().minAccelerationStructureScratchOffsetAlignment;
  const VkDeviceSize scratch_size = (update ? sizes.updateScratchSize : sizes.buildScratchSize) +
                                    alignment;
  if (!scratch_.is_allocated() || uint64_t(scratch_.size_in_bytes()) < scratch_size) {
    if (scratch_.is_allocated()) {
      scratch_.free();
    }
    if (!scratch_.create(scratch_size,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                         0,
                         1.0f))
    {
      return false;
    }
  }
  node.info.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR :
                            VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  node.info.srcAccelerationStructure = update ? handle_ : VK_NULL_HANDLE;
  node.info.dstAccelerationStructure = handle_;
  node.info.scratchData.deviceAddress = (scratch_.device_address_get() + alignment - 1) &
                                        ~(alignment - 1);
  node.info.pGeometries = nullptr; /* Resolved by the graph node at execution. */
  node.range.primitiveCount = primitive_count;
  node.destination = storage_.vk_handle();
  node.scratch = scratch_.vk_handle();
  node.inputs.extend(inputs);
  for (const auto &blas : bottom_levels_) {
    const VkBuffer buffer = static_cast<VKAccelerationStructure &>(*blas).storage_.vk_handle();
    if (!node.bottom_levels.contains(buffer)) {
      node.bottom_levels.append(buffer);
    }
  }
  GPU_debug_group_begin(type_ == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR ?
                            "HardwareRT.BLAS" :
                            "HardwareRT.TLAS");
  VKContext::get()->render_graph().add_node(node);
  GPU_debug_group_end();
  primitive_count_ = primitive_count;
  valid_ = true;
  return true;
}

void VKAccelerationStructure::bind(int slot)
{
  BLI_assert(valid_ && slot >= 0);
  VKContext::get()->state_manager_get().storage_buffer_bind(
      BindSpaceStorageBuffers::Type::AccelerationStructure, this, slot, 0);
}
void VKAccelerationStructure::add_dependencies(render_graph::VKResourceAccessInfo &access) const
{
  access.buffers.append({storage_.vk_handle(), VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR});
  /* Address references are invisible to normal descriptor dependency discovery. */
  for (const auto &blas : bottom_levels_) {
    const VkBuffer buffer = static_cast<VKAccelerationStructure &>(*blas).storage_.vk_handle();
    access.buffers.append({buffer, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR});
  }
}
size_t VKAccelerationStructure::memory_usage() const
{
  return storage_.allocated_size_in_bytes() + scratch_.allocated_size_in_bytes() +
         vertices_.allocated_size_in_bytes() + indices_.allocated_size_in_bytes() +
         instances_.allocated_size_in_bytes();
}
}  // namespace blender::gpu
