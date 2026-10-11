/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "BLI_math_matrix.hh"
#include "GPU_acceleration_structure.hh"
#include "GPU_capabilities.hh"
#include "GPU_compute.hh"
#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_storage_buffer.hh"
#include "GPU_texture.hh"
#include "MEM_guardedalloc.h"
#include "gpu_testing.hh"
#include <array>

namespace blender::gpu::tests {

static void test_ray_query_scene_updates()
{
  if (!GPU_ray_query_support()) {
    EXPECT_EQ(acceleration_structure_create(), nullptr);
    GTEST_SKIP() << "Hardware ray query unavailable";
  }
  Shader *shader = GPU_shader_create_from_info_name("gpu_ray_query_test");
  ASSERT_NE(shader, nullptr);
  auto blas = acceleration_structure_create();
  auto tlas = acceleration_structure_create();
  ASSERT_NE(blas, nullptr);
  ASSERT_NE(tlas, nullptr);
  Vector<float3> positions = {{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
  Vector<uint3> triangles = {{0, 1, 2}};
  ASSERT_TRUE(blas->build_mesh(positions, triangles, true));
  Vector<RayTracingInstance> instances = {{blas, float4x4::identity(), 42, 0xff}};
  ASSERT_TRUE(tlas->build_instances(instances));

  StorageBuf *output = GPU_storagebuf_create_ex(
      2 * sizeof(float4), nullptr, GPU_USAGE_DYNAMIC, "RayQueryResult");
  StorageBuf *rays = GPU_storagebuf_create_ex(
      2 * sizeof(float4), nullptr, GPU_USAGE_DYNAMIC, "RayQueryInput");
  auto query = [&](float3 origin, float3 direction) {
    const float4 input[2] = {float4(origin, 0.001f), float4(direction, 100.0f)};
    GPU_storagebuf_update(rays, input);
    GPU_shader_bind(shader);
    tlas->bind(0);
    GPU_storagebuf_bind(output, 1);
    GPU_storagebuf_bind(rays, 2);
    GPU_compute_dispatch(shader, 1, 1, 1);
    GPU_memory_barrier(GPU_BARRIER_BUFFER_UPDATE);
    std::array<float4, 2> result;
    GPU_storagebuf_read(output, result.data());
    return result;
  };

  auto result = query({0, 0, 2}, {0, 0, -1});
  EXPECT_NEAR(result[0].x, 2.0f, 1e-5f);
  EXPECT_EQ(result[0].y, 42.0f);
  EXPECT_EQ(result[0].z, 0.0f);
  EXPECT_EQ(result[0].w, 1.0f);
  EXPECT_NEAR(result[1].x, 0.25f, 1e-5f);
  EXPECT_NEAR(result[1].y, 0.5f, 1e-5f);
  EXPECT_EQ(query({3, 0, 2}, {0, 0, -1})[0].w, 0.0f);
  EXPECT_NE(query({0, 0, -2}, {0, 0, 1})[1].z, result[1].z);

  /* Shared geometry still yields independent instance identities. */
  float4x4 second_transform = float4x4::identity();
  second_transform[3].x = 4;
  instances.append({blas, second_transform, 73, 0xff});
  ASSERT_TRUE(tlas->build_instances(instances));
  EXPECT_EQ(query({4, 0, 2}, {0, 0, -1})[0].y, 73.0f);
  EXPECT_EQ(query({0, 0, 2}, {0, 0, -1})[0].y, 42.0f);

  /* TLAS update without rebuilding BLAS, including an orientation reversing instance. */
  instances[0].transform[3].z = 1;
  instances[0].transform[0].x = -1;
  ASSERT_TRUE(tlas->build_instances(instances));
  EXPECT_NEAR(query({0, 0, 2}, {0, 0, -1})[0].x, 1.0f, 1e-5f);

  instances[0].transform[0].x = 0;
  EXPECT_FALSE(tlas->build_instances(instances));
  EXPECT_FALSE(tlas->is_valid());
  instances[0].transform[0].x = -1;
  ASSERT_TRUE(tlas->build_instances(instances));
  EXPECT_NEAR(query({0, 0, 2}, {0, 0, -1})[0].x, 1.0f, 1e-5f);

  /* Vertex deformation, then a topology change requiring a BLAS rebuild. */
  for (float3 &position : positions) {
    position.z = -0.5f;
  }
  ASSERT_TRUE(blas->build_mesh(positions, triangles, true));
  ASSERT_TRUE(tlas->build_instances(instances));
  EXPECT_NEAR(query({0, 0, 2}, {0, 0, -1})[0].x, 1.5f, 1e-5f);
  positions.extend({{9, -1, 0}, {11, -1, 0}, {10, 1, 0}});
  triangles = {{3, 4, 5}, {0, 1, 2}};
  ASSERT_TRUE(blas->build_mesh(positions, triangles, true));
  ASSERT_TRUE(tlas->build_instances(instances));
  EXPECT_EQ(query({0, 0, 2}, {0, 0, -1})[0].z, 1.0f);

  /* TLAS retains referenced BLAS; emptying it releases the reference after queued use. */
  std::weak_ptr<AccelerationStructure> weak = blas;
  instances.clear();
  blas.reset();
  EXPECT_FALSE(weak.expired());
  EXPECT_EQ(query({0, 0, 2}, {0, 0, -1})[0].w, 1.0f);
  ASSERT_TRUE(tlas->build_instances({}));
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(query({0, 0, 2}, {0, 0, -1})[0].w, 0.0f);

  GPU_shader_unbind();
  GPU_storagebuf_unbind(output);
  GPU_storagebuf_unbind(rays);
  GPU_storagebuf_free(output);
  GPU_storagebuf_free(rays);
  GPU_shader_free(shader);
}
GPU_TEST(ray_query_scene_updates)

static void test_ray_query_non_opaque_candidates()
{
  if (!GPU_ray_query_support()) {
    GTEST_SKIP();
  }
  Shader *shader = GPU_shader_create_from_info_name("gpu_ray_query_candidate_test");
  ASSERT_NE(shader, nullptr);
  auto blas = acceleration_structure_create();
  auto tlas = acceleration_structure_create();
  Vector<float3> positions = {
      {-1, -1, 1}, {1, -1, 1}, {0, 1, 1}, {-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
  Vector<uint3> triangles = {{0, 1, 2}, {3, 4, 5}};
  const float4 input[2] = {float4(0, 0, 2, 0.001f), float4(0, 0, -1, 100)};
  StorageBuf *rays = GPU_storagebuf_create_ex(sizeof(input), input, GPU_USAGE_STATIC, "Rays");
  StorageBuf *output = GPU_storagebuf_create_ex(sizeof(input), nullptr, GPU_USAGE_DYNAMIC, "Hits");
  /* Changing opacity with identical topology must rebuild rather than illegally update BLAS. */
  for (const bool opaque : {false, true, false}) {
    ASSERT_TRUE(blas->build_mesh(positions, triangles, true, opaque));
    Vector<RayTracingInstance> instances = {{blas, float4x4::identity(), 42, 0xff}};
    ASSERT_TRUE(tlas->build_instances(instances));
    GPU_shader_bind(shader);
    tlas->bind(0);
    GPU_storagebuf_bind(output, 1);
    GPU_storagebuf_bind(rays, 2);
    GPU_compute_dispatch(shader, 1, 1, 1);
    GPU_memory_barrier(GPU_BARRIER_BUFFER_UPDATE);
    std::array<float4, 2> result;
    GPU_storagebuf_read(output, result.data());
    EXPECT_EQ(result[0].w, 1.0f);
    EXPECT_EQ(result[0].y, 42.0f);
    EXPECT_EQ(result[0].z, opaque ? 0.0f : 1.0f);
    EXPECT_NEAR(result[0].x, opaque ? 1.0f : 2.0f, 1e-5f);
    if (opaque) {
      EXPECT_EQ(result[1].w, 0.0f);
    }
    else {
      EXPECT_GE(result[1].w, 1.0f);
      EXPECT_NEAR(result[1].x, 0.25f, 1e-5f);
      EXPECT_NEAR(result[1].y, 0.5f, 1e-5f);
      EXPECT_EQ(result[1].z, 42.0f);
    }
  }
  GPU_shader_unbind();
  GPU_storagebuf_unbind(output);
  GPU_storagebuf_unbind(rays);
  GPU_storagebuf_free(output);
  GPU_storagebuf_free(rays);
  GPU_shader_free(shader);
}
GPU_VULKAN_TEST(ray_query_non_opaque_candidates)

static void test_ray_query_material_image_mips()
{
  if (!GPU_ray_query_support()) {
    GTEST_SKIP();
  }
  Shader *shader = GPU_shader_create_from_info_name("eevee_hardware_trace");
  ASSERT_NE(shader, nullptr);
  auto constants = GPU_shader_get_default_constant_state(shader);
  ASSERT_FALSE(constants.is_empty());
  constants.set_value(GPU_shader_get_constant(shader, "closure_index"), 1);
  GPU_shader_bind(shader, &constants);

  /* Match read-only HWRT arrays. The compute mip helper cannot restore specialization state. */
  Texture *image = GPU_texture_create_2d_array("RT material mip regression",
                                               16,
                                               16,
                                               2,
                                               5,
                                               TextureFormat::SFLOAT_16_16_16_16,
                                               GPU_TEXTURE_USAGE_SHADER_READ |
                                                   GPU_TEXTURE_USAGE_HOST_READ,
                                               nullptr);
  ASSERT_NE(image, nullptr);
  const float4 color(0.25f, 0.5f, 0.75f, 1);
  GPU_texture_clear(image, GPU_DATA_FLOAT, color);
  GPU_texture_update_mipmap_chain(image);
  EXPECT_EQ(GPU_shader_get_bound(), shader);
  float4 *pixels = static_cast<float4 *>(GPU_texture_read(image, GPU_DATA_FLOAT, 4));
  ASSERT_NE(pixels, nullptr);
  EXPECT_EQ(pixels[0], color);
  EXPECT_EQ(pixels[1], color);
  MEM_delete(pixels);
  GPU_texture_free(image);
  GPU_shader_unbind();
  GPU_shader_free(shader);
}
GPU_VULKAN_TEST(ray_query_material_image_mips)

static void test_ray_query_eevee_shaders()
{
  if (!GPU_ray_query_support()) {
    GTEST_SKIP();
  }
  for (const char *name : {"eevee_hardware_radiance_capture",
                           "eevee_hardware_cache_select",
                           "eevee_hardware_cache_resolve",
                           "eevee_hardware_trace",
                           "eevee_hardware_debug",
                           "eevee_hardware_irradiance_update"})
  {
    Shader *shader = GPU_shader_create_from_info_name(name);
    ASSERT_NE(shader, nullptr) << name;
    GPU_shader_free(shader);
  }
}
GPU_TEST(ray_query_eevee_shaders)
}  // namespace blender::gpu::tests
