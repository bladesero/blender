/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "testing/testing.h"

#include "vk_render_graph_test_types.hh"

namespace blender::gpu::render_graph {

class VKRenderGraphTestCompute : public VKRenderGraphTest {};

TEST_F(VKRenderGraphTestCompute, external_rr_inputs_and_reuse)
{
  VKExternalComputeData rr;
  const VkImage output = VkHandle<VkImage>(8);
  resources.add_image(output, false);
  for (uint64_t i = 1; i <= 7; i++) {
    const VkImage input = VkHandle<VkImage>(i);
    resources.add_image(input, false);
    rr.inputs.append(input);
    VKResourceAccessInfo access;
    access.images.append({input, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT});
    VKDispatchNode::CreateInfo producer(access);
    producer.dispatch_node.group_count_x = producer.dispatch_node.group_count_y =
        producer.dispatch_node.group_count_z = 1;
    render_graph->add_node(producer);
  }
  rr.outputs.append(output);
  rr.record = [](VkCommandBuffer) {};
  render_graph->add_node(rr);
  /* Reuse the first guide after the SDK reads it. It must not be reordered before RR. */
  VKResourceAccessInfo reuse;
  reuse.images.append({rr.inputs[0], VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT});
  reuse.images.append({output, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT});
  VKDispatchNode::CreateInfo consumer(reuse);
  consumer.dispatch_node.group_count_x = consumer.dispatch_node.group_count_y =
      consumer.dispatch_node.group_count_z = 1;
  render_graph->add_node(consumer);
  submit(render_graph, command_buffer);
  int dispatches = 0, external = 0, input_barriers = 0;
  bool reuse_barrier = false;
  for (const std::string &entry : log) {
    if (entry == "external_compute()") {
      EXPECT_EQ(dispatches, 7);
      external++;
    }
    dispatches += entry.find("dispatch(") == 0;
    size_t offset = 0;
    while ((offset = entry.find("dst_access_mask=VK_ACCESS_MEMORY_READ_BIT", offset)) !=
           std::string::npos)
    {
      input_barriers++;
      offset++;
    }
    reuse_barrier |= external > 0 &&
                     entry.find("src_access_mask=VK_ACCESS_MEMORY_READ_BIT") !=
                         std::string::npos &&
                     entry.find("dst_access_mask=VK_ACCESS_SHADER_WRITE_BIT") != std::string::npos;
  }
  EXPECT_EQ(external, 1);
  EXPECT_EQ(dispatches, 8);
  /* Seven inputs plus the SDK output, which declares read/write access. */
  EXPECT_EQ(input_barriers, 8);
  EXPECT_TRUE(reuse_barrier);
}

TEST_F(VKRenderGraphTestCompute, external_compute_dependencies_and_rebind)
{
  const VkImage input = VkHandle<VkImage>(1);
  const VkImage output = VkHandle<VkImage>(2);
  resources.add_image(input, false);
  resources.add_image(output, false);
  auto dispatch = [&](VkImage image, VkAccessFlags flags) {
    VKResourceAccessInfo access;
    access.images.append({image, flags, VK_IMAGE_ASPECT_COLOR_BIT});
    VKDispatchNode::CreateInfo info(access);
    info.dispatch_node.pipeline_data.vk_pipeline = VkHandle<VkPipeline>(3);
    info.dispatch_node.pipeline_data.vk_pipeline_layout = VkHandle<VkPipelineLayout>(4);
    info.dispatch_node.pipeline_data.vk_descriptor_set = VkHandle<VkDescriptorSet>(5);
    info.dispatch_node.group_count_x = info.dispatch_node.group_count_y =
        info.dispatch_node.group_count_z = 1;
    render_graph->add_node(info);
  };
  dispatch(input, VK_ACCESS_SHADER_WRITE_BIT);
  VKExternalComputeData external;
  external.inputs.append(input);
  external.outputs.append(output);
  external.record = [](VkCommandBuffer) {};
  render_graph->add_node(external);
  dispatch(output, VK_ACCESS_SHADER_READ_BIT);
  submit(render_graph, command_buffer);

  int external_index = -1, dispatches = 0, pipeline_binds = 0, descriptor_binds = 0;
  bool input_barrier = false, output_barrier = false;
  for (int i : log.index_range()) {
    const std::string &entry = log[i];
    if (entry == "external_compute()") {
      external_index = i;
      EXPECT_EQ(dispatches, 1);
    }
    dispatches += entry.find("dispatch(") == 0;
    pipeline_binds += entry.find("bind_pipeline(") == 0;
    descriptor_binds += entry.find("bind_descriptor_sets(") == 0;
    input_barrier |= external_index < 0 &&
                     entry.find("src_access_mask=VK_ACCESS_SHADER_WRITE_BIT") != std::string::npos &&
                     entry.find("dst_access_mask=VK_ACCESS_MEMORY_READ_BIT") != std::string::npos;
    output_barrier |= external_index >= 0 &&
                      entry.find("VK_ACCESS_MEMORY_WRITE_BIT") != std::string::npos &&
                      entry.find("dst_access_mask=VK_ACCESS_SHADER_READ_BIT") != std::string::npos;
  }
  EXPECT_GE(external_index, 0);
  EXPECT_EQ(dispatches, 2);
  EXPECT_TRUE(input_barrier);
  EXPECT_TRUE(output_barrier);
  /* The same pipeline and descriptor set must be rebound after the external SDK call. */
  EXPECT_EQ(pipeline_binds, 2);
  EXPECT_EQ(descriptor_binds, 2);
}

TEST_F(VKRenderGraphTestCompute, acceleration_structure_dependencies)
{
  const VkBuffer vertices = VkHandle<VkBuffer>(1);
  const VkBuffer scratch = VkHandle<VkBuffer>(2);
  const VkBuffer blas = VkHandle<VkBuffer>(3);
  const VkBuffer tlas = VkHandle<VkBuffer>(4);
  for (VkBuffer buffer : {vertices, scratch, blas, tlas}) {
    resources.add_buffer(buffer);
  }
  VKFillBufferNode::CreateInfo upload = {};
  upload.vk_buffer = vertices;
  upload.size = 256;
  render_graph->add_node(upload);
  VKBuildAccelerationStructureData build;
  build.destination = blas;
  build.scratch = scratch;
  build.inputs.append(vertices);
  build.range.primitiveCount = 3;
  build.info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  render_graph->add_node(build);
  build.destination = tlas;
  build.bottom_levels.append(blas);
  build.range.primitiveCount = 1;
  render_graph->add_node(build);

  VKResourceAccessInfo access;
  access.buffers.append({tlas, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR});
  access.buffers.append({blas, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR});
  VKDispatchNode::CreateInfo dispatch(access);
  dispatch.dispatch_node.pipeline_data.vk_pipeline = VkHandle<VkPipeline>(5);
  dispatch.dispatch_node.pipeline_data.vk_pipeline_layout = VkHandle<VkPipelineLayout>(6);
  dispatch.dispatch_node.pipeline_data.vk_descriptor_set = VkHandle<VkDescriptorSet>(7);
  dispatch.dispatch_node.group_count_x = 1;
  dispatch.dispatch_node.group_count_y = 1;
  dispatch.dispatch_node.group_count_z = 1;
  render_graph->add_node(dispatch);
  submit(render_graph, command_buffer);

  int blas_index = -1, tlas_index = -1, trace_index = -1;
  bool upload_barrier = false, build_barrier = false, query_barrier = false;
  for (int i : log.index_range()) {
    const std::string &entry = log[i];
    if (entry == "build_acceleration_structure(primitives=3)") {
      blas_index = i;
    }
    if (entry == "build_acceleration_structure(primitives=1)") {
      tlas_index = i;
    }
    if (entry == "dispatch(group_count_x=1, group_count_y=1, group_count_z=1)") {
      trace_index = i;
    }
    auto contains = [&](const char *text) { return entry.find(text) != std::string::npos; };
    upload_barrier |= contains("src_access_mask=VK_ACCESS_TRANSFER_WRITE_BIT") &&
                      contains(
                          "dst_stage_mask=VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR");
    build_barrier |= blas_index >= 0 && tlas_index < 0 &&
                     contains("VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR") &&
                     contains(
                         "dst_stage_mask=VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR");
    query_barrier |= tlas_index >= 0 && trace_index < 0 &&
                     contains("VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR") &&
                     contains("dst_stage_mask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT");
  }
  EXPECT_GE(blas_index, 0);
  EXPECT_GT(tlas_index, blas_index);
  EXPECT_GT(trace_index, tlas_index);
  EXPECT_TRUE(upload_barrier);
  EXPECT_TRUE(build_barrier);
  EXPECT_TRUE(query_barrier);
}

TEST_F(VKRenderGraphTestCompute, dispatch_read_back)
{
  VkHandle<VkBuffer> buffer(1u);
  VkHandle<VkPipeline> pipeline(2u);
  VkHandle<VkPipelineLayout> pipeline_layout(3u);
  VkHandle<VkDescriptorSet> descriptor_set(4u);

  resources.add_buffer(buffer);
  VKResourceAccessInfo access_info = {};
  access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
  VKDispatchNode::CreateInfo dispatch_info(access_info);
  dispatch_info.dispatch_node.pipeline_data.vk_pipeline = pipeline;
  dispatch_info.dispatch_node.pipeline_data.vk_pipeline_layout = pipeline_layout;
  dispatch_info.dispatch_node.pipeline_data.vk_descriptor_set = descriptor_set;
  dispatch_info.dispatch_node.group_count_x = 1;
  dispatch_info.dispatch_node.group_count_y = 1;
  dispatch_info.dispatch_node.group_count_z = 1;
  render_graph->add_node(dispatch_info);
  submit(render_graph, command_buffer);
  EXPECT_EQ(3, log.size());
  EXPECT_EQ("bind_pipeline(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, pipeline=0x2)",
            log[0]);
  EXPECT_EQ(
      "bind_descriptor_sets(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, layout=0x3, "
      "p_descriptor_sets=0x4)",
      log[1]);
  EXPECT_EQ("dispatch(group_count_x=1, group_count_y=1, group_count_z=1)", log[2]);
}

/**
 * Test that the descriptor sets are updated once when chaining dispatching.
 */
TEST_F(VKRenderGraphTestCompute, dispatch_dispatch_read_back)
{
  VkHandle<VkBuffer> buffer(1u);
  VkHandle<VkPipeline> pipeline(2u);
  VkHandle<VkPipelineLayout> pipeline_layout(3u);
  VkHandle<VkDescriptorSet> descriptor_set(4u);

  resources.add_buffer(buffer);

  {
    VKResourceAccessInfo access_info = {};
    access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
    VKDispatchNode::CreateInfo dispatch_info(access_info);
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline = pipeline;
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline_layout = pipeline_layout;
    dispatch_info.dispatch_node.pipeline_data.vk_descriptor_set = descriptor_set;
    dispatch_info.dispatch_node.group_count_x = 1;
    dispatch_info.dispatch_node.group_count_y = 1;
    dispatch_info.dispatch_node.group_count_z = 1;
    render_graph->add_node(dispatch_info);
  }
  {
    VKResourceAccessInfo access_info = {};
    access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
    VKDispatchNode::CreateInfo dispatch_info(access_info);
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline = pipeline;
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline_layout = pipeline_layout;
    dispatch_info.dispatch_node.pipeline_data.vk_descriptor_set = descriptor_set;
    dispatch_info.dispatch_node.group_count_x = 2;
    dispatch_info.dispatch_node.group_count_y = 2;
    dispatch_info.dispatch_node.group_count_z = 2;
    render_graph->add_node(dispatch_info);
  }
  submit(render_graph, command_buffer);
  EXPECT_EQ(5, log.size());
  EXPECT_EQ("bind_pipeline(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, pipeline=0x2)",
            log[0]);
  EXPECT_EQ(
      "bind_descriptor_sets(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, layout=0x3, "
      "p_descriptor_sets=0x4)",
      log[1]);
  EXPECT_EQ("dispatch(group_count_x=1, group_count_y=1, group_count_z=1)", log[2]);
  EXPECT_EQ(
      "pipeline_barrier(src_stage_mask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "
      "dst_stage_mask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT" +
          endl() +
          " - buffer_barrier(src_access_mask=VK_ACCESS_SHADER_WRITE_BIT, "
          "dst_access_mask=VK_ACCESS_SHADER_WRITE_BIT, buffer=0x1, offset=0, "
          "size=18446744073709551615)" +
          endl() + ")",
      log[3]);
  EXPECT_EQ("dispatch(group_count_x=2, group_count_y=2, group_count_z=2)", log[4]);
}

/**
 * Test that the descriptor sets are updated when chaining dispatching with different descriptor
 * sets.
 */
TEST_F(VKRenderGraphTestCompute, dispatch_dispatch_read_back_with_changing_descriptor_sets)
{
  VkHandle<VkBuffer> buffer(1u);
  VkHandle<VkPipeline> pipeline(2u);
  VkHandle<VkPipelineLayout> pipeline_layout(3u);
  VkHandle<VkDescriptorSet> descriptor_set_a(4u);
  VkHandle<VkDescriptorSet> descriptor_set_b(5u);

  resources.add_buffer(buffer);

  {
    VKResourceAccessInfo access_info = {};
    access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
    VKDispatchNode::CreateInfo dispatch_info(access_info);
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline = pipeline;
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline_layout = pipeline_layout;
    dispatch_info.dispatch_node.pipeline_data.vk_descriptor_set = descriptor_set_a;
    dispatch_info.dispatch_node.group_count_x = 1;
    dispatch_info.dispatch_node.group_count_y = 1;
    dispatch_info.dispatch_node.group_count_z = 1;
    render_graph->add_node(dispatch_info);
  }
  {
    VKResourceAccessInfo access_info = {};
    access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
    VKDispatchNode::CreateInfo dispatch_info(access_info);
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline = pipeline;
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline_layout = pipeline_layout;
    dispatch_info.dispatch_node.pipeline_data.vk_descriptor_set = descriptor_set_b;
    dispatch_info.dispatch_node.group_count_x = 2;
    dispatch_info.dispatch_node.group_count_y = 2;
    dispatch_info.dispatch_node.group_count_z = 2;
    render_graph->add_node(dispatch_info);
  }
  submit(render_graph, command_buffer);
  EXPECT_EQ(6, log.size());
  EXPECT_EQ("bind_pipeline(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, pipeline=0x2)",
            log[0]);
  EXPECT_EQ(
      "bind_descriptor_sets(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, layout=0x3, "
      "p_descriptor_sets=0x4)",
      log[1]);
  EXPECT_EQ("dispatch(group_count_x=1, group_count_y=1, group_count_z=1)", log[2]);
  EXPECT_EQ(
      "pipeline_barrier(src_stage_mask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "
      "dst_stage_mask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT" +
          endl() +
          " - buffer_barrier(src_access_mask=VK_ACCESS_SHADER_WRITE_BIT, "
          "dst_access_mask=VK_ACCESS_SHADER_WRITE_BIT, buffer=0x1, offset=0, "
          "size=18446744073709551615)" +
          endl() + ")",
      log[3]);
  EXPECT_EQ(
      "bind_descriptor_sets(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, layout=0x3, "
      "p_descriptor_sets=0x5)",
      log[4]);
  EXPECT_EQ("dispatch(group_count_x=2, group_count_y=2, group_count_z=2)", log[5]);
}

/**
 * Test that the descriptor sets are updated when chaining dispatching with different pipelines.
 */
TEST_F(VKRenderGraphTestCompute, dispatch_dispatch_read_back_with_changing_pipelines)
{
  VkHandle<VkBuffer> buffer(1u);
  VkHandle<VkPipeline> pipeline_a(2u);
  VkHandle<VkPipeline> pipeline_b(3u);
  VkHandle<VkPipelineLayout> pipeline_layout(4u);
  VkHandle<VkDescriptorSet> descriptor_set(5u);

  resources.add_buffer(buffer);

  {
    VKResourceAccessInfo access_info = {};
    access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
    VKDispatchNode::CreateInfo dispatch_info(access_info);
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline = pipeline_a;
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline_layout = pipeline_layout;
    dispatch_info.dispatch_node.pipeline_data.vk_descriptor_set = descriptor_set;
    dispatch_info.dispatch_node.group_count_x = 1;
    dispatch_info.dispatch_node.group_count_y = 1;
    dispatch_info.dispatch_node.group_count_z = 1;
    render_graph->add_node(dispatch_info);
  }
  {
    VKResourceAccessInfo access_info = {};
    access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
    VKDispatchNode::CreateInfo dispatch_info(access_info);
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline = pipeline_b;
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline_layout = pipeline_layout;
    dispatch_info.dispatch_node.pipeline_data.vk_descriptor_set = descriptor_set;
    dispatch_info.dispatch_node.group_count_x = 2;
    dispatch_info.dispatch_node.group_count_y = 2;
    dispatch_info.dispatch_node.group_count_z = 2;
    render_graph->add_node(dispatch_info);
  }
  submit(render_graph, command_buffer);
  EXPECT_EQ(6, log.size());
  EXPECT_EQ("bind_pipeline(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, pipeline=0x2)",
            log[0]);
  EXPECT_EQ(
      "bind_descriptor_sets(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, layout=0x4, "
      "p_descriptor_sets=0x5)",
      log[1]);
  EXPECT_EQ("dispatch(group_count_x=1, group_count_y=1, group_count_z=1)", log[2]);
  EXPECT_EQ(
      "pipeline_barrier(src_stage_mask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "
      "dst_stage_mask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT" +
          endl() +
          " - buffer_barrier(src_access_mask=VK_ACCESS_SHADER_WRITE_BIT, "
          "dst_access_mask=VK_ACCESS_SHADER_WRITE_BIT, buffer=0x1, offset=0, "
          "size=18446744073709551615)" +
          endl() + ")",
      log[3]);
  EXPECT_EQ("bind_pipeline(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, pipeline=0x3)",
            log[4]);
  EXPECT_EQ("dispatch(group_count_x=2, group_count_y=2, group_count_z=2)", log[5]);
}

/**
 * Test that the descriptor sets are updated when chaining dispatching with different pipelines and
 * descriptor sets.
 */
TEST_F(VKRenderGraphTestCompute,
       dispatch_dispatch_read_back_with_changing_pipelines_descriptor_sets)
{
  VkHandle<VkBuffer> buffer(1u);
  VkHandle<VkPipeline> pipeline_a(2u);
  VkHandle<VkPipeline> pipeline_b(3u);
  VkHandle<VkPipelineLayout> pipeline_layout(4u);
  VkHandle<VkDescriptorSet> descriptor_set_a(5u);
  VkHandle<VkDescriptorSet> descriptor_set_b(6u);

  resources.add_buffer(buffer);

  {
    VKResourceAccessInfo access_info = {};
    access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
    VKDispatchNode::CreateInfo dispatch_info(access_info);
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline = pipeline_a;
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline_layout = pipeline_layout;
    dispatch_info.dispatch_node.pipeline_data.vk_descriptor_set = descriptor_set_a;
    dispatch_info.dispatch_node.group_count_x = 1;
    dispatch_info.dispatch_node.group_count_y = 1;
    dispatch_info.dispatch_node.group_count_z = 1;
    render_graph->add_node(dispatch_info);
  }
  {
    VKResourceAccessInfo access_info = {};
    access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
    VKDispatchNode::CreateInfo dispatch_info(access_info);
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline = pipeline_b;
    dispatch_info.dispatch_node.pipeline_data.vk_pipeline_layout = pipeline_layout;
    dispatch_info.dispatch_node.pipeline_data.vk_descriptor_set = descriptor_set_b;
    dispatch_info.dispatch_node.group_count_x = 2;
    dispatch_info.dispatch_node.group_count_y = 2;
    dispatch_info.dispatch_node.group_count_z = 2;
    render_graph->add_node(dispatch_info);
  }
  submit(render_graph, command_buffer);
  EXPECT_EQ(7, log.size());
  EXPECT_EQ("bind_pipeline(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, pipeline=0x2)",
            log[0]);
  EXPECT_EQ(
      "bind_descriptor_sets(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, layout=0x4, "
      "p_descriptor_sets=0x5)",
      log[1]);
  EXPECT_EQ("dispatch(group_count_x=1, group_count_y=1, group_count_z=1)", log[2]);
  EXPECT_EQ(
      "pipeline_barrier(src_stage_mask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "
      "dst_stage_mask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT" +
          endl() +
          " - buffer_barrier(src_access_mask=VK_ACCESS_SHADER_WRITE_BIT, "
          "dst_access_mask=VK_ACCESS_SHADER_WRITE_BIT, buffer=0x1, offset=0, "
          "size=18446744073709551615)" +
          endl() + ")",
      log[3]);
  EXPECT_EQ("bind_pipeline(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, pipeline=0x3)",
            log[4]);
  EXPECT_EQ(
      "bind_descriptor_sets(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, layout=0x4, "
      "p_descriptor_sets=0x6)",
      log[5]);
  EXPECT_EQ("dispatch(group_count_x=2, group_count_y=2, group_count_z=2)", log[6]);
}

/**
 * Test dispatch indirect
 */
TEST_F(VKRenderGraphTestCompute, dispatch_indirect_read_back)
{
  VkHandle<VkBuffer> buffer(1u);
  VkHandle<VkBuffer> command_buffer(2u);
  VkHandle<VkPipeline> pipeline(3u);
  VkHandle<VkPipelineLayout> pipeline_layout(4u);
  VkHandle<VkDescriptorSet> descriptor_set(5u);

  resources.add_buffer(buffer);
  resources.add_buffer(command_buffer);

  VKResourceAccessInfo access_info = {};
  access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
  VKDispatchIndirectNode::CreateInfo dispatch_indirect_info(access_info);
  dispatch_indirect_info.dispatch_indirect_node.pipeline_data.vk_pipeline = pipeline;
  dispatch_indirect_info.dispatch_indirect_node.pipeline_data.vk_pipeline_layout = pipeline_layout;
  dispatch_indirect_info.dispatch_indirect_node.pipeline_data.vk_descriptor_set = descriptor_set;
  dispatch_indirect_info.dispatch_indirect_node.buffer = command_buffer;
  dispatch_indirect_info.dispatch_indirect_node.offset = 0;
  render_graph->add_node(dispatch_indirect_info);
  submit(render_graph, this->command_buffer);
  EXPECT_EQ(3, log.size());
  EXPECT_EQ("bind_pipeline(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, pipeline=0x3)",
            log[0]);
  EXPECT_EQ(
      "bind_descriptor_sets(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, layout=0x4, "
      "p_descriptor_sets=0x5)",
      log[1]);
  EXPECT_EQ("dispatch_indirect(buffer=0x2, offset=0)", log[2]);
}

TEST_F(VKRenderGraphTestCompute, dispatch_indirect_dispatch_indirect_read_back)
{
  VkHandle<VkBuffer> buffer(1u);
  VkHandle<VkBuffer> command_buffer(2u);
  VkHandle<VkPipeline> pipeline(3u);
  VkHandle<VkPipelineLayout> pipeline_layout(4u);
  VkHandle<VkDescriptorSet> descriptor_set(5u);

  resources.add_buffer(buffer);
  resources.add_buffer(command_buffer);

  {
    VKResourceAccessInfo access_info = {};
    access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
    VKDispatchIndirectNode::CreateInfo dispatch_indirect_info(access_info);
    dispatch_indirect_info.dispatch_indirect_node.pipeline_data.vk_pipeline = pipeline;
    dispatch_indirect_info.dispatch_indirect_node.pipeline_data.vk_pipeline_layout =
        pipeline_layout;
    dispatch_indirect_info.dispatch_indirect_node.pipeline_data.vk_descriptor_set = descriptor_set;
    dispatch_indirect_info.dispatch_indirect_node.buffer = command_buffer;
    dispatch_indirect_info.dispatch_indirect_node.offset = 0;
    render_graph->add_node(dispatch_indirect_info);
  }
  {
    VKResourceAccessInfo access_info = {};
    access_info.buffers.append({buffer, VK_ACCESS_SHADER_WRITE_BIT});
    VKDispatchIndirectNode::CreateInfo dispatch_indirect_info(access_info);
    dispatch_indirect_info.dispatch_indirect_node.pipeline_data.vk_pipeline = pipeline;
    dispatch_indirect_info.dispatch_indirect_node.pipeline_data.vk_pipeline_layout =
        pipeline_layout;
    dispatch_indirect_info.dispatch_indirect_node.pipeline_data.vk_descriptor_set = descriptor_set;
    dispatch_indirect_info.dispatch_indirect_node.buffer = command_buffer;
    dispatch_indirect_info.dispatch_indirect_node.offset = 12;
    render_graph->add_node(dispatch_indirect_info);
  }
  submit(render_graph, this->command_buffer);
  EXPECT_EQ(5, log.size());

  EXPECT_EQ("bind_pipeline(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, pipeline=0x3)",
            log[0]);
  EXPECT_EQ(
      "bind_descriptor_sets(pipeline_bind_point=VK_PIPELINE_BIND_POINT_COMPUTE, layout=0x4, "
      "p_descriptor_sets=0x5)",
      log[1]);
  EXPECT_EQ("dispatch_indirect(buffer=0x2, offset=0)", log[2]);
  EXPECT_EQ(
      "pipeline_barrier(src_stage_mask=VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, "
      "VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "
      "dst_stage_mask=VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT" +
          endl() +
          " - buffer_barrier(src_access_mask=VK_ACCESS_SHADER_WRITE_BIT, "
          "dst_access_mask=VK_ACCESS_SHADER_WRITE_BIT, buffer=0x1, offset=0, "
          "size=18446744073709551615)" +
          endl() + ")",
      log[3]);
  EXPECT_EQ("dispatch_indirect(buffer=0x2, offset=12)", log[4]);
}

}  // namespace blender::gpu::render_graph
