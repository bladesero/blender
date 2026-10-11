/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <array>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <mutex>

#include "BLI_math_matrix.hh"
#include "GPU_state.hh"
#include "GPU_texture.hh"
#include "vk_backend.hh"
#include "vk_context.hh"
#include "vk_dlss.hh"
#include "vk_texture.hh"

#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>

/* RR Vulkan helpers require the SR Vulkan declarations above. */
#include <nvsdk_ngx_helpers_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>

namespace blender::gpu {
namespace {
std::mutex ngx_mutex;

/** Shared initialization, but separate feature/parameter maps for every viewport. */
struct NGXRuntime {
  VkDevice device = VK_NULL_HANDLE;
  ~NGXRuntime()
  {
    if (device) {
      NVSDK_NGX_VULKAN_Shutdown1(device);
    }
  }
};
std::weak_ptr<NGXRuntime> runtime_weak;

std::string ngx_error(NVSDK_NGX_Result result)
{
  char message[96];
  snprintf(message, sizeof(message), "NVIDIA DLSS failed (NGX 0x%08x)", uint32_t(result));
  return message;
}

NVSDK_NGX_Resource_VK resource(Texture *texture, bool write)
{
  auto &vk_texture = *static_cast<VKTexture *>(texture);
  const VKImageView &view = vk_texture.image_view_get(VKImageViewArrayed::NOT_ARRAYED,
                                                      VKImageViewFlags::DEFAULT);
  VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  return NVSDK_NGX_Create_ImageView_Resource_VK(view.vk_handle(),
                                                vk_texture.vk_image_handle(),
                                                range,
                                                view.vk_format(),
                                                GPU_texture_width(texture),
                                                GPU_texture_height(texture),
                                                write);
}
}  // namespace

struct VKDLSS::State {
  std::shared_ptr<NGXRuntime> runtime;
  NVSDK_NGX_Parameter *params = nullptr;
  NVSDK_NGX_Handle *handle = nullptr;
  NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
  DLSSMode mode = DLSSMode::SuperResolution;
  uint64_t evaluations = 0;
  VkQueryPool profile_pool = VK_NULL_HANDLE;
  uint32_t profile_count = 0;
  float timestamp_period = 0.0f;
  int2 input_size = int2(0);
  int2 output_size = int2(0);
  std::atomic<NVSDK_NGX_Result> result{NVSDK_NGX_Result_Success};
};

VKDLSS::VKDLSS() : state_(std::make_unique<State>()) {}

std::unique_ptr<DLSS> VKDLSS::create(std::string &error, DLSSMode mode)
{
  std::lock_guard lock(ngx_mutex);
  auto runtime = runtime_weak.lock();
  VKDevice &device = VKBackend::get().device;
  if (!runtime) {
    /* An application-owned project UUID, not an ID borrowed from another application. */
    std::error_code ec;
    const std::filesystem::path log_dir = std::filesystem::temp_directory_path(ec) /
                                          "blender-dlss";
    std::filesystem::create_directories(log_dir, ec);
    if (ec) {
      error = "Cannot create the NVIDIA DLSS log directory";
      return nullptr;
    }
    const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        "d7b87b5c-913c-4773-94bb-df53c09f6079",
        NVSDK_NGX_ENGINE_TYPE_CUSTOM,
        "Blender-EEVEE-SR-1",
        log_dir.c_str(),
        device.instance_get(),
        device.physical_device_get(),
        device.vk_handle(),
        vkGetInstanceProcAddr,
        vkGetDeviceProcAddr);
    if (NVSDK_NGX_FAILED(result)) {
      error = ngx_error(result);
      return nullptr;
    }
    runtime = std::make_shared<NGXRuntime>();
    runtime->device = device.vk_handle();
    runtime_weak = runtime;
  }
  if (runtime->device != device.vk_handle()) {
    error = "NVIDIA DLSS is already initialized on a different Vulkan device";
    return nullptr;
  }
  auto instance = std::unique_ptr<VKDLSS>(new VKDLSS());
  State &state = *instance->state_;
  state.runtime = runtime;
  state.mode = mode;
  NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&state.params);
  int available = 0;
  if (NVSDK_NGX_SUCCEED(result)) {
    result = state.params->Get(mode == DLSSMode::RayReconstruction ?
                                   NVSDK_NGX_Parameter_SuperSamplingDenoising_Available :
                                   NVSDK_NGX_Parameter_SuperSampling_Available,
                               &available);
  }
  if (NVSDK_NGX_FAILED(result) || !available) {
    error = NVSDK_NGX_FAILED(result)            ? ngx_error(result) :
            mode == DLSSMode::RayReconstruction ? "DLSS RR runtime or device support unavailable" :
                                                  "DLSS SR runtime or device support unavailable";
    if (state.params) {
      NVSDK_NGX_VULKAN_DestroyParameters(state.params);
      state.params = nullptr;
    }
    /* No commands were recorded. Do not run the GPU-waiting destructor under ngx_mutex. */
    state.runtime.reset();
    instance->state_.reset();
    return nullptr;
  }
  if (std::getenv("BLENDER_DLSS_PROFILE") &&
      device.physical_device_properties_get().limits.timestampComputeAndGraphics)
  {
    VkQueryPoolCreateInfo query_info = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = 128;
    if (vkCreateQueryPool(device.vk_handle(), &query_info, nullptr, &state.profile_pool) ==
        VK_SUCCESS)
    {
      state.timestamp_period = device.physical_device_properties_get().limits.timestampPeriod;
    }
  }
  return instance;
}

VKDLSS::~VKDLSS()
{
  if (!state_) {
    return;
  }
  /* The graph owns CPU callbacks only. NGX resources must outlive submitted GPU work. */
  GPU_finish();
  std::lock_guard lock(ngx_mutex);
  if (state_->profile_pool) {
    Vector<uint64_t> timestamps(state_->profile_count * 2);
    if (!timestamps.is_empty() &&
        vkGetQueryPoolResults(state_->runtime->device,
                              state_->profile_pool,
                              0,
                              timestamps.size(),
                              timestamps.size() * sizeof(uint64_t),
                              timestamps.data(),
                              sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
    {
      double total = 0.0, steady = 0.0;
      for (uint32_t i = 0; i < state_->profile_count; i++) {
        const double ms = double(timestamps[i * 2 + 1] - timestamps[i * 2]) *
                          state_->timestamp_period * 1e-6;
        total += ms;
        if (i > 0) {
          steady += ms;
        }
      }
      unsigned long long bytes = 0;
      if (state_->mode == DLSSMode::RayReconstruction) {
        NGX_DLSSD_GET_STATS(state_->params, &bytes);
      }
      else {
        NGX_DLSS_GET_STATS(state_->params, &bytes);
      }
      fprintf(stderr,
              "DLSS GPU profile: mode=%s, evaluations=%u, total_ms=%.3f, steady_mean_ms=%.3f, "
              "SDK_vram_bytes=%llu\n",
              state_->mode == DLSSMode::RayReconstruction ? "RR" : "SR",
              state_->profile_count,
              total,
              steady / std::max(1u, state_->profile_count - 1),
              bytes);
    }
    vkDestroyQueryPool(state_->runtime->device, state_->profile_pool, nullptr);
  }
  if (state_->handle) {
    NVSDK_NGX_VULKAN_ReleaseFeature(state_->handle);
  }
  if (state_->params) {
    NVSDK_NGX_VULKAN_DestroyParameters(state_->params);
  }
  state_.reset();
}

bool VKDLSS::configure(int2 output_size, DLSSQuality quality, int2 &input_size)
{
  State &state = *state_;
  if (output_size.x <= 0 || output_size.y <= 0) {
    state.result.store(NVSDK_NGX_Result_FAIL_InvalidParameter);
    return false;
  }
  const auto ngx_quality = quality == DLSSQuality::Quality ?
                               NVSDK_NGX_PerfQuality_Value_MaxQuality :
                           quality == DLSSQuality::Balanced ?
                               NVSDK_NGX_PerfQuality_Value_Balanced :
                               NVSDK_NGX_PerfQuality_Value_MaxPerf;
  if (NVSDK_NGX_FAILED(state.result.load())) {
    return false;
  }
  if (state.output_size != output_size || state.quality != ngx_quality) {
    GPU_finish();
    std::lock_guard lock(ngx_mutex);
    if (state.handle) {
      NVSDK_NGX_VULKAN_ReleaseFeature(state.handle);
      state.handle = nullptr;
    }
    unsigned int width = 0, height = 0, min_width = 0, min_height = 0, max_width = 0,
                 max_height = 0;
    float sharpness;
    const auto optimal_settings = state.mode == DLSSMode::RayReconstruction ?
                                      NGX_DLSSD_GET_OPTIMAL_SETTINGS :
                                      NGX_DLSS_GET_OPTIMAL_SETTINGS;
    const NVSDK_NGX_Result result = optimal_settings(state.params,
                                                     output_size.x,
                                                     output_size.y,
                                                     ngx_quality,
                                                     &width,
                                                     &height,
                                                     &max_width,
                                                     &max_height,
                                                     &min_width,
                                                     &min_height,
                                                     &sharpness);
    state.result.store(result);
    if (NVSDK_NGX_FAILED(result) || width == 0 || height == 0) {
      if (NVSDK_NGX_SUCCEED(result)) {
        state.result.store(NVSDK_NGX_Result_FAIL_InvalidParameter);
      }
      return false;
    }
    state.input_size = int2(width, height);
    state.output_size = output_size;
    state.quality = ngx_quality;
  }
  input_size = state.input_size;
  return true;
}

bool VKDLSS::evaluate(const DLSSFrame &frame)
{
  State *state = state_.get();
  if (NVSDK_NGX_FAILED(state->result.load())) {
    return false;
  }
  auto valid_size = [](Texture *texture, int2 size) {
    return texture && GPU_texture_width(texture) == size.x &&
           GPU_texture_height(texture) == size.y;
  };
  if (!valid_size(frame.color, state->input_size) || !valid_size(frame.depth, state->input_size) ||
      !valid_size(frame.motion, state->input_size) ||
      !valid_size(frame.output, state->output_size))
  {
    state->result.store(NVSDK_NGX_Result_FAIL_InvalidParameter);
    return false;
  }
  /* Snapshot descriptors and constants: graph recording can run after the next CPU update. */
  const auto color_resource = resource(frame.color, false);
  const auto depth_resource = resource(frame.depth, false);
  const auto motion_resource = resource(frame.motion, false);
  const auto output_resource = resource(frame.output, true);
  const bool rr = state->mode == DLSSMode::RayReconstruction;
  if (rr && (!valid_size(frame.diffuse_albedo, state->input_size) ||
             !valid_size(frame.specular_albedo, state->input_size) ||
             !valid_size(frame.normal_roughness, state->input_size) ||
             !valid_size(frame.specular_hit_distance, state->input_size)))
  {
    state->result.store(NVSDK_NGX_Result_FAIL_InvalidParameter);
    return false;
  }
  auto diffuse = rr ? resource(frame.diffuse_albedo, false) : NVSDK_NGX_Resource_VK{};
  auto specular = rr ? resource(frame.specular_albedo, false) : NVSDK_NGX_Resource_VK{};
  auto normals = rr ? resource(frame.normal_roughness, false) : NVSDK_NGX_Resource_VK{};
  auto hit_distance = rr ? resource(frame.specular_hit_distance, false) : NVSDK_NGX_Resource_VK{};
  std::array<NVSDK_NGX_Resource_VK, 7> auxiliary{};
  const std::array<Texture *, 7> auxiliary_textures = {frame.color_before_transparency,
                                                       frame.transparency_radiance,
                                                       frame.transparency_opacity,
                                                       frame.transparency_motion,
                                                       frame.subsurface_guide,
                                                       frame.disocclusion_mask,
                                                       frame.bias_current_color_mask};
  if (rr) {
    const bool separate = frame.transparency_input == DLSSTransparencyInput::SeparateLayer;
    if ((!separate && (frame.transparency_radiance || frame.transparency_opacity ||
                       frame.transparency_motion)) ||
        (separate && (!frame.transparency_radiance || !frame.transparency_opacity)) ||
        (frame.color_before_transparency &&
         GPU_texture_format(frame.color_before_transparency) != GPU_texture_format(frame.color)))
    {
      state->result.store(NVSDK_NGX_Result_FAIL_InvalidParameter);
      return false;
    }
    for (int i = 0; i < auxiliary.size(); i++) {
      Texture *texture = auxiliary_textures[i];
      if (texture) {
        if (!valid_size(texture, state->input_size)) {
          state->result.store(NVSDK_NGX_Result_FAIL_InvalidParameter);
          return false;
        }
        auxiliary[i] = resource(texture, false);
      }
    }
  }
  bool reset = frame.reset;
  const float2 jitter = frame.jitter;
  render_graph::VKExternalComputeData node;
  node.inputs = {color_resource.Resource.ImageViewInfo.Image,
                 depth_resource.Resource.ImageViewInfo.Image,
                 motion_resource.Resource.ImageViewInfo.Image};
  node.outputs = {output_resource.Resource.ImageViewInfo.Image};
  if (rr) {
    for (auto &r : {diffuse, specular, normals, hit_distance}) {
      node.inputs.append(r.Resource.ImageViewInfo.Image);
    }
  }
  if (rr) {
    for (const auto &r : auxiliary) {
      if (r.Resource.ImageViewInfo.Image != VK_NULL_HANDLE) {
        node.inputs.append(r.Resource.ImageViewInfo.Image);
      }
    }
  }
  node.record = [=](VkCommandBuffer command) mutable {
    std::lock_guard lock(ngx_mutex);
    const bool profile = state->profile_pool && state->profile_count < 64;
    if (profile) {
      vkCmdResetQueryPool(command, state->profile_pool, state->profile_count * 2, 2);
      vkCmdWriteTimestamp(command,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          state->profile_pool,
                          state->profile_count * 2);
    }
    NVSDK_NGX_Result result = state->result.load();
    if (!state->handle && NVSDK_NGX_SUCCEED(result)) {
      NVSDK_NGX_DLSS_Create_Params create = {};
      create.Feature.InWidth = state->input_size.x;
      create.Feature.InHeight = state->input_size.y;
      create.Feature.InTargetWidth = state->output_size.x;
      create.Feature.InTargetHeight = state->output_size.y;
      create.Feature.InPerfQualityValue = state->quality;
      create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
                                    NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                    NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
      for (const char *preset : {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                                 NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
                                 NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance})
      {
        state->params->Set(preset, unsigned(NVSDK_NGX_DLSS_Hint_Render_Preset_K));
      }
      if (rr) {
        NVSDK_NGX_DLSSD_Create_Params rc = {};
        rc.InWidth = state->input_size.x;
        rc.InHeight = state->input_size.y;
        rc.InTargetWidth = state->output_size.x;
        rc.InTargetHeight = state->output_size.y;
        rc.InPerfQualityValue = state->quality;
        rc.InFeatureCreateFlags = create.InFeatureCreateFlags;
        rc.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
        rc.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_HW;
        for (const char *preset :
             {NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
              NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
              NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance})
        {
          state->params->Set(preset, unsigned(NVSDK_NGX_RayReconstruction_Hint_Render_Preset_D));
        }
        result = NGX_VULKAN_CREATE_DLSSD_EXT1(
            state->runtime->device, command, 1, 1, &state->handle, state->params, &rc);
      }
      else {
        result = NGX_VULKAN_CREATE_DLSS_EXT1(
            state->runtime->device, command, 1, 1, &state->handle, state->params, &create);
      }
      reset = true;
    }
    if (NVSDK_NGX_SUCCEED(result)) {
      /* Opt-in fault injection for the final-frame native retry acceptance test. */
      if (G.debug & G_DEBUG_GPU) {
        const char *fail_at = std::getenv("BLENDER_DLSS_TEST_FAIL_AT");
        if (fail_at && state->evaluations == std::strtoull(fail_at, nullptr, 10)) {
          result = NVSDK_NGX_Result_FAIL_InvalidParameter;
        }
      }
    }
    if (NVSDK_NGX_SUCCEED(result)) {
      auto c = color_resource, d = depth_resource, m = motion_resource, o = output_resource;
      NVSDK_NGX_VK_DLSS_Eval_Params evaluate = {};
      evaluate.Feature.pInColor = &c;
      evaluate.Feature.pInOutput = &o;
      evaluate.pInDepth = &d;
      evaluate.pInMotionVectors = &m;
      evaluate.InJitterOffsetX = jitter.x;
      evaluate.InJitterOffsetY = jitter.y;
      evaluate.InMVScaleX = evaluate.InMVScaleY = 1.0f;
      evaluate.InPreExposure = evaluate.InExposureScale = 1.0f;
      evaluate.InRenderSubrectDimensions = {unsigned(state->input_size.x),
                                            unsigned(state->input_size.y)};
      evaluate.InReset = reset;
      if (rr) {
        NVSDK_NGX_VK_DLSSD_Eval_Params re = {};
        re.pInColor = &c;
        re.pInOutput = &o;
        re.pInDepth = &d;
        re.pInMotionVectors = &m;
        re.pInDiffuseAlbedo = &diffuse;
        re.pInSpecularAlbedo = &specular;
        re.pInNormals = &normals;
        re.pInSpecularHitDistance = &hit_distance;
        auto optional = [&](int i) -> NVSDK_NGX_Resource_VK * {
          return auxiliary[i].Resource.ImageViewInfo.Image != VK_NULL_HANDLE ? &auxiliary[i] :
                                                                               nullptr;
        };
        re.pInColorBeforeTransparency = optional(0);
        re.pInTransparencyLayer = optional(1);
        re.pInTransparencyLayerOpacity = optional(2);
        re.pInTransparencyLayerMvecs = optional(3);
        re.pInScreenSpaceSubsurfaceScatteringGuide = optional(4);
        re.pInDisocclusionMask = optional(5);
        re.pInBiasCurrentColorMask = optional(6);
        re.InJitterOffsetX = jitter.x;
        re.InJitterOffsetY = jitter.y;
        re.InMVScaleX = re.InMVScaleY = 1.0f;
        re.InPreExposure = re.InExposureScale = 1.0f;
        /* A reconstruction sample has a fixed logical timestep. In particular, final-render
         * denoising must not depend on machine speed or on a zero/unspecified frame interval. */
        re.InFrameTimeDeltaInMsec = 1000.0f / 60.0f;
        re.InRenderSubrectDimensions = evaluate.InRenderSubrectDimensions;
        re.InReset = reset;
        /* NGX uses row-major matrices. Blender stores columns. */
        float4x4 world_to_view = math::transpose(frame.world_to_view);
        float4x4 view_to_clip = math::transpose(frame.view_to_clip);
        re.pInWorldToViewMatrix = world_to_view.base_ptr();
        re.pInViewToClipMatrix = view_to_clip.base_ptr();
        result = NGX_VULKAN_EVALUATE_DLSSD_EXT(command, state->handle, state->params, &re);
      }
      else {
        result = NGX_VULKAN_EVALUATE_DLSS_EXT(command, state->handle, state->params, &evaluate);
      }
    }
    if (profile) {
      vkCmdWriteTimestamp(command,
                          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          state->profile_pool,
                          state->profile_count * 2 + 1);
      state->profile_count++;
    }
    if (G.debug & G_DEBUG_GPU) {
      fprintf(stderr,
              "DLSS %s: %dx%d -> %dx%d, reset=%d, NGX=0x%08x, frame=%llu, viewport=%p\n",
              rr ? "RR" : "SR",
              state->input_size.x,
              state->input_size.y,
              state->output_size.x,
              state->output_size.y,
              int(reset),
              uint32_t(result),
              static_cast<unsigned long long>(state->evaluations++),
              static_cast<void *>(state));
    }
    state->result.store(result);
    if (NVSDK_NGX_FAILED(result)) {
      /* Always produce a defined image for this frame. Next frame returns to native EEVEE.
       * GENERAL is preserved, as promised to the render graph. */
      VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(command,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0,
                           1,
                           &barrier,
                           0,
                           nullptr,
                           0,
                           nullptr);
      VkImageBlit blit = {};
      blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      blit.srcOffsets[1] = {state->input_size.x, state->input_size.y, 1};
      blit.dstOffsets[1] = {state->output_size.x, state->output_size.y, 1};
      vkCmdBlitImage(command,
                     color_resource.Resource.ImageViewInfo.Image,
                     VK_IMAGE_LAYOUT_GENERAL,
                     output_resource.Resource.ImageViewInfo.Image,
                     VK_IMAGE_LAYOUT_GENERAL,
                     1,
                     &blit,
                     VK_FILTER_LINEAR);
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(command,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           0,
                           1,
                           &barrier,
                           0,
                           nullptr,
                           0,
                           nullptr);
      fprintf(stderr, "%s\n", ngx_error(result).c_str());
    }
  };
  VKContext::get()->render_graph().add_node(node);
  return true;
}

bool VKDLSS::failed() const
{
  return NVSDK_NGX_FAILED(state_->result.load());
}

std::string VKDLSS::error() const
{
  return ngx_error(state_->result.load());
}
}  // namespace blender::gpu
