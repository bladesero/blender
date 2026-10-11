/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "eevee_dlss.hh"
#include "DEG_depsgraph_query.hh"
#include "DNA_camera_types.h"
#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_texture.hh"
#include "MEM_guardedalloc.h"
#include "draw_view_data.hh"
#include "eevee_instance.hh"
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace blender::eevee {

DLSSModule::~DLSSModule()
{
  /* Finish SDK commands before releasing their input images or compiled preparation shaders. */
  context_.reset();
  GPU_shader_unbind();
  GPU_SHADER_FREE_SAFE(prepare_shader_);
  GPU_SHADER_FREE_SAFE(guide_shader_);
  GPU_SHADER_FREE_SAFE(hit_shader_);
}

void DLSSModule::init(int2 extent, bool full_viewport)
{
  const bool was_enabled = enabled_;
  unsupported_geometry_ = false;
  enabled_ = false;
  const bool viewport = inst_.is_viewport();
  const int mode = viewport ? inst_.scene->eevee.dlss_quality :
                              inst_.scene->eevee.dlss_render_quality;
  const bool request_rr = viewport ? inst_.scene->eevee.dlss_rr :
                                     inst_.scene->eevee.dlss_render_rr;
  const uint64_t update = DEG_get_update_count(inst_.depsgraph);
  if (mode != mode_ || request_rr != requested_rr_ ||
      (viewport && force_native_ && update != blocked_update_))
  {
    context_.reset();
    attempted_ = false;
    force_native_ = false;
    reset_ = true;
    has_result_ = false;
    frame_ = 0;
  }
  mode_ = mode;
  requested_rr_ = request_rr;
  if (!viewport) {
    /* A new output frame never borrows history, including when reusing an Instance. */
    frame_ = 0;
    reset_ = true;
    has_result_ = false;
  }
  if (mode == 0) {
    context_.reset();
    attempted_ = false;
    force_native_ = false;
    error_.clear();
  }
  else if (!force_native_ && !inst_.is_baking() && !inst_.is_viewport_image_render) {
    const auto *camera = inst_.camera_eval_object ? reinterpret_cast<const blender::Camera *>(
                                                        inst_.camera_eval_object->data) :
                                                    nullptr;
    const bool unsupported_view = viewport &&
                                  (inst_.is_viewport_compositor_enabled ||
                                   inst_.draw_ctx->is_viewport_xr() ||
                                   inst_.v3d->shading.render_pass != EEVEE_RENDER_PASS_COMBINED ||
                                   (inst_.use_studio_light() &&
                                    inst_.v3d->shading.studiolight_background < 1.0f));
    const bool unsupported_render =
        !viewport && (inst_.view_layer->passflag & ~SCE_PASS_COMBINED ||
                      inst_.view_layer->eevee.render_passes & ~EEVEE_RENDER_PASS_COMBINED ||
                      inst_.view_layer->cryptomatte_flag &
                          (VIEW_LAYER_CRYPTOMATTE_OBJECT | VIEW_LAYER_CRYPTOMATTE_ASSET |
                           VIEW_LAYER_CRYPTOMATTE_MATERIAL) ||
                      !inst_.view_layer->aovs.is_empty() ||
                      (inst_.scene->r.scemode & R_MULTIVIEW) || (inst_.scene->r.mode & R_BORDER));
    if (!full_viewport || inst_.camera.is_panoramic() || inst_.camera.overscan() != 0.0f ||
        unsupported_view || unsupported_render || inst_.scene->r.alphamode == R_ALPHAPREMUL ||
        (inst_.scene->r.mode & R_MBLUR) || (camera && (camera->dof.flag & CAM_DOF_ENABLED)))
    {
      report("DLSS: unsupported effects or passes; using native EEVEE");
    }
    else {
      if (!attempted_) {
        attempted_ = true;
        rr_ = request_rr && (inst_.scene->eevee.flag & SCE_EEVEE_SSR_ENABLED) &&
              inst_.scene->eevee.ray_tracing_method == RAYTRACE_EEVEE_METHOD_HARDWARE;
        context_ = gpu::DLSS::create(
            error_, rr_ ? gpu::DLSSMode::RayReconstruction : gpu::DLSSMode::SuperResolution);
        if (!context_ && rr_) {
          report("DLSS RR unavailable: " + error_ + ". Using native denoising with SR");
          rr_ = false;
          context_ = gpu::DLSS::create(error_);
        }
      }
      int2 input_size;
      const gpu::DLSSQuality quality = mode == 1 ? gpu::DLSSQuality::Quality :
                                       mode == 2 ? gpu::DLSSQuality::Balanced :
                                                   gpu::DLSSQuality::Performance;
      if (context_ && context_->configure(extent, quality, input_size)) {
        enabled_ = true;
        if (input_size != input_size_ || extent != output_size_) {
          reset_ = true;
          has_result_ = false;
        }
        input_size_ = input_size;
        output_size_ = extent;
      }
      else {
        if (context_) {
          error_ = context_->error();
        }
        report("DLSS: " + error_ + ". Using native EEVEE");
      }
    }
  }
  if (!viewport && enabled_) {
    inst_.sampling.restart();
    inst_.sampling.reconstruction_samples();
  }
  if (was_enabled != enabled_) {
    reset_ = true;
    has_result_ = false;
    if (viewport) {
      inst_.sampling.reset();
    }
  }
}

void DLSSModule::report(const std::string &message)
{
  if (inst_.is_viewport()) {
    inst_.info_append_i18n("{}", message);
  }
  else {
    fprintf(stderr, "%s\n", message.c_str());
  }
}

void DLSSModule::reject(const char *reason)
{
  if (!enabled_) {
    return;
  }
  report(std::string("DLSS: ") + reason + ". Restarting with native EEVEE");
  force_native_ = true;
  blocked_update_ = DEG_get_update_count(inst_.depsgraph);
  restart_ = true;
  enabled_ = false;
  has_result_ = false;
  context_.reset();
}

void DLSSModule::end_sync()
{
  if (!enabled_) {
    return;
  }
  const auto closures = inst_.pipelines.deferred.closure_bits_get();
  if (inst_.volume.will_enable() || inst_.pipelines.forward.reconstruction_unsupported() ||
      (closures & (CLOSURE_TRANSMISSION | CLOSURE_TRANSPARENCY | CLOSURE_HOLDOUT)) ||
      inst_.gpencil_engine_enabled())
  {
    reject("unsupported volume, transparency, transmission, SSS or forward shading");
    return;
  }
  if (rr_ && (unsupported_geometry_ || !inst_.raytracing.hardware.enabled() ||
              !inst_.raytracing.hardware.scene.ready() ||
              inst_.raytracing.hardware.scene.skipped_materials ||
              inst_.raytracing.hardware.scene.materials.unsupported))
  {
    reject("hardware RT material or geometry is outside the supported RR subset");
  }
}

float2 DLSSModule::jitter() const
{
  /* Independent of EEVEE's accumulation index, which resets during navigation. */
  auto radical_inverse = [](uint64_t index, uint64_t base) {
    float value = 0.0f, weight = 1.0f;
    while (index) {
      weight /= float(base);
      value += float(index % base) * weight;
      index /= base;
    }
    return value;
  };
  const uint64_t index = (frame_ % 32) + 1;
  return float2(radical_inverse(index, 2), radical_inverse(index, 3)) - 0.5f;
}

bool DLSSViewportAdapter::update_history(const Instance &inst)
{
  const CameraData &camera = inst.camera.data_get();
  const float time = inst.scene->r.cfra + inst.scene->r.subframe;
  const uint64_t update = DEG_get_update_count(inst.depsgraph);
  /* Preserve history for continuous camera navigation and consecutive animation frames.
   * Edits at a fixed timeline position have no reliable lighting/material motion model. */
  const bool reset = (camera_ != inst.camera_orig_object || projection_ != camera.winmat ||
                      (update != depsgraph_update_ && time == scene_time_) ||
                      fabsf(time - scene_time_) > 1.0f || time < scene_time_ ||
                      math::dot(float3(view_inverse_[2]), float3(camera.viewinv[2])) < 0.5f ||
                      math::distance(float3(view_inverse_[3]), float3(camera.viewinv[3])) >
                          max_ff(1.0f, inst.rv3d ? inst.rv3d->dist : 1.0f));
  camera_ = inst.camera_orig_object;
  projection_ = camera.winmat;
  view_inverse_ = camera.viewinv;
  depsgraph_update_ = update;
  scene_time_ = time;
  return reset;
}

void DLSSModule::render(draw::View &view, gpu::Texture *input)
{
  const CameraData &camera = inst_.camera.data_get();
  if (inst_.is_viewport()) {
    reset_ |= viewport_adapter_.update_history(inst_);
  }
  if (rr_ && hardware_revision_ != inst_.raytracing.hardware.history_revision()) {
    reset_ = true;
    hardware_revision_ = inst_.raytracing.hardware.history_revision();
  }

  if (!prepare_shader_) {
    prepare_shader_ = GPU_shader_create_from_info_name("eevee_dlss_prepare");
  }
  if (!prepare_shader_) {
    reject("reconstruction shader compilation failed");
    return;
  }
  color_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, input_size_);
  depth_.ensure_2d(gpu::TextureFormat::SFLOAT_32, input_size_);
  motion_.ensure_2d(gpu::TextureFormat::SFLOAT_32_32, input_size_);
  output_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, output_size_);

  last_jitter_ = jitter();
  prepare_.init();
  prepare_.shader_set(prepare_shader_);
  prepare_.bind_texture("input_color", input);
  prepare_.bind_texture("input_depth", inst_.render_buffers.depth_tx);
  prepare_.bind_texture("input_vector", inst_.render_buffers.vector_tx);
  prepare_.bind_ubo("camera_prev", &(*inst_.velocity.camera_steps[STEP_PREVIOUS]));
  prepare_.bind_ubo("camera_curr", &(*inst_.velocity.camera_steps[STEP_CURRENT]));
  prepare_.bind_ubo("camera_next", &(*inst_.velocity.camera_steps[STEP_PREVIOUS]));
  prepare_.push_constant("static_frame", !inst_.is_viewport());
  prepare_.bind_image("sr_color", color_);
  prepare_.bind_image("sr_depth", depth_);
  prepare_.bind_image("sr_motion", motion_);
  prepare_.dispatch(int3(math::divide_ceil(input_size_, int2(8)), 1));
  inst_.manager->submit(prepare_, view);

  /* window_translate_m4 shifts projected geometry by +jitter in Blender's bottom-left
   * coordinates. NGX wants that projected displacement, with Y flipped to top-left. */
  gpu::DLSSFrame frame;
  frame.color = color_;
  frame.depth = depth_;
  frame.motion = motion_;
  frame.output = output_;
  frame.jitter = float2(last_jitter_.x, -last_jitter_.y);
  frame.reset = reset_;
  if (rr_) {
    frame.diffuse_albedo = diffuse_;
    frame.specular_albedo = specular_;
    frame.normal_roughness = normals_;
    frame.specular_hit_distance = hit_;
    frame.world_to_view = camera.viewmat;
    frame.view_to_clip = camera.winmat;
    /* Match the conventional [0, 1] hardware depth sent to NGX. EEVEE's camera
     * projection
     * stores OpenGL [-1, 1] NDC Z, before the backend depth remap. */
    for (int column = 0; column < 4; column++) {
      frame.view_to_clip[column][2] = 0.5f * (camera.winmat[column][2] + camera.winmat[column][3]);
    }
  }
  if (context_->evaluate(frame)) {
    has_result_ = true;
    reset_ = false;
    frame_++;
  }
  if (frame_ == 1) {
    dump_guides();
  }
  if (inst_.is_viewport()) {
    display();
  }
  else if (has_result_ && !render_adapter_.resolve(inst_, output_, output_size_)) {
    reject("final reconstruction output shader compilation failed");
  }
}

void DLSSModule::begin_trace()
{
  if (!ray_reconstruction()) {
    return;
  }
  hit_.ensure_2d(gpu::TextureFormat::SFLOAT_32, input_size_);
  hit_.clear(float4(0));
  diffuse_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, input_size_);
  specular_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, input_size_);
  normals_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, input_size_);
  diffuse_.clear(float4(0));
  specular_.clear(float4(0));
  normals_.clear(float4(0, 0, 1, 1));
}

void DLSSModule::capture_hit(draw::View &view, gpu::Texture *hit, int closure)
{
  if (!ray_reconstruction()) {
    return;
  }
  if (!hit_shader_) {
    hit_shader_ = GPU_shader_create_from_info_name("eevee_dlss_hit");
  }
  if (!hit_shader_) {
    reject("RR hit-distance shader compilation failed");
    return;
  }
  hit_pass_.init();
  hit_pass_.shader_set(hit_shader_);
  inst_.gbuffer.bind_resources(hit_pass_);
  hit_pass_.bind_texture("ray_time", hit);
  hit_pass_.bind_image("rr_hit", hit_);
  hit_pass_.push_constant("closure_index", closure);
  hit_pass_.dispatch(int3(math::divide_ceil(input_size_, int2(8)), 1));
  inst_.manager->submit(hit_pass_, view);
}

void DLSSModule::prepare_guides(draw::View &view)
{
  if (!ray_reconstruction()) {
    return;
  }
  if (!guide_shader_) {
    guide_shader_ = GPU_shader_create_from_info_name("eevee_dlss_guides");
  }
  if (!guide_shader_) {
    reject("RR guide shader compilation failed");
    return;
  }
  diffuse_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, input_size_);
  specular_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, input_size_);
  normals_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, input_size_);
  guides_.init();
  guides_.shader_set(guide_shader_);
  inst_.gbuffer.bind_resources(guides_);
  guides_.bind_image("rr_diffuse", diffuse_);
  guides_.bind_image("rr_specular", specular_);
  guides_.bind_image("rr_normals", normals_);
  guides_.dispatch(int3(math::divide_ceil(input_size_, int2(8)), 1));
  inst_.manager->submit(guides_, view);
}

void DLSSModule::dump_guides()
{
  /* Explicit debug opt-in; production rendering never incurs these readbacks. */
  const char *directory = std::getenv("BLENDER_DLSS_DUMP_DIR");
  if (!directory || !*directory) {
    return;
  }
  GPU_finish();
  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  if (ec || failed()) {
    return;
  }
  auto dump = [&](const char *name, gpu::Texture *texture, int channels) {
    if (!texture) {
      return;
    }
    float *data = static_cast<float *>(GPU_texture_read(texture, GPU_DATA_FLOAT, 0));
    if (!data) {
      return;
    }
    const int width = GPU_texture_width(texture), height = GPU_texture_height(texture);
    const auto base = std::filesystem::path(directory) / name;
    std::ofstream meta(base.string() + ".txt");
    meta << width << " " << height << " " << channels << "\n";
    std::ofstream bytes(base.string() + ".f32", std::ios::binary);
    bytes.write(reinterpret_cast<char *>(data), size_t(width) * height * channels * sizeof(float));
    MEM_delete(data);
  };
  dump("color", color_, 4);
  dump("depth", depth_, 1);
  dump("motion", motion_, 2);
  dump("output", output_, 4);
  if (rr_) {
    dump("diffuse", diffuse_, 4);
    dump("specular", specular_, 4);
    dump("normal_roughness", normals_, 4);
    dump("specular_hit", hit_, 1);
  }
}

void DLSSModule::display()
{
  if (has_result_ &&
      !viewport_adapter_.display(inst_, output_, depth_, output_size_, last_jitter_))
  {
    reject("viewport reconstruction display shader compilation failed");
  }
}

DLSSViewportAdapter::~DLSSViewportAdapter()
{
  GPU_shader_unbind();
  GPU_SHADER_FREE_SAFE(shader_);
}

bool DLSSViewportAdapter::display(
    Instance &inst, gpu::Texture *color, gpu::Texture *depth, int2 extent, float2 jitter)
{
  if (!shader_) {
    shader_ = GPU_shader_create_from_info_name("eevee_dlss_display");
  }
  if (!shader_) {
    return false;
  }
  DefaultFramebufferList *framebuffers = inst.draw_ctx->viewport_framebuffer_list_get();
  GPU_framebuffer_bind(framebuffers->default_fb);
  GPU_framebuffer_viewport_set(framebuffers->default_fb, 0, 0, extent.x, extent.y);
  pass_.init();
  pass_.shader_set(shader_);
  pass_.state_set(DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_ALWAYS);
  pass_.bind_texture("sr_output", color);
  pass_.bind_texture("sr_depth", depth);
  pass_.push_constant("sr_jitter", jitter);
  pass_.draw_procedural(GPU_PRIM_TRIS, 1, 3);
  inst.manager->submit(pass_);
  return true;
}

DLSSRenderAdapter::~DLSSRenderAdapter()
{
  GPU_shader_unbind();
  GPU_SHADER_FREE_SAFE(shader_);
}

bool DLSSRenderAdapter::resolve(Instance &inst, gpu::Texture *color, int2 extent)
{
  if (!shader_) {
    shader_ = GPU_shader_create_from_info_name("eevee_dlss_output");
  }
  if (!shader_) {
    return false;
  }
  output_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, extent);
  pass_.init();
  pass_.shader_set(shader_);
  pass_.bind_texture("sr_output", color);
  pass_.bind_image("final_output", output_);
  pass_.dispatch(int3(math::divide_ceil(extent, int2(8)), 1));
  inst.manager->submit(pass_);
  return true;
}
}  // namespace blender::eevee
