/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "eevee_dlss.hh"
#include "DEG_depsgraph_query.hh"
#include "DNA_camera_types.h"
#include "DNA_material_types.h"
#include "GPU_material.hh"
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
  GPU_SHADER_FREE_SAFE(blend_shader_);
  GPU_SHADER_FREE_SAFE(integrate_shader_);
  GPU_SHADER_FREE_SAFE(sss_shader_);
  GPU_SHADER_FREE_SAFE(before_transparency_shader_);
}

void DLSSModule::init(int2 extent, bool full_viewport)
{
  const bool was_enabled = enabled_;
  unsupported_geometry_ = false;
  enabled_ = false;
  reasons_.clear();
  visible_size_ = extent;
  crop_ = int2(Film::overscan_pixels_get(inst_.camera.overscan(), extent));
  extent += crop_ * 2;
  const bool viewport = inst_.is_viewport();
  const int mode = viewport ? inst_.scene->eevee.dlss_quality :
                              inst_.scene->eevee.dlss_render_quality;
  const bool request_rr = viewport ? inst_.scene->eevee.dlss_rr :
                                     inst_.scene->eevee.dlss_render_rr;
  const bool request_hardware_rr = request_rr &&
                                   (inst_.scene->eevee.flag & SCE_EEVEE_SSR_ENABLED) &&
                                   inst_.scene->eevee.ray_tracing_method ==
                                       RAYTRACE_EEVEE_METHOD_HARDWARE;
  const uint64_t update = DEG_get_update_count(inst_.depsgraph);
  if (request_hardware_rr != hardware_rr_requested_ || mode != mode_ ||
      request_rr != requested_rr_ || (viewport && force_native_ && update != blocked_update_))
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
  hardware_rr_requested_ = request_hardware_rr;
  if (!viewport) {
    /* A new output frame never borrows history, including when reusing an Instance. */
    frame_ = 0;
    blend_samples_ = 0;
    integrated_samples_ = 0;
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
    auto block = [&](bool condition, const char *code, const std::string &detail) {
      if (condition) {
        reason(code, detail);
      }
    };
    block(!full_viewport, "REGION", "partial render region is unsupported");
    block(inst_.camera.is_panoramic(), "PANORAMA", "panoramic camera is unsupported");
    block(viewport && inst_.is_viewport_compositor_enabled,
          "VIEW_COMPOSITOR",
          "disable viewport compositor");
    block(viewport && inst_.draw_ctx->is_viewport_xr(), "XR", "XR view is unsupported");
    block(viewport && inst_.v3d->shading.render_pass != EEVEE_RENDER_PASS_COMBINED,
          "VIEW_PASS",
          "viewport must display Combined");
    block(viewport && inst_.use_studio_light() && inst_.v3d->shading.studiolight_background < 1.0f,
          "STUDIO_ALPHA",
          "studio background must be opaque");
    if (!viewport) {
      const std::string layer = inst_.view_layer->name;
      /* Film resolves legacy flags and removes obsolete passes before initializing DLSS.
       * Saved Cycles-only passes are not EEVEE outputs and must not block reconstruction.
       * Do not use enabled_passes_get(): it also includes DLSS's internal motion vectors. */
      const auto data_passes = inst_.film.output_passes_get() &
                               ~(EEVEE_RENDER_PASS_COMBINED |
                                 EEVEE_RENDER_PASS_CRYPTOMATTE_OBJECT |
                                 EEVEE_RENDER_PASS_CRYPTOMATTE_ASSET |
                                 EEVEE_RENDER_PASS_CRYPTOMATTE_MATERIAL);
      std::string pass_names;
      for (int bit = 0; bit <= EEVEE_RENDER_PASS_MAX_BIT; bit++) {
        const auto pass = eViewLayerEEVEEPassType(data_passes & (1 << bit));
        if (pass != 0) {
          for (const std::string &name : Film::pass_to_render_pass_names(pass, inst_.view_layer)) {
            if (!pass_names.empty()) {
              pass_names += ", ";
            }
            pass_names += name;
          }
        }
      }

      block(data_passes != 0,
            "DATA_PASSES",
            "View Layer '" + layer + "': only Combined is supported; enabled data passes: " +
                pass_names);
      block(inst_.view_layer->cryptomatte_flag &
                (VIEW_LAYER_CRYPTOMATTE_OBJECT | VIEW_LAYER_CRYPTOMATTE_ASSET |
                 VIEW_LAYER_CRYPTOMATTE_MATERIAL),
            "CRYPTOMATTE",
            "View Layer '" + layer + "': Cryptomatte is enabled");
      block(!inst_.view_layer->aovs.is_empty(),
            "AOV",
            "View Layer '" + layer + "': AOVs are enabled");
      block(inst_.scene->r.scemode & R_MULTIVIEW, "MULTIVIEW", "multiview is enabled");
      block(inst_.scene->r.mode & R_BORDER, "BORDER", "render border is enabled");
    }
    block(inst_.scene->r.alphamode == R_ALPHAPREMUL,
          "FILM_ALPHA",
          "transparent film is unsupported");
    block(inst_.scene->r.mode & R_MBLUR, "MOTION_BLUR", "motion blur is enabled");
    block(
        camera && (camera->dof.flag & CAM_DOF_ENABLED), "DOF", "camera depth of field is enabled");
    if (!reasons_.is_empty()) {
      /* Material diagnostics are collected during sync even when settings already block SR. */
    }
    else {
      if (!attempted_) {
        attempted_ = true;
        rr_ = request_hardware_rr;
        if (request_rr && !rr_) {
          report(
              "DLSS [RR_REQUIRES_HWRT]: requested RR; using SR because hardware ray tracing is "
              "disabled");
        }
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
  settings_reasons_ = reasons_;
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

void DLSSModule::reason(const char *code, const std::string &detail)
{
  const std::string text = "[" + std::string(code) + "] " + detail;
  if (!reasons_.contains(text)) {
    reasons_.append(text);
  }
}

void DLSSModule::material_check(const Object *object,
                                const blender::Material *material,
                                GPUMaterial *gpu)
{
  if (mode_ == 0 || !gpu) {
    return;
  }
  const std::string owner = std::string(object->id.name + 2) + " / " +
                            std::string(material->id.name + 2);
  auto flag = [&](eGPUMaterialFlag flag, const char *code, const char *message) {
    if (GPU_material_flag_get(gpu, flag)) {
      reason(code, owner + ": " + message);
    }
  };
  if (!requested_rr_) {
    flag(GPU_MATFLAG_REFRACT, "MATERIAL_REFRACTION", "refraction is unsupported by SR");
  }
  flag(GPU_MATFLAG_HOLDOUT, "MATERIAL_HOLDOUT", "holdout is unsupported");
  flag(GPU_MATFLAG_SHADER_TO_RGBA, "MATERIAL_SHADER_RGB", "Shader to RGB is unsupported");
}

void DLSSModule::report(const std::string &message)
{
  if (inst_.is_viewport()) {
    const size_t first = message.find('\n');
    const size_t second = first == std::string::npos ? first : message.find('\n', first + 1);
    inst_.info_append_i18n("{}",
                           second == std::string::npos ?
                               message :
                               message.substr(0, second) + " (more reasons in console)");
  }
  if (message != last_report_) {
    fprintf(stderr, "%s\n", message.c_str());
    last_report_ = message;
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
  if (mode_ == 0) {
    last_report_.clear();
    return;
  }
  if (inst_.materials.queued_shaders_count > 0 || inst_.materials.queued_textures_count > 0) {
    return;
  }
  const auto closures = inst_.pipelines.deferred.closure_bits_get();
  if (inst_.volume.will_enable()) {
    reason("VOLUME", "volume rendering is unsupported");
  }
  if (inst_.gpencil_engine_enabled()) {
    reason("GREASE_PENCIL", "Grease Pencil is unsupported");
  }
  if (!rr_ && (closures & CLOSURE_REFRACTION)) {
    reason("REFRACTION", "refractive surface is unsupported");
  }
  if (closures & CLOSURE_HOLDOUT || inst_.pipelines.forward.has_holdout()) {
    reason("HOLDOUT", "holdout surface is unsupported");
  }

  if (!rr_ && inst_.pipelines.forward.has_transparent() && inst_.pipelines.has_raycast) {
    reason("BLENDED_RAYCAST", "Blended materials with screen ray-cast nodes are unsupported");
  }
  if (rr_) {
    if (unsupported_geometry_) {
      reason("RR_GEOMETRY", "RR does not support this geometry type");
    }
    if (!inst_.raytracing.hardware.enabled() || !inst_.raytracing.hardware.scene.ready()) {
      reason("RR_HWRT", "hardware ray tracing scene is unavailable");
    }
  }
  if (!reasons_.is_empty()) {
    std::string message = "requested " + std::string(requested_rr_ ? "RR" : "SR") +
                          "; actual Native";
    for (const std::string &r : reasons_) {
      message += "\n  " + r;
    }
    if (enabled_) {
      reject(message.c_str());
    }
    else {
      report("DLSS: " + message);
    }
  }
  else if (enabled_) {
    std::string message = std::string("DLSS: ") + (rr_ ? "RR active; mixed tracing" : "SR active");
    if (rr_) {
      for (const std::string &detail : inst_.raytracing.hardware.scene.materials.diagnostics) {
        message += "\n  [LOCAL_SCREEN_PROBE] " + detail;
      }
    }
    else if (output_blended()) {
      message += "; Blended at output resolution";
    }
    report(message);
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
  has_composite_ = false;
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
  /* SR is an antialiasing reconstructor, not a stochastic transparency denoiser.
   * Preserve static-frame Monte Carlo convergence before reconstruction. The color is
   * accumulated on an unjittered grid, then sampled at the current jitter by prepare. */
  bool integrated = !rr_ && (inst_.pipelines.deferred.closure_bits_get() &
                             (CLOSURE_TRANSPARENCY | CLOSURE_SSS | CLOSURE_TRANSLUCENT));
  if (integrated) {
    if (!integrate_shader_) {
      integrate_shader_ = GPU_shader_create_from_info_name("eevee_dlss_integrate");
    }
    if (!integrate_shader_) {
      reject("static color integration shader compilation failed");
      return;
    }
    const uint64_t update = DEG_get_update_count(inst_.depsgraph);
    const float time = inst_.scene->r.cfra + inst_.scene->r.subframe;
    if (reset_ || update != integrated_update_ || time != integrated_time_ ||
        (inst_.is_viewport() && inst_.velocity.camera_has_motion()))
    {
      integrated_samples_ = 0;
    }
    integrated_update_ = update;
    integrated_time_ = time;
    if (integrated_color_.ensure_2d(gpu::TextureFormat::SFLOAT_32_32_32_32, input_size_)) {
      integrated_samples_ = 0;
    }
    integrate_pass_.init();
    integrate_pass_.shader_set(integrate_shader_);
    integrate_pass_.bind_texture("sample_color", input);
    integrate_pass_.bind_image("integrated_color", integrated_color_);
    integrate_pass_.push_constant("jitter", last_jitter_);
    integrate_pass_.push_constant("weight", 1.0f / float(++integrated_samples_));
    integrate_pass_.dispatch(int3(math::divide_ceil(input_size_, int2(8)), 1));
    inst_.manager->submit(integrate_pass_, view);
  }
  prepare_.init();
  prepare_.shader_set(prepare_shader_);
  prepare_.bind_texture("input_color", integrated ? integrated_color_ : input);
  prepare_.push_constant("integrated", integrated);
  prepare_.push_constant("jitter", last_jitter_);
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
  if (rr_ && inst_.is_viewport() &&
      (inst_.pipelines.forward.has_transparent() ||
       (inst_.pipelines.deferred.closure_bits_get() &
        (CLOSURE_TRANSPARENCY | CLOSURE_REFRACTION | CLOSURE_TRANSLUCENT))))
  {
    /* Optional SDK rejection inputs are not a verified correctness guarantee. Conservatively
     * reset when layered surfaces or their camera change; static samples still accumulate. */
    const uint64_t update = DEG_get_update_count(inst_.depsgraph);
    frame.reset |= update != layered_update_ || inst_.velocity.camera_has_motion();
    layered_update_ = update;
  }
  if (rr_) {
    frame.diffuse_albedo = diffuse_;
    frame.specular_albedo = specular_;
    frame.normal_roughness = normals_;
    frame.specular_hit_distance = hit_;
    frame.subsurface_guide = has_sss_guide_ ? static_cast<gpu::Texture *>(sss_guide_) : nullptr;
    frame.color_before_transparency = has_before_transparency_ ?
                                          static_cast<gpu::Texture *>(color_before_transparency_) :
                                          nullptr;
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
  const char *dump_sample = std::getenv("BLENDER_DLSS_DUMP_SAMPLE");
  if (frame_ == (dump_sample ? std::max<uint64_t>(1, std::strtoull(dump_sample, nullptr, 10)) : 1)) {
    dump_guides();
  }
}

bool DLSSModule::output_blended() const
{
  return enabled_ && !rr_ && inst_.pipelines.forward.has_transparent();
}

void DLSSModule::finish()
{
  if (!enabled_ || !has_result_) {
    return;
  }
  gpu::Texture *result = has_composite_ ? composite_ : output_;
  if (inst_.is_viewport()) {
    display();
  }
  else if (has_result_ && !render_adapter_.resolve(inst_, result, visible_size_, crop_)) {
    reject("final reconstruction output shader compilation failed");
  }
}

void DLSSModule::blend_layer(
    draw::View &view, gpu::Texture *r, gpu::Texture *g, gpu::Texture *b, bool colored, bool reset)
{
  if (!has_result_ || !enabled_) {
    return;
  }
  if (!blend_shader_) {
    blend_shader_ = GPU_shader_create_from_info_name("eevee_dlss_blend");
  }
  if (!blend_shader_) {
    reject("Blended resolve shader compilation failed");
    return;
  }
  if (reset || frame_ == 1 && blend_samples_ == 0) {
    blend_samples_ = 0;
  }
  bool resized = blend_radiance_.ensure_2d(gpu::TextureFormat::SFLOAT_32_32_32_32, output_size_);
  resized |= blend_transmittance_.ensure_2d(gpu::TextureFormat::SFLOAT_32_32_32_32, output_size_);
  if (resized) {
    blend_samples_ = 0;
  }
  composite_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, output_size_);
  blend_pass_.init();
  blend_pass_.shader_set(blend_shader_);
  blend_pass_.bind_texture("layer_r", r);
  blend_pass_.bind_texture("layer_g", g);
  blend_pass_.bind_texture("layer_b", b);
  blend_pass_.bind_texture("reconstructed", output_);
  blend_pass_.bind_image("mean_radiance", blend_radiance_);
  blend_pass_.bind_image("mean_transmittance", blend_transmittance_);
  blend_pass_.bind_image("composited", composite_);
  blend_pass_.push_constant("colored", colored);
  blend_pass_.push_constant("weight", 1.0f / float(++blend_samples_));
  blend_pass_.dispatch(int3(math::divide_ceil(output_size_, int2(8)), 1));
  inst_.manager->submit(blend_pass_, view);
  has_composite_ = true;
}

void DLSSModule::begin_trace()
{
  has_sss_guide_ = false;
  has_before_transparency_ = false;
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

void DLSSModule::capture_before_transparency(
    draw::View &view, gpu::Texture *r, gpu::Texture *g, gpu::Texture *b, bool colored)
{
  if (!enabled_ || (!rr_ && !((G.debug & G_DEBUG_GPU) && std::getenv("BLENDER_DLSS_DUMP_DIR")))) {
    return;
  }
  if (!before_transparency_shader_) {
    before_transparency_shader_ = GPU_shader_create_from_info_name(
        "eevee_dlss_before_transparency");
  }
  color_before_transparency_.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, input_size_);
  if (!before_transparency_shader_ || !color_before_transparency_.is_valid()) {
    reject("before-transparency guide allocation or shader compilation failed");
    return;
  }
  before_transparency_pass_.init();
  before_transparency_pass_.shader_set(before_transparency_shader_);
  before_transparency_pass_.bind_texture("background", inst_.render_buffers.combined_tx);
  before_transparency_pass_.bind_texture("layer_r", r);
  before_transparency_pass_.bind_texture("layer_g", colored ? g : r);
  before_transparency_pass_.bind_texture("layer_b", colored ? b : r);
  before_transparency_pass_.bind_image("before_color", color_before_transparency_);
  before_transparency_pass_.push_constant("colored", colored);
  before_transparency_pass_.dispatch(int3(math::divide_ceil(input_size_, int2(8)), 1));
  inst_.manager->submit(before_transparency_pass_, view);
  has_before_transparency_ = true;
}

void DLSSModule::capture_hit(draw::View &view, gpu::Texture *hit, gpu::Texture *path, int closure)
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
  hit_pass_.bind_texture("ray_path", path);
  hit_pass_.bind_image("rr_hit", hit_);
  hit_pass_.push_constant("closure_index", closure);
  hit_pass_.dispatch(int3(math::divide_ceil(input_size_, int2(8)), 1));
  inst_.manager->submit(hit_pass_, view);
}

bool DLSSModule::subsurface_guide_needed() const
{
  /* Allow guide diagnostics on the supported SR path without claiming secondary-hit SSS or
   *
   * enabling RR on an unsupported material. */
  return enabled_ && (rr_ || ((G.debug & G_DEBUG_GPU) && std::getenv("BLENDER_DLSS_DUMP_DIR")));
}

void DLSSModule::capture_subsurface(draw::View &view,
                                    gpu::Texture *direct,
                                    gpu::Texture *indirect,
                                    bool before)
{
  if (!subsurface_guide_needed()) {
    return;
  }
  if (!sss_shader_) {
    sss_shader_ = GPU_shader_create_from_info_name("eevee_dlss_subsurface");
  }
  if (!sss_shader_) {
    reject("SSS guide shader compilation failed");
    return;
  }
  sss_before_.ensure_2d(gpu::TextureFormat::SFLOAT_32_32_32_32, input_size_);
  sss_guide_.ensure_2d(gpu::TextureFormat::SFLOAT_16, input_size_);
  sss_pass_.init();
  sss_pass_.shader_set(sss_shader_);
  inst_.gbuffer.bind_resources(sss_pass_);
  sss_pass_.bind_texture("direct_light", direct);
  sss_pass_.bind_texture("indirect_light", indirect);
  sss_pass_.bind_image("before_light", sss_before_);
  sss_pass_.bind_image("sss_guide", sss_guide_);
  sss_pass_.push_constant("capture_before", before);
  sss_pass_.dispatch(int3(math::divide_ceil(input_size_, int2(8)), 1));
  inst_.manager->submit(sss_pass_, view);
  has_sss_guide_ = !before;
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
  if (has_before_transparency_) {
    dump("color_before_transparency", color_before_transparency_, 4);
  }
  dump("depth", depth_, 1);
  dump("motion", motion_, 2);
  dump("output", output_, 4);
  if (rr_) {
    dump("diffuse", diffuse_, 4);
    dump("specular", specular_, 4);
    dump("normal_roughness", normals_, 4);
    dump("specular_hit", hit_, 1);
  }
  if (has_sss_guide_) {
    dump("subsurface_before", sss_before_, 4);
    dump("subsurface_guide", sss_guide_, 1);
  }
}

void DLSSModule::display()
{
  if (has_result_ && !viewport_adapter_.display(inst_,
                                                has_composite_ ? composite_ : output_,
                                                depth_,
                                                visible_size_,
                                                crop_,
                                                last_jitter_))
  {
    reject("viewport reconstruction display shader compilation failed");
  }
}

DLSSViewportAdapter::~DLSSViewportAdapter()
{
  GPU_shader_unbind();
  GPU_SHADER_FREE_SAFE(shader_);
}

bool DLSSViewportAdapter::display(Instance &inst,
                                  gpu::Texture *color,
                                  gpu::Texture *depth,
                                  int2 extent,
                                  int2 crop,
                                  float2 jitter)
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
  pass_.push_constant("crop", crop);
  pass_.draw_procedural(GPU_PRIM_TRIS, 1, 3);
  inst.manager->submit(pass_);
  return true;
}

DLSSRenderAdapter::~DLSSRenderAdapter()
{
  GPU_shader_unbind();
  GPU_SHADER_FREE_SAFE(shader_);
}

bool DLSSRenderAdapter::resolve(Instance &inst, gpu::Texture *color, int2 extent, int2 crop)
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
  pass_.push_constant("crop", crop);
  pass_.dispatch(int3(math::divide_ceil(extent, int2(8)), 1));
  inst.manager->submit(pass_);
  return true;
}
}  // namespace blender::eevee
