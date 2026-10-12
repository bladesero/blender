/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "DRW_gpu_wrapper.hh"
#include "GPU_dlss.hh"
#include "draw_pass.hh"

namespace blender {
struct Object;
struct Material;
struct GPUMaterial;
}  // namespace blender

namespace blender::eevee {

enum class ReconstructionState { Off, Preparing, Active, Unavailable, Failed };
class Instance;

/** Window-dependent history and presentation, never used for final rendering. */
class DLSSViewportAdapter {
  uint64_t depsgraph_update_ = 0;
  float scene_time_ = 0.0f;
  const Object *camera_ = nullptr;
  float4x4 projection_ = float4x4::identity();
  float4x4 view_inverse_ = float4x4::identity();
  gpu::Shader *shader_ = nullptr;
  draw::PassSimple pass_{"DLSS.Display"};

 public:
  ~DLSSViewportAdapter();
  bool update_history(const Instance &inst, std::string &reason);
  bool display(Instance &inst,
               gpu::Texture *color,
               gpu::Texture *depth,
               int2 extent,
               int2 crop,
               float2 jitter);
};

/** Window-independent conversion to Film's bottom-left scene-linear Combined texture. */
class DLSSRenderAdapter {
  draw::Texture output_{"DLSS.Final"};
  gpu::Shader *shader_ = nullptr;
  draw::PassSimple pass_{"DLSS.Output"};

 public:
  ~DLSSRenderAdapter();
  bool resolve(Instance &inst, gpu::Texture *color, int2 extent, int2 crop);
  gpu::Texture *texture()
  {
    return output_;
  }
};

/** EEVEE reconstruction shared by viewport and frame-independent image rendering. */
class DLSSModule {
  Instance &inst_;
  std::unique_ptr<gpu::DLSS> context_;
  std::string error_;
  std::string last_report_;
  std::string reset_reason_ = "activation";
  Vector<std::string> reasons_, settings_reasons_;
  bool attempted_ = false;
  bool enabled_ = false;
  bool reset_ = true;
  bool has_result_ = false;
  ReconstructionState state_ = ReconstructionState::Off;
  bool compiling_ = false;
  uint64_t sample_ = 0;
  bool history_valid_ = false;
  float2 previous_jitter_ = float2(0);
  draw::Texture guide_depth_{"DLSS.ReferenceDepth"}, guide_motion_{"DLSS.ReferenceMotion"};
  draw::Framebuffer guide_fb_{"DLSS.Reference"};
  draw::Texture bias_{"DLSS.CurrentColorBias"};
  draw::Texture history_color_{"DLSS.PreviousColor"}, history_depth_{"DLSS.PreviousDepth"};
  draw::Texture coverage_{"DLSS.Coverage"};
  draw::Texture effect_base_{"DLSS.BeforeVolume"}, effect_risk_{"DLSS.EffectRisk"};
  gpu::Shader *risk_shader_ = nullptr;
  draw::PassSimple risk_pass_{"DLSS.RefractionRisk"};
  gpu::Shader *alpha_shader_ = nullptr;
  draw::PassSimple alpha_pass_{"DLSS.Alpha"};
  int mode_ = 0;
  bool rr_ = false, requested_rr_ = false;
  bool hardware_rr_requested_ = false;
  bool force_native_ = false, restart_ = false;
  bool unsupported_geometry_ = false;
  uint64_t blocked_update_ = 0, hardware_revision_ = 0;
  draw::Texture diffuse_{"DLSS.DiffuseAlbedo"}, specular_{"DLSS.SpecularAlbedo"};
  draw::Texture normals_{"DLSS.NormalRoughness"}, hit_{"DLSS.SpecularHitDistance"};
  DLSSViewportAdapter viewport_adapter_;
  DLSSRenderAdapter render_adapter_;
  draw::Texture sss_before_{"DLSS.SSSBefore"}, sss_guide_{"DLSS.SSSGuide"};
  gpu::Shader *sss_shader_ = nullptr;
  draw::PassSimple sss_pass_{"DLSS.SSSGuide"};
  bool has_sss_guide_ = false;
  gpu::Shader *guide_shader_ = nullptr, *hit_shader_ = nullptr;
  draw::PassSimple guides_{"DLSS.Guides"}, hit_pass_{"DLSS.Hit"};
  int2 input_size_ = int2(0), output_size_ = int2(0);
  int2 visible_size_ = int2(0), crop_ = int2(0);
  uint64_t frame_ = 0;
  draw::Texture color_{"DLSS.Color"};
  draw::Texture color_before_transparency_{"DLSS.BeforeTransparency"};
  draw::PassSimple before_transparency_pass_{"DLSS.BeforeTransparency"};
  gpu::Shader *before_transparency_shader_ = nullptr;
  bool has_before_transparency_ = false;
  draw::Texture depth_{"DLSS.Depth"};
  draw::Texture motion_{"DLSS.Motion"};
  draw::Texture output_{"DLSS.Output"};
  gpu::Shader *prepare_shader_ = nullptr;
  gpu::Shader *blend_shader_ = nullptr;
  gpu::Shader *integrate_shader_ = nullptr;
  draw::PassSimple integrate_pass_{"DLSS.StaticIntegration"};
  draw::Texture integrated_color_{"DLSS.IntegratedColor"};
  uint64_t layered_update_ = 0;
  uint64_t integrated_samples_ = 0, integrated_update_ = 0;
  float integrated_time_ = 0.0f;
  draw::PassSimple blend_pass_{"DLSS.Blended"};
  draw::Texture blend_radiance_{"DLSS.BlendRadiance"},
      blend_transmittance_{"DLSS.BlendTransmittance"};
  draw::Texture composite_{"DLSS.Composite"};
  uint64_t blend_samples_ = 0;
  bool has_composite_ = false;
  draw::PassSimple prepare_{"DLSS.Prepare"};
  float2 last_jitter_ = float2(0.0f);

 public:
  DLSSModule(Instance &inst) : inst_(inst) {}
  ~DLSSModule();
  void init(int2 extent, bool full_viewport);
  void begin_sync()
  {
    reasons_ = settings_reasons_;
  }
  void end_sync();
  void unsupported_geometry()
  {
    unsupported_geometry_ = true;
  }
  void reject(const char *reason);
  void report(const std::string &message);
  void reason(const char *code, const std::string &detail);
  void material_check(const Object *object, const blender::Material *material, GPUMaterial *gpu);
  void dump_guides();
  bool consume_restart()
  {
    bool value = restart_;
    restart_ = false;
    return value;
  }
  bool failed() const
  {
    return enabled_ && (!context_ || context_->failed());
  }
  bool ray_reconstruction() const
  {
    return enabled_ && rr_;
  }
  bool super_resolution() const
  {
    return enabled_ && !rr_;
  }
  bool single_evaluation() const;
  void render_reference(draw::View &view);
  void capture_effect_base(gpu::Texture *color);
  ReconstructionState state() const
  {
    return state_;
  }
  void begin_trace();
  void capture_before_transparency(
      draw::View &view, gpu::Texture *r, gpu::Texture *g, gpu::Texture *b, bool colored);
  bool restart_pending() const
  {
    return restart_;
  }
  void capture_hit(draw::View &view, gpu::Texture *hit, gpu::Texture *path, int closure);
  void prepare_guides(draw::View &view);
  bool subsurface_guide_needed() const;
  void capture_subsurface(draw::View &view,
                          gpu::Texture *direct,
                          gpu::Texture *indirect,
                          bool before);
  gpu::Texture *final_texture()
  {
    return has_result_ ? render_adapter_.texture() : nullptr;
  }
  bool enabled() const
  {
    return enabled_;
  }
  int2 input_size() const
  {
    return input_size_;
  }
  float2 jitter() const;
  void render(draw::View &view, gpu::Texture *color);
  void finish();
  void blend_layer(draw::View &view,
                   gpu::Texture *r,
                   gpu::Texture *g,
                   gpu::Texture *b,
                   bool colored,
                   bool reset);
  bool output_blended() const;
  int2 output_size() const
  {
    return output_size_;
  }
  void display();
};
}  // namespace blender::eevee
