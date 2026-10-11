/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "DRW_gpu_wrapper.hh"
#include "GPU_dlss.hh"
#include "draw_pass.hh"

namespace blender {
struct Object;
}

namespace blender::eevee {
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
  bool update_history(const Instance &inst);
  bool display(
      Instance &inst, gpu::Texture *color, gpu::Texture *depth, int2 extent, float2 jitter);
};

/** Window-independent conversion to Film's bottom-left scene-linear Combined texture. */
class DLSSRenderAdapter {
  draw::Texture output_{"DLSS.Final"};
  gpu::Shader *shader_ = nullptr;
  draw::PassSimple pass_{"DLSS.Output"};

 public:
  ~DLSSRenderAdapter();
  bool resolve(Instance &inst, gpu::Texture *color, int2 extent);
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
  bool attempted_ = false;
  bool enabled_ = false;
  bool reset_ = true;
  bool has_result_ = false;
  int mode_ = 0;
  bool rr_ = false, requested_rr_ = false;
  bool force_native_ = false, restart_ = false;
  bool unsupported_geometry_ = false;
  uint64_t blocked_update_ = 0, hardware_revision_ = 0;
  draw::Texture diffuse_{"DLSS.DiffuseAlbedo"}, specular_{"DLSS.SpecularAlbedo"};
  draw::Texture normals_{"DLSS.NormalRoughness"}, hit_{"DLSS.SpecularHitDistance"};
  DLSSViewportAdapter viewport_adapter_;
  DLSSRenderAdapter render_adapter_;
  gpu::Shader *guide_shader_ = nullptr, *hit_shader_ = nullptr;
  draw::PassSimple guides_{"DLSS.Guides"}, hit_pass_{"DLSS.Hit"};
  int2 input_size_ = int2(0), output_size_ = int2(0);
  uint64_t frame_ = 0;
  draw::Texture color_{"DLSS.Color"};
  draw::Texture depth_{"DLSS.Depth"};
  draw::Texture motion_{"DLSS.Motion"};
  draw::Texture output_{"DLSS.Output"};
  gpu::Shader *prepare_shader_ = nullptr;
  draw::PassSimple prepare_{"DLSS.Prepare"};
  float2 last_jitter_ = float2(0.0f);

 public:
  DLSSModule(Instance &inst) : inst_(inst) {}
  ~DLSSModule();
  void init(int2 extent, bool full_viewport);
  void end_sync();
  void unsupported_geometry() { unsupported_geometry_ = true; }
  void reject(const char *reason);
  void report(const std::string &message);
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
  void begin_trace();
  bool restart_pending() const
  {
    return restart_;
  }
  void capture_hit(draw::View &view, gpu::Texture *hit, int closure);
  void prepare_guides(draw::View &view);
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
  void display();
};
}  // namespace blender::eevee
