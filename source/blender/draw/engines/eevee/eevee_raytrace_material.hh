/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */
#pragma once
#include "BLI_set.hh"
#include "DNA_image_types.h"
#include "DRW_gpu_wrapper.hh"
#include "eevee_raytrace_shared.hh"
#include <array>

namespace blender {
struct Material;
struct Image;
struct PartialUpdateUser;
struct bNode;
struct bNodeSocket;
}  // namespace blender

namespace blender::eevee {
/** Fixed texture descriptors and bounded closures for secondary surfaces. Unsupported graphs
 * report local screen/probe fallback; they never become default materials or disable frame RR. */
class RayTracingMaterials {
  Map<const blender::Material *, uint32_t> indices_;
  struct TextureRecord {
    uint4 info = uint4(0);
    blender::Image *image = nullptr;
    uint32_t session_uid = 0;
    PartialUpdateUser *user = nullptr;
    ImageUser image_user = {};
    bool dirty = true;
  };
  Map<const blender::Image *, TextureRecord> textures_;
  struct ImageGroup {
    int2 size = int2(0);
    bool floating = false;
    Vector<const blender::Image *> images;
    draw::Texture texture{"RTMaterialImages"};
  };
  std::array<ImageGroup, 8> image_groups_;
  std::array<gpu::Texture *, 8> bound_images_{};
  draw::Texture dummy_image_{"RTMaterialImageDummy"};
  Set<const blender::Material *> visited_;
  Vector<HardwareRTMaterial> materials_;
  Vector<HardwareRTMaterial> closures_, previous_closures_;
  draw::StorageArrayBuffer<HardwareRTMaterial, 16> closure_buf_ = {"RTClosures"};
  Vector<const bNode *> parameter_context_;
  std::string parameter_error_;
  Vector<std::string> parameter_uvs_;
  bool dirty_ = true;
  draw::StorageArrayBuffer<HardwareRTMaterial, 16> material_buf_ = {"RTMaterials"};
  bool parameter(const bNodeSocket *socket, HardwareRTParameter &value, std::string *uv = nullptr);
  bool parameter_leaf(const bNodeSocket *socket,
                      HardwareRTParameter &value,
                      std::string *uv = nullptr,
                      bool ignore_coordinates = false);

 public:
  struct NormalInfo {
    bool enabled = false;
    bool tangent = false;
    std::string tangent_uv;
    Vector<std::string> tangent_uvs;
  };
  Vector<NormalInfo> normal_info;
  Vector<Vector<std::string>> uv_names;
  Vector<std::string> diagnostics;
  ~RayTracingMaterials();
  bool changed = false;
  int unsupported = 0;
  bool lightweight(uint32_t index) const
  {
    return materials_[index].properties.z > 0.0f;
  }
  void begin_sync();
  void invalidate();
  uint32_t sync(const blender::Material *material);
  bool upload();
  template<typename Pass> void bind_resources(Pass &pass)
  {
    pass.bind_ssbo("rt_materials", &material_buf_);
    pass.bind_ssbo("rt_closures", &closure_buf_);
    const char *names[] = {"rt_images0",
                           "rt_images1",
                           "rt_images2",
                           "rt_images3",
                           "rt_images4",
                           "rt_images5",
                           "rt_images6",
                           "rt_images7"};
    for (int i = 0; i < image_groups_.size(); i++) {
      pass.bind_texture(names[i], &bound_images_[i]);
    }
  }
};
}  // namespace blender::eevee
