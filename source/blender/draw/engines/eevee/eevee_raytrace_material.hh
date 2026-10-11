/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */
#pragma once
#include "BLI_set.hh"
#include "DRW_gpu_wrapper.hh"
#include "eevee_raytrace_shared.hh"

namespace blender {
struct Material;
struct Image;
struct PartialUpdateUser;
struct bNode;
struct bNodeSocket;
}  // namespace blender

namespace blender::eevee {
/** Deliberately bounded PBR subset for secondary surfaces. Unsupported graphs are explicit misses.
 */
class RayTracingMaterials {
  Map<const blender::Material *, uint32_t> indices_;
  struct TextureRecord {
    uint4 info = uint4(0);
    blender::Image *image = nullptr;
    uint32_t session_uid = 0;
    PartialUpdateUser *user = nullptr;
  };
  Map<const bNode *, TextureRecord> textures_;
  Set<const blender::Material *> visited_;
  Vector<HardwareRTMaterial> materials_;
  Vector<float4> pixels_;
  bool dirty_ = true;
  draw::StorageArrayBuffer<HardwareRTMaterial, 16> material_buf_ = {"RTMaterials"};
  draw::StorageArrayBuffer<float4, 16> pixel_buf_ = {"RTTexturePixels"};
  bool parameter(const bNodeSocket *socket, HardwareRTParameter &value);

 public:
  ~RayTracingMaterials();
  bool changed = false;
  int unsupported = 0;
  void begin_sync();
  void invalidate();
  uint32_t sync(const blender::Material *material);
  bool upload();
  template<typename Pass> void bind_resources(Pass &pass)
  {
    pass.bind_ssbo("rt_materials", &material_buf_);
    pass.bind_ssbo("rt_pixels", &pixel_buf_);
  }
};
}  // namespace blender::eevee
