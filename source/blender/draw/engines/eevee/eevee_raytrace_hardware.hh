/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once
#include "DRW_gpu_wrapper.hh"
#include "GPU_acceleration_structure.hh"
#include "eevee_raytrace_material.hh"
#include "eevee_raytrace_shared.hh"
#include "eevee_sync.hh"

namespace blender::draw {
class View;
}

namespace blender::eevee {
class Instance;
struct MaterialArray;

class RayTracingScene {
  struct GeometryKey {
    const Mesh *mesh;
    Vector<uint8_t> supported_materials;
    Vector<uint32_t> material_layout;
    uint64_t hash() const
    {
      uint64_t value = get_default_hash(mesh);
      for (uint8_t supported : supported_materials) {
        value = value * 33 + supported;
      }
      for (uint32_t material : material_layout) {
        value = value * 33 + material;
      }
      return value;
    }
    bool operator==(const GeometryKey &other) const
    {
      return mesh == other.mesh && supported_materials == other.supported_materials &&
             material_layout == other.material_layout;
    }
  };
  struct Geometry {
    gpu::AccelerationStructurePtr blas;
    Vector<float3> positions;
    Vector<uint3> triangles;
    Vector<float4> normals;
    Vector<HardwareRTTriangle> attributes;
    Vector<HardwareRTUVAttribute> uv_attributes;
    uint64_t last_sync = 0;
    uint32_t generation = 1;
    uint32_t normal_offset = 0;
    int invalid_tangents = 0;
  };
  struct InstanceRecord {
    Geometry *geometry = nullptr;
    Vector<uint32_t> materials;
    uint4 linking = uint4(0);
    float4x4 transform;
    uint32_t id = 0;
    uint64_t generation = 0;
    uint64_t last_sync = 0;
  };
  Instance &inst_;
  Map<GeometryKey, std::unique_ptr<Geometry>> geometries_;
  Map<draw::ObjectKey, InstanceRecord> instances_;
  uint64_t sync_id_ = 0;
  uint32_t next_id_ = 1;
  bool dirty_ = true;
  bool resync_geometry_ = true;
  bool failed_ = false;
  gpu::AccelerationStructurePtr tlas_;
  draw::StorageArrayBuffer<HardwareRTInstance, 16> instance_data_ = {"HardwareRTInstances"};
  draw::StorageArrayBuffer<float4, 16> triangle_normals_ = {"HardwareRTNormals"};

  draw::StorageArrayBuffer<HardwareRTTriangle, 16> triangle_data_ = {"RTTriangleData"};
  draw::StorageArrayBuffer<uint32_t, 16> material_map_ = {"RTMaterialMap"};
  draw::StorageArrayBuffer<HardwareRTUVAttribute, 16> uv_data_ = {"RTUVAttributes"};

 public:
  RayTracingMaterials materials;
  gpu::AccelerationStructure *tlas_binding = nullptr;
  uint64_t revision = 0;
  int skipped_materials = 0;
  int blas_build_count = 0;
  int tlas_build_count = 0;
  size_t memory_bytes = 0;
  double cpu_sync_ms = 0;
  RayTracingScene(Instance &inst) : inst_(inst) {}
  void begin_sync(bool invalidate_materials);
  void sync_mesh(const ObjectHandle &object, const MaterialArray &materials);
  void end_sync();
  void irradiance_probes(Vector<HardwareIrradianceProbe> &probes);
  void invalidate_geometry()
  {
    resync_geometry_ = true;
  }
  bool ready() const
  {
    return !failed_ && tlas_binding && tlas_binding->is_valid();
  }
  template<typename Pass> void bind_resources(Pass &pass)
  {
    pass.bind_acceleration_structure("rt_scene", &tlas_binding);
    pass.bind_ssbo("rt_instances", &instance_data_);
    pass.bind_ssbo("rt_normals", &triangle_normals_);
    pass.bind_ssbo("rt_triangles", &triangle_data_);
    pass.bind_ssbo("rt_material_map", &material_map_);
    pass.bind_ssbo("rt_uv_attributes", &uv_data_);
    materials.bind_resources(pass);
  }
};

/** A bounded, direct-mapped sparse surface cache. Hash collisions are misses, never false hits. */
class SurfaceRadianceCache {
 public:
  static constexpr uint32_t entry_count = 1u << 20;
  gpu::StorageBuf *entries = nullptr;
  gpu::StorageBuf *winners = nullptr;
  draw::Texture direct_radiance = {"HardwareRTDirectRadiance"};
  draw::Texture surface_keys = {"HardwareRTSurfaceKeys"};
  ~SurfaceRadianceCache();
  bool ensure(int2 extent);
  void invalidate();
  template<typename Pass> void bind_resources(Pass &pass)
  {
    pass.bind_ssbo("rt_cache", &entries);
  }
};

/** World-space irradiance sampled by independent hemisphere rays, never screen radiance. */
class WorldIrradianceCache {
 public:
  static constexpr uint32_t entry_count = 1u << 15;
  draw::StorageArrayBuffer<HardwareIrradianceProbe, entry_count> probes = {"RTIrradianceProbes"};
  gpu::StorageBuf *history = nullptr;
  gpu::StorageBuf *current = nullptr;
  uint32_t frame = 0;
  ~WorldIrradianceCache();
  bool ensure();
  void reset(RayTracingScene &scene);
  template<typename Pass> void bind_resources(Pass &pass)
  {
    pass.bind_ssbo("rt_irradiance_probes", &probes);
    pass.bind_ssbo("rt_irradiance", &current);
  }
};

class HardwareRayTracing {
  Instance &inst_;
  gpu::Shader *capture_shader_ = nullptr;
  gpu::Shader *select_shader_ = nullptr;
  gpu::Shader *resolve_shader_ = nullptr;
  gpu::Shader *trace_shader_ = nullptr;
  gpu::Shader *debug_shader_ = nullptr;
  gpu::Shader *irradiance_shader_ = nullptr;
  bool requested_ = false;
  bool enabled_ = false;
  uint64_t scene_revision_ = 0;
  uint64_t depsgraph_update_ = 0;
  bool invalidated_ = true;
  bool had_pending_materials_ = false;
  int light_count_ = 0;
  int previous_light_count_ = -1;
  uint64_t history_revision_ = 0;

 public:
  RayTracingScene scene;
  SurfaceRadianceCache cache;
  WorldIrradianceCache irradiance;
  /* Referenced by the screen trace push constant and updated per deferred layer at render time.
   * Use the 32-bit reference overload: a bool pointer would bind a constant true instead. */
  bool32_t active = false;
  HardwareRayTracing(Instance &inst) : inst_(inst), scene(inst) {}
  ~HardwareRayTracing();
  void init();
  void begin_sync();
  void sync_mesh(const ObjectHandle &object, const MaterialArray &materials);
  void end_sync();
  bool enabled() const
  {
    return enabled_;
  }
  void invalidate()
  {
    invalidated_ = true;
  }
  void sync_light(bool changed)
  {
    if (enabled_) {
      light_count_++;
      invalidated_ |= changed;
    }
  }
  uint64_t history_revision() const
  {
    return history_revision_;
  }
  bool prepare(draw::View &view, gpu::Texture *emission, gpu::Texture *direct[3]);
  void debug_draw(draw::View &view, gpu::Texture *combined);
  gpu::Shader *trace_shader() const
  {
    return trace_shader_;
  }
  template<typename Pass> void bind_resources(Pass &pass)
  {
    scene.bind_resources(pass);
    cache.bind_resources(pass);
    irradiance.bind_resources(pass);
    pass.bind_texture("rt_direct_tx", &cache.direct_radiance);
  }
};
}  // namespace blender::eevee
