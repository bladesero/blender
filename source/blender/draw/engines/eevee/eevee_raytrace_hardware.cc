/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <chrono>
#include <cmath>

#include "BKE_attribute.hh"
#include "BKE_material.hh"
#include "BKE_mesh.hh"
#include "BKE_mesh_tangent.hh"
#include "BKE_object.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_vector.hh"
#include "DEG_depsgraph_query.hh"
#include "DNA_material_types.h"
#include "GPU_capabilities.hh"
#include "GPU_material.hh"
#include "GPU_shader.hh"
#include "GPU_storage_buffer.hh"
#include "eevee_instance.hh"
#include "eevee_raytrace_hardware.hh"

namespace blender::eevee {

/* Recoverable RT fallback is a viewport notice, not a final-render failure. */
template<typename... Args>
static void hardware_notice(Instance &inst, const char *message, Args &&...args)
{
  if (inst.is_viewport()) {
    inst.info_append(message, args...);
  }
  else {
    const std::string text = fmt::format(fmt::runtime(message), args...);
    fprintf(stderr, "%s\n", text.c_str());
  }
}

void RayTracingScene::begin_sync(bool invalidate_materials)
{
  sync_id_++;
  materials.begin_sync();
  if (invalidate_materials) {
    materials.invalidate();
    resync_geometry_ = true;
    dirty_ = true;
  }
  failed_ = false;
  skipped_materials = 0;
  blas_build_count = tlas_build_count = 0;
  cpu_sync_ms = 0;
}

void RayTracingScene::sync_mesh(const ObjectHandle &object, const MaterialArray &materials)
{
  const Mesh *mesh = BKE_object_get_evaluated_mesh(object.object);
  if (!mesh) {
    return;
  }
  GeometryKey key{mesh, {}, {}};
  Vector<uint8_t> &supported = key.supported_materials;
  Vector<uint32_t> material_ids;
  for (const Material &material : materials.materials) {
    GPUMaterial *gpu = material.shading.gpumat;
    /* A GPUMaterial may refer to an older evaluated material while its compiled pass is reused. */
    const blender::Material *mat = inst_.materials.material_override;
    if (!mat) {
      mat = BKE_object_material_get_eval(object.object, material_ids.size() + 1);
    }
    if (!mat) {
      mat = BKE_material_default_surface();
    }
    const bool include = gpu && material.has_surface &&
                         !GPU_material_flag_get(gpu, GPU_MATFLAG_HOLDOUT);
    const bool opaque = include && !material.is_alpha_blend_transparent &&
                        !GPU_material_flag_get(
                            gpu, eGPUMaterialFlag(GPU_MATFLAG_TRANSPARENT | GPU_MATFLAG_REFRACT));
    const int64_t diagnostic_start = this->materials.diagnostics.size();
    material_ids.append(this->materials.sync(mat));
    for (int64_t i = diagnostic_start; i < this->materials.diagnostics.size(); i++) {
      this->materials.diagnostics[i] = std::string(object.object->id.name + 2) + " / " +
                                       this->materials.diagnostics[i];
    }
    const auto normal = this->materials.normal_info[material_ids.last()];
    const auto required_uvs = this->materials.uv_names[material_ids.last()];
    bool attributes_valid = true;
    auto require_uv = [&](bool required, const std::string &name) {
      const std::string resolved = name.empty() ? std::string(mesh->default_uv_map_name()) : name;
      if (required && (resolved.empty() ||
                       !mesh->attributes().lookup<float2>(resolved, bke::AttrDomain::Corner)))
      {
        attributes_valid = false;
        this->materials.diagnostics.append(
            std::string(object.object->id.name + 2) + " / " + std::string(mat->id.name + 2) +
            ": missing UV map '" + (resolved.empty() ? "<active render UV>" : resolved) + "'");
      }
    };
    require_uv(normal.enabled && normal.tangent, normal.tangent_uv);
    for (const std::string &name : required_uvs) {
      require_uv(true, name);
    }
    supported.append(include ? (attributes_valid ? (opaque ? 1 : 2) : 3) : 0);
    skipped_materials += !include;
  }
  resync_geometry_ |= this->materials.changed;
  key.material_layout = material_ids;
  Geometry &geometry = *geometries_.lookup_or_add_cb(
      key, []() { return std::make_unique<Geometry>(); });
  if (geometry.last_sync != sync_id_) {
    geometry.last_sync = sync_id_;
    if (resync_geometry_ || !geometry.blas || !geometry.blas->is_valid() ||
        (object.recalc & (ID_RECALC_GEOMETRY | ID_RECALC_SHADING)))
    {
      const auto material_indices = mesh->attributes().lookup_or_default<int>(
          "material_index", bke::AttrDomain::Face, 0);
      const Span<int> corners = mesh->corner_verts();
      const Span<int3> corner_triangles = mesh->corner_tris();
      const Span<int> faces = mesh->corner_tri_faces();
      Vector<uint3> triangles;
      Vector<HardwareRTTriangle> attributes;
      Vector<HardwareRTUVAttribute> uv_attributes;
      const StringRefNull uv_name = mesh->default_uv_map_name();
      const auto uvs = mesh->attributes().lookup_or_default<float2>(
          uv_name, bke::AttrDomain::Corner, float2(0.0f));
      const Span<float3> shading_normals = mesh->corner_normals();
      geometry.invalid_tangents = 0;
      Map<std::string, Array<float4>> tangent_maps;
      auto uv_values = [&](const std::string &name) {
        return mesh->attributes().lookup_or_default<float2>(name.empty() ? StringRef(uv_name) :
                                                                           StringRef(name),
                                                            bke::AttrDomain::Corner,
                                                            float2(0));
      };
      auto ensure_tangent = [&](const std::string &name) {
        if (tangent_maps.contains(name)) {
          return;
        }
        const VArraySpan<float2> coords(uv_values(name).varray);
        const VArraySpan<bool> sharp(
            mesh->attributes()
                .lookup_or_default<bool>("sharp_face", bke::AttrDomain::Face, false)
                .varray);
        const Span<float2> uv_span = coords;
        auto calculated = bke::mesh::calc_uv_tangents(mesh->vert_positions(),
                                                      mesh->faces(),
                                                      corners,
                                                      corner_triangles,
                                                      faces,
                                                      sharp,
                                                      mesh->vert_normals(),
                                                      mesh->face_normals(),
                                                      shading_normals,
                                                      Span<Span<float2>>(&uv_span, 1));
        tangent_maps.add(name, std::move(calculated[0]));
      };
      for (uint32_t id : material_ids) {
        const auto &normal = this->materials.normal_info[id];
        if (normal.enabled && normal.tangent) {
          ensure_tangent(normal.tangent_uv);
        }
        for (const std::string &name : this->materials.uv_names[id]) {
          ensure_tangent(name);
        }
      }
      for (const int i : corner_triangles.index_range()) {
        const int material = material_indices.varray[faces[i]];
        if (material < 0 || material >= supported.size() || !supported[material]) {
          continue;
        }
        const int3 tri = corner_triangles[i];
        triangles.append(uint3(corners[tri.x], corners[tri.y], corners[tri.z]));
        HardwareRTTriangle attribute{};
        attribute.corner0 = float4(shading_normals[tri.x], uvs.varray[tri.x].x);
        attribute.corner1 = float4(shading_normals[tri.y], uvs.varray[tri.y].x);
        attribute.corner2 = float4(shading_normals[tri.z], uvs.varray[tri.z].x);
        attribute.uv_y = float4(
            uvs.varray[tri.x].y, uvs.varray[tri.y].y, uvs.varray[tri.z].y, 0.0f);
        const auto &names = this->materials.uv_names[material_ids[material]];
        attribute.attributes = uint4(uv_attributes.size(), names.size(), 0, 0);
        for (const std::string &name : names) {
          const auto values = uv_values(name);
          HardwareRTUVAttribute uv{};
          uv.uv01 = float4(values.varray[tri.x], values.varray[tri.y]);
          uv.uv2 = float4(values.varray[tri.z], 0, 0);
          const auto &tangents = tangent_maps.lookup(name);
          uv.tangent0 = tangents[tri.x];
          uv.tangent1 = tangents[tri.y];
          uv.tangent2 = tangents[tri.z];
          uv_attributes.append(uv);
        }
        const auto &normal = this->materials.normal_info[material_ids[material]];
        attribute.material = uint4(material, supported[material] == 3 ? 1 : 0, 0, 0);
        for (const std::string &name : normal.tangent_uvs) {
          const auto &t = tangent_maps.lookup(name);
          const auto coords = uv_values(name);
          const float2 du = coords.varray[tri.y] - coords.varray[tri.x];
          const float2 dv = coords.varray[tri.z] - coords.varray[tri.x];
          const float determinant = du.x * dv.y - du.y * dv.x;
          bool valid = std::isfinite(determinant) && std::abs(determinant) >= 1e-12f;
          for (int corner : {tri.x, tri.y, tri.z}) {
            const float length = math::length_squared(t[corner].xyz());
            valid &= std::isfinite(length) && length > 1e-12f;
          }
          geometry.invalid_tangents += !valid;
          attribute.material.y |= !valid;
        }
        attributes.append(attribute);
      }
      const bool uv_changed = uv_attributes.size() != geometry.uv_attributes.size() ||
                              (!uv_attributes.is_empty() &&
                               memcmp(uv_attributes.data(),
                                      geometry.uv_attributes.data(),
                                      uv_attributes.size() * sizeof(HardwareRTUVAttribute)) != 0);
      if (uv_changed || attributes.size() != geometry.attributes.size() ||
          (!attributes.is_empty() && memcmp(attributes.data(),
                                            geometry.attributes.data(),
                                            attributes.size() * sizeof(HardwareRTTriangle)) != 0))
      {
        geometry.attributes = std::move(attributes);
        geometry.uv_attributes = std::move(uv_attributes);
        dirty_ = true;
        geometry.generation++;
      }
      const Span<float3> positions = mesh->vert_positions();
      if (!geometry.blas || !geometry.blas->is_valid() ||
          geometry.positions.as_span() != positions || geometry.triangles != triangles)
      {
        geometry.generation++;
        dirty_ = true;
        const bool same_topology = geometry.triangles == triangles;
        geometry.positions = positions;
        geometry.triangles = std::move(triangles);
        geometry.normals.clear();
        for (const uint3 &tri : geometry.triangles) {
          geometry.normals.append(
              float4(math::normalize(math::cross(positions[tri.y] - positions[tri.x],
                                                 positions[tri.z] - positions[tri.x])),
                     0.0f));
        }
        if (!same_topology || !geometry.blas) {
          geometry.blas = gpu::acceleration_structure_create();
        }
        if (!geometry.triangles.is_empty()) {
          if (!geometry.blas || !geometry.blas->build_mesh(
                                    positions,
                                    geometry.triangles,
                                    true,
                                    std::all_of(supported.begin(), supported.end(), [](uint8_t v) {
                                      return v == 1;
                                    })))
          {
            geometry.blas.reset();
            failed_ = true;
            return;
          }
          blas_build_count++;
        }
      }
    }
  }
  skipped_materials += geometry.invalid_tangents != 0;
  if (geometry.invalid_tangents) {
    this->materials.diagnostics.append(std::string(object.object->id.name + 2) +
                                       ": degenerate normal-map tangent");
  }
  if (geometry.triangles.is_empty() || !geometry.blas || !geometry.blas->is_valid()) {
    return;
  }
  for (int i = 0; i < object.instances_count(); i++) {
    const draw::ObjectKey key(object, i);
    const float4x4 transform = object.object_to_world(i);
    const float determinant = math::determinant(float3x3(transform));
    if (!std::isfinite(determinant) || determinant == 0.0f) {
      /* Zero-scale instances have no traceable surface, and are illegal TLAS inputs. */
      continue;
    }
    InstanceRecord &record = instances_.lookup_or_add_default(key);
    const uint64_t generation = object.instance_generation();
    if (record.generation != generation) {
      record.id = 0;
      record.generation = generation;
    }
    if (record.id == 0) {
      /* IDs are never recycled while cache entries exist. */
      if (next_id_ == 0) {
        failed_ = true;
        return;
      }
      record.id = next_id_++;
      dirty_ = true;
    }
    if (record.geometry != &geometry || record.transform != transform) {
      dirty_ = true;
    }
    const auto *linking = object.object->light_linking;
    const uint4 sets(linking ? linking->runtime.receiver_light_set : 0,
                     linking ? linking->runtime.blocker_shadow_set : 0,
                     (object.object->visibility_flag & OB_HIDE_SHADOW) ? 1u : 0u,
                     0);
    dirty_ |= record.materials != material_ids || record.linking != sets;
    record.materials = material_ids;
    record.linking = sets;
    record.geometry = &geometry;
    record.transform = transform;
    record.last_sync = sync_id_;
  }
}

void RayTracingScene::end_sync()
{
  dirty_ |= instances_.remove_if(
                [&](const auto &item) { return item.value.last_sync != sync_id_; }) != 0;
  geometries_.remove_if([&](const auto &item) { return item.value->last_sync != sync_id_; });
  if (!failed_) {
    resync_geometry_ = false;
  }
  failed_ |= !materials.upload();
  if (failed_ || !dirty_) {
    revision += materials.changed && !failed_;
    return;
  }
  Vector<InstanceRecord *> sorted;
  for (auto &record : instances_.values()) {
    sorted.append(&record);
  }
  std::sort(
      sorted.begin(), sorted.end(), [](const auto *a, const auto *b) { return a->id < b->id; });
  if (sorted.size() > 0x00ffffff) {
    failed_ = true;
    return;
  }
  Vector<float4> normals;
  Vector<HardwareRTTriangle> attributes;
  Vector<HardwareRTUVAttribute> uv_attributes;
  Vector<uint32_t> material_map;
  memory_bytes = 0;
  for (const auto &geometry : geometries_.values()) {
    geometry->normal_offset = normals.size();
    normals.extend(geometry->normals);
    for (HardwareRTTriangle triangle : geometry->attributes) {
      triangle.attributes.x += uv_attributes.size();
      attributes.append(triangle);
    }
    uv_attributes.extend(geometry->uv_attributes);
    if (geometry->blas) {
      memory_bytes += geometry->blas->memory_usage();
    }
  }
  /* Fail safely before growing auxiliary scene storage without bound. */
  const size_t max_storage = GPU_max_storage_buffer_size();
  if (normals.size() > (1u << 26) || memory_bytes > (size_t(2) << 30) ||
      attributes.size() * sizeof(HardwareRTTriangle) > max_storage ||
      uv_attributes.size() * sizeof(HardwareRTUVAttribute) > max_storage ||
      sorted.size() * sizeof(HardwareRTInstance) > max_storage)
  {
    failed_ = true;
    return;
  }
  instance_data_.resize(std::max<int64_t>(1, sorted.size()));
  triangle_normals_.resize(std::max<int64_t>(1, normals.size()));
  for (int64_t i = 0; i < normals.size(); i++) {
    triangle_normals_[i] = normals[i];
  }
  triangle_data_.resize(std::max<int64_t>(1, attributes.size()));
  uv_data_.resize(std::max<int64_t>(1, uv_attributes.size()));
  for (int i : uv_attributes.index_range()) {
    uv_data_[i] = uv_attributes[i];
  }
  for (int i : attributes.index_range()) {
    triangle_data_[i] = attributes[i];
  }
  Vector<gpu::RayTracingInstance> instances;
  for (int64_t i = 0; i < sorted.size(); i++) {
    const auto &record = *sorted[i];
    instance_data_[i].world_to_object = math::invert(record.transform);
    instance_data_[i].identity = uint4(record.id,
                                       record.geometry->normal_offset,
                                       record.geometry->generation,
                                       material_map.size());
    instance_data_[i].linking = record.linking;
    material_map.extend(record.materials);
    instances.append({record.geometry->blas, record.transform, uint32_t(i), 0xff});
  }
  if (material_map.size() * sizeof(uint32_t) > max_storage) {
    failed_ = true;
    return;
  }
  material_map_.resize(std::max<int64_t>(1, material_map.size()));
  for (int i : material_map.index_range()) {
    material_map_[i] = material_map[i];
  }
  material_map_.push_update();
  triangle_data_.push_update();
  uv_data_.push_update();
  instance_data_.push_update();
  triangle_normals_.push_update();
  if (!GPU_storagebuf_ensure_allocated(instance_data_) ||
      !GPU_storagebuf_ensure_allocated(triangle_normals_) ||
      !GPU_storagebuf_ensure_allocated(triangle_data_) ||
      !GPU_storagebuf_ensure_allocated(uv_data_) ||
      !GPU_storagebuf_ensure_allocated(material_map_))
  {
    failed_ = true;
    return;
  }
  if (!tlas_) {
    tlas_ = gpu::acceleration_structure_create();
  }
  if (!tlas_ || !tlas_->build_instances(instances)) {
    tlas_binding = nullptr;
    tlas_.reset();
    failed_ = true;
    return;
  }
  tlas_binding = tlas_.get();
  tlas_build_count++;
  memory_bytes += tlas_->memory_usage();
  memory_bytes += std::max<int64_t>(1, sorted.size()) * sizeof(HardwareRTInstance) +
                  std::max<int64_t>(1, normals.size()) * sizeof(float4) +
                  std::max<int64_t>(1, attributes.size()) * sizeof(HardwareRTTriangle) +
                  std::max<int64_t>(1, material_map.size()) * sizeof(uint32_t);
  dirty_ = false;
  revision++;
}

void RayTracingScene::irradiance_probes(Vector<HardwareIrradianceProbe> &probes)
{
  probes.resize(WorldIrradianceCache::entry_count);
  for (auto &probe : probes) {
    probe = {};
  }
  Vector<InstanceRecord *> sorted;
  for (auto &record : instances_.values()) {
    sorted.append(&record);
  }
  std::sort(sorted.begin(), sorted.end(), [](auto *a, auto *b) { return a->id < b->id; });
  double area = 0.0;
  for (const auto *record : sorted) {
    const Geometry &g = *record->geometry;
    for (const uint3 &tri : g.triangles) {
      const float3 a = math::transform_point(record->transform, g.positions[tri.x]);
      const float3 b = math::transform_point(record->transform, g.positions[tri.y]);
      const float3 c = math::transform_point(record->transform, g.positions[tri.z]);
      area += 0.5 * math::length(math::cross(b - a, c - a));
    }
  }
  /* About 8000 surface cells, two sides. Independent of the camera and raster visibility. */
  const float spacing = std::max(0.05f, float(std::sqrt(area / 8000.0)));
  auto insert = [&](float3 P, float3 N) {
    const int3 cell(math::floor(P / spacing));
    const int octant = int(N.x >= 0.0f) + 2 * int(N.y >= 0.0f) + 4 * int(N.z >= 0.0f);
    const int4 key(cell.x, cell.y, cell.z, octant);
    uint32_t hash = uint32_t(cell.x) * 0x9e3779b9u ^ uint32_t(cell.y) * 0x85ebca6bu ^
                    uint32_t(cell.z) * 0xc2b2ae35u ^ uint32_t(octant) * 0x27d4eb2du;
    hash ^= hash >> 16u;
    HardwareIrradianceProbe &probe = probes[hash & (WorldIrradianceCache::entry_count - 1)];
    if (probe.normal.w != 0.0f) {
      return; /* Budget/hash collision: caller falls back, never reuse an unrelated cell. */
    }
    probe.position = float4(P, spacing);
    probe.normal = float4(N, 1.0f);
    probe.key = key;
  };
  uint32_t candidates = 0;
  for (const auto *record : sorted) {
    const Geometry &g = *record->geometry;
    for (int t : g.triangles.index_range()) {
      const uint3 tri = g.triangles[t];
      const float3 a = math::transform_point(record->transform, g.positions[tri.x]);
      const float3 b = math::transform_point(record->transform, g.positions[tri.y]);
      const float3 c = math::transform_point(record->transform, g.positions[tri.z]);
      const float3 N = math::normalize(math::transpose(math::invert(float3x3(record->transform))) *
                                       g.normals[t].xyz());
      const float edge = std::max({math::length(b - a), math::length(c - a), math::length(c - b)});
      const int steps = std::clamp(int(std::ceil(edge / spacing)), 1, 64);
      for (int y = 0; y < steps; y++) {
        for (int x = 0; x < steps - y; x++) {
          if (++candidates > (1u << 20)) {
            probes[0].position.w = spacing;
            return; /* Bounded CPU allocation work in very dense scenes. */
          }
          const float u = (x + 1.0f / 3.0f) / steps;
          const float v = (y + 1.0f / 3.0f) / steps;
          const float3 P = a + u * (b - a) + v * (c - a);
          insert(P, N);
          insert(P, -N);
        }
      }
    }
  }
  /* Slot zero also carries grid scale, even when no probe occupies it. */
  probes[0].position.w = spacing;
}

WorldIrradianceCache::~WorldIrradianceCache()
{
  if (history) {
    GPU_storagebuf_free(history);
  }
  if (current) {
    GPU_storagebuf_free(current);
  }
}

bool WorldIrradianceCache::ensure()
{
  if (!history) {
    history = GPU_storagebuf_create_ex(
        entry_count * sizeof(float4), nullptr, GPU_USAGE_DYNAMIC, "RTIrradianceHistory");
    current = GPU_storagebuf_create_ex(
        entry_count * sizeof(float4), nullptr, GPU_USAGE_DYNAMIC, "RTIrradianceCurrent");
  }
  return GPU_storagebuf_ensure_allocated(history) && GPU_storagebuf_ensure_allocated(current) &&
         GPU_storagebuf_ensure_allocated(probes);
}

void WorldIrradianceCache::reset(RayTracingScene &scene)
{
  Vector<HardwareIrradianceProbe> data;
  scene.irradiance_probes(data);
  for (int i : data.index_range()) {
    probes[i] = data[i];
  }
  probes.push_update();
  GPU_storagebuf_clear_to_zero(history);
  GPU_storagebuf_clear_to_zero(current);
  frame = 0;
}

SurfaceRadianceCache::~SurfaceRadianceCache()
{
  if (entries) {
    GPU_storagebuf_free(entries);
  }
  if (winners) {
    GPU_storagebuf_free(winners);
  }
}
bool SurfaceRadianceCache::ensure(int2 extent)
{
  if (GPU_max_storage_buffer_size() < entry_count * sizeof(HardwareRadianceEntry)) {
    return false;
  }
  if (!entries) {
    entries = GPU_storagebuf_create_ex(entry_count * sizeof(HardwareRadianceEntry),
                                       nullptr,
                                       GPU_USAGE_DYNAMIC,
                                       "HardwareRadianceCache");
    winners = GPU_storagebuf_create_ex(
        entry_count * sizeof(uint32_t), nullptr, GPU_USAGE_DYNAMIC, "HardwareRadianceWinners");
    if (!GPU_storagebuf_ensure_allocated(entries) || !GPU_storagebuf_ensure_allocated(winners)) {
      if (entries) {
        GPU_storagebuf_free(entries);
        entries = nullptr;
      }
      if (winners) {
        GPU_storagebuf_free(winners);
        winners = nullptr;
      }
      return false;
    }
    invalidate();
  }
  const auto usage = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  direct_radiance.ensure_2d(gpu::TextureFormat::SFLOAT_16_16_16_16, extent, usage);
  surface_keys.ensure_2d(gpu::TextureFormat::UINT_32_32_32_32, extent, usage);
  return direct_radiance.is_valid() && surface_keys.is_valid();
}
void SurfaceRadianceCache::invalidate()
{
  if (entries) {
    GPU_storagebuf_clear_to_zero(entries);
  }
}

HardwareRayTracing::~HardwareRayTracing()
{
  for (gpu::Shader *shader : {capture_shader_,
                              select_shader_,
                              resolve_shader_,
                              trace_shader_,
                              debug_shader_,
                              irradiance_shader_})
  {
    if (shader) {
      GPU_shader_free(shader);
    }
  }
}
void HardwareRayTracing::init()
{
  const bool was_enabled = enabled_;
  requested_ = inst_.scene->eevee.ray_tracing_method == RAYTRACE_EEVEE_METHOD_HARDWARE &&
               (inst_.scene->eevee.flag & SCE_EEVEE_SSR_ENABLED) && !inst_.is_light_bake;
  enabled_ = requested_ && GPU_ray_query_support();
  active = false;
  if (requested_ && !enabled_) {
    hardware_notice(inst_, "Hardware ray tracing unavailable: using screen tracing");
  }
  if (enabled_ && !(capture_shader_ && select_shader_ && resolve_shader_ && trace_shader_ &&
                    irradiance_shader_))
  {
    for (gpu::Shader **shader : {&capture_shader_,
                                 &select_shader_,
                                 &resolve_shader_,
                                 &trace_shader_,
                                 &irradiance_shader_})
    {
      if (*shader) {
        GPU_shader_free(*shader);
        *shader = nullptr;
      }
    }
    capture_shader_ = GPU_shader_create_from_info_name("eevee_hardware_radiance_capture");
    select_shader_ = GPU_shader_create_from_info_name("eevee_hardware_cache_select");
    resolve_shader_ = GPU_shader_create_from_info_name("eevee_hardware_cache_resolve");
    trace_shader_ = GPU_shader_create_from_info_name("eevee_hardware_trace");
    irradiance_shader_ = GPU_shader_create_from_info_name("eevee_hardware_irradiance_update");
  }
  enabled_ &= capture_shader_ && select_shader_ && resolve_shader_ && trace_shader_ &&
              irradiance_shader_;
  if (enabled_ && !was_enabled) {
    /* Dependency-graph recalc flags may have been consumed while another mode was active. */
    scene.invalidate_geometry();
    invalidated_ = true;
  }
  if (requested_ && GPU_ray_query_support() && !enabled_) {
    hardware_notice(inst_, "Hardware ray tracing shader compilation failed: using screen tracing");
  }
}
void HardwareRayTracing::begin_sync()
{
  if (!enabled_) {
    return;
  }
  light_count_ = 0;
  const uint64_t update = DEG_get_update_count(inst_.depsgraph);
  if (update != depsgraph_update_) {
    invalidated_ |= DEG_id_type_updated(inst_.depsgraph, ID_MA) ||
                    DEG_id_type_updated(inst_.depsgraph, ID_LA) ||
                    DEG_id_type_updated(inst_.depsgraph, ID_WO) ||
                    DEG_id_type_updated(inst_.depsgraph, ID_IM) ||
                    DEG_id_type_updated(inst_.depsgraph, ID_NT);
    depsgraph_update_ = update;
  }
  scene.begin_sync(invalidated_);
}
void HardwareRayTracing::sync_mesh(const ObjectHandle &object, const MaterialArray &materials)
{
  if (enabled_) {
    const auto start = std::chrono::steady_clock::now();
    invalidated_ |= (object.recalc & ID_RECALC_SHADING) != 0;
    scene.sync_mesh(object, materials);
    scene.cpu_sync_ms += std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
  }
}
void HardwareRayTracing::end_sync()
{
  if (!enabled_) {
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  scene.end_sync();
  scene.cpu_sync_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (G.debug & G_DEBUG_GPU) {
    printf("HardwareRT scene: BLAS=%d TLAS=%d bytes=%llu sync_ms=%.3f valid=%d\n",
           scene.blas_build_count,
           scene.tlas_build_count,
           (unsigned long long)scene.memory_bytes,
           scene.cpu_sync_ms,
           int(scene.ready()));
  }
  const bool pending_materials = inst_.materials.queued_shaders_count != 0 ||
                                 inst_.materials.queued_textures_count != 0;
  invalidated_ |= pending_materials || had_pending_materials_;
  had_pending_materials_ = pending_materials;
  invalidated_ |= light_count_ != previous_light_count_;
  previous_light_count_ = light_count_;
  if (!scene.ready()) {
    hardware_notice(inst_, "Hardware RT scene build failed: using screen tracing");
  }
  if (scene.materials.unsupported) {
    hardware_notice(inst_,
                    "Hardware RT: {} materials use cached/probe lighting (outside PBR subset)",
                    scene.materials.unsupported);
  }
  if (scene.skipped_materials) {
    hardware_notice(inst_,
                    "Hardware RT: {} unsupported material or geometry entries require fallback",
                    scene.skipped_materials);
  }
}

bool HardwareRayTracing::prepare(draw::View &view, gpu::Texture *emission, gpu::Texture *direct[3])
{
  active = false;
  if (!enabled_ || !scene.ready()) {
    return false;
  }
  const int2 extent = inst_.film.render_extent_get();
  if (!cache.ensure(extent) || !irradiance.ensure()) {
    hardware_notice(inst_, "Hardware RT cache allocation failed: using screen tracing");
    return false;
  }
  if (invalidated_ || scene_revision_ != scene.revision) {
    cache.invalidate();
    irradiance.reset(scene);
    scene_revision_ = scene.revision;
    invalidated_ = false;
    history_revision_++;
  }
  GPU_storagebuf_clear_to_zero(cache.winners);
  draw::PassSimple capture("HardwareRT.Capture");
  capture.shader_set(capture_shader_);
  capture.bind_acceleration_structure("rt_scene", &scene.tlas_binding);
  capture.bind_texture("rt_emission_tx", emission);
  capture.bind_texture("rt_direct_0_tx", direct[0]);
  capture.bind_texture("rt_direct_1_tx", direct[1]);
  capture.bind_texture("rt_direct_2_tx", direct[2]);
  capture.bind_image("rt_direct_img", &cache.direct_radiance);
  capture.bind_resources(inst_.gbuffer);
  capture.dispatch(math::divide_ceil(extent, int2(8)));
  capture.barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
  inst_.manager->submit(capture, view);

  draw::PassSimple select("HardwareRT.CacheSelect");
  select.shader_set(select_shader_);
  scene.bind_resources(select);
  select.bind_ssbo("rt_winners", &cache.winners);
  select.bind_image("rt_key_img", &cache.surface_keys);
  select.bind_texture("rt_depth_tx", &inst_.render_buffers.depth_tx);
  select.bind_resources(inst_.gbuffer);
  select.dispatch(math::divide_ceil(extent, int2(8)));
  select.barrier(GPU_BARRIER_SHADER_STORAGE | GPU_BARRIER_SHADER_IMAGE_ACCESS);
  inst_.manager->submit(select, view);

  draw::PassSimple resolve("HardwareRT.CacheResolve");
  resolve.shader_set(resolve_shader_);
  resolve.bind_acceleration_structure("rt_scene", &scene.tlas_binding);
  resolve.bind_ssbo("rt_winners", &cache.winners);
  resolve.bind_resources(cache);
  resolve.bind_texture("rt_direct_tx", &cache.direct_radiance);
  resolve.bind_image("rt_key_img", &cache.surface_keys);
  resolve.dispatch(math::divide_ceil(extent, int2(8)));
  resolve.barrier(GPU_BARRIER_SHADER_STORAGE | GPU_BARRIER_TEXTURE_FETCH);
  inst_.manager->submit(resolve, view);
  std::swap(irradiance.history, irradiance.current);
  draw::PassSimple update("HardwareRT.IrradianceUpdate");
  update.shader_set(irradiance_shader_);
  scene.bind_resources(update);
  cache.bind_resources(update);
  update.bind_texture("rt_direct_tx", &cache.direct_radiance);
  update.bind_ssbo("rt_irradiance_probes", &irradiance.probes);
  update.bind_ssbo("rt_irradiance", &irradiance.history);
  update.bind_ssbo("rt_irradiance_next", &irradiance.current);
  update.push_constant("rt_irradiance_frame", int(irradiance.frame++));
  inst_.lights.bind_raytrace_resources(update);
  update.bind_ubo("rt_sunlight", &inst_.world.sunlight);
  update.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
  update.bind_resources(inst_.uniform_data);
  update.bind_resources(inst_.volume_probes);
  update.bind_resources(inst_.sphere_probes);
  update.dispatch(int3(WorldIrradianceCache::entry_count / 64, 1, 1));
  update.barrier(GPU_BARRIER_SHADER_STORAGE);
  inst_.manager->submit(update, view);
  active = true;
  return true;
}

void HardwareRayTracing::debug_draw(draw::View &view, gpu::Texture *combined)
{
  if (!active || !ELEM(inst_.debug_mode,
                       DEBUG_HARDWARE_RT_NORMAL,
                       DEBUG_HARDWARE_RT_DISTANCE,
                       DEBUG_HARDWARE_RT_INSTANCE,
                       DEBUG_HARDWARE_RT_COVERAGE,
                       DEBUG_HARDWARE_RT_SHADING_NORMAL,
                       DEBUG_HARDWARE_RT_NORMAL_ERROR,
                       DEBUG_HARDWARE_RT_DIFFUSE_ALBEDO,
                       DEBUG_HARDWARE_RT_DIFFUSE_ERROR,
                       DEBUG_HARDWARE_RT_IRRADIANCE))
  {
    return;
  }
  if (!debug_shader_) {
    debug_shader_ = GPU_shader_create_from_info_name("eevee_hardware_debug");
  }
  if (!debug_shader_) {
    return;
  }
  draw::PassSimple pass("HardwareRT.Debug");
  pass.shader_set(debug_shader_);
  pass.push_constant("rt_debug_mode", int(inst_.debug_mode));
  scene.bind_resources(pass);
  cache.bind_resources(pass);
  irradiance.bind_resources(pass);
  pass.bind_texture("rt_direct_tx", &cache.direct_radiance);
  pass.bind_image("rt_debug_img", combined);
  pass.bind_texture("rt_depth_tx", &inst_.render_buffers.depth_tx);
  pass.bind_resources(inst_.gbuffer);
  pass.dispatch(math::divide_ceil(inst_.film.render_extent_get(), int2(8)));
  pass.barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
  inst_.manager->submit(pass, view);
  hardware_notice(inst_,
                  "Hardware RT: {} BLAS builds, {} TLAS builds, {} MiB scene; red = uncovered",
                  scene.blas_build_count,
                  scene.tlas_build_count,
                  scene.memory_bytes >> 20);
}
}  // namespace blender::eevee
