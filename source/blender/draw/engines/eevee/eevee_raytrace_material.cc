/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include "eevee_raytrace_material.hh"
#include "BKE_image.hh"
#include "BKE_image_partial_update.hh"
#include "BKE_node.hh"
#include "BKE_node_runtime.hh"
#include "DNA_image_types.h"
#include "DNA_material_types.h"
#include "DNA_node_types.h"
#include "GPU_capabilities.hh"
#include "GPU_storage_buffer.hh"
#include "IMB_colormanagement.hh"
#include "IMB_imbuf_types.hh"
#include "NOD_shader.h"

namespace blender::eevee {

static const bNodeSocket *linked_output(const bNodeSocket *input)
{
  /* Following reroutes only also prevents accidentally interpreting groups or muted nodes. */
  for (int depth = 0; input && depth < 32; depth++) {
    const auto links = input->directly_linked_links();
    if (links.size() != 1 || !links[0]->is_available()) {
      return nullptr;
    }
    const bNode *node = links[0]->fromnode;
    if (node->flag & NODE_MUTED) {
      return nullptr;
    }
    if (!node->is_reroute()) {
      return links[0]->fromsock;
    }
    input = node->input_sockets().first();
  }
  return nullptr;
}

RayTracingMaterials::~RayTracingMaterials()
{
  invalidate();
}

void RayTracingMaterials::begin_sync()
{
  visited_.clear();
  unsupported = 0;
  changed = false;
}

void RayTracingMaterials::invalidate()
{
  indices_.clear();
  for (const auto &texture : textures_.values()) {
    if (texture.user) {
      BKE_image_partial_update_free(texture.user);
    }
  }
  textures_.clear();
  visited_.clear();
  materials_.clear();
  pixels_.clear();
  unsupported = 0;
  dirty_ = true;
}

bool RayTracingMaterials::parameter(const bNodeSocket *input, HardwareRTParameter &param)
{
  param = {};
  if (!input) {
    return false;
  }
  const bNodeSocket *socket = input;
  if (input->is_directly_linked()) {
    socket = linked_output(input);
    if (!socket) {
      return false;
    }
    const bNode &node = socket->owner_node();
    if (node.is_type("ShaderNodeTexImage"_ustr)) {
      const auto *tex = static_cast<const NodeTexImage *>(node.storage);
      blender::Image *image = reinterpret_cast<blender::Image *>(node.id);
      const bNodeSocket *vector = node.input_by_identifier("Vector"_ustr);
      if (!tex || !image || !ELEM(image->source, IMA_SRC_FILE, IMA_SRC_GENERATED) ||
          tex->projection != SHD_PROJ_FLAT ||
          !ELEM(tex->interpolation, SHD_INTERP_LINEAR, SHD_INTERP_CLOSEST))
      {
        return false;
      }
      if (vector->is_directly_linked()) {
        const bNodeSocket *uv = linked_output(vector);
        if (!uv || !uv->owner_node().is_type("ShaderNodeTexCoord"_ustr) ||
            !STREQ(uv->identifier, "UV"))
        {
          return false;
        }
      }
      using namespace bke::image::partial_update;
      TextureRecord &record = textures_.lookup_or_add_default(&node);
      if (record.image != image || record.session_uid != image->id.session_uid) {
        if (record.user) {
          BKE_image_partial_update_free(record.user);
        }
        record = {};
        record.image = image;
        record.session_uid = image->id.session_uid;
        record.user = BKE_image_partial_update_create(image);
      }
      if (BKE_image_partial_update_collect_changes(image, record.user) !=
          ePartialUpdateCollectResult::NoChangesDetected)
      {
        void *lock = nullptr;
        ImageUser user = tex->iuser;
        ImBuf *ibuf = BKE_image_acquire_ibuf(image, &user, &lock);
        uint4 info(0u);
        /* 64 MiB budget, shared by all secondary-material images. No silent downsampling. */
        if (ibuf && ibuf->x > 0 && ibuf->y > 0 &&
            (ibuf->byte_buffer.data || (ibuf->float_buffer.data && ELEM(ibuf->channels, 1, 3, 4))))
        {
          const bool reuse = record.info.y == uint(ibuf->x) && record.info.z == uint(ibuf->y);
          const size_t offset = reuse ? record.info.x : pixels_.size();
          if (size_t(ibuf->x) * ibuf->y + offset <= (1u << 22)) {
            if (!reuse) {
              pixels_.resize(offset + size_t(ibuf->x) * ibuf->y);
            }
            /* Match Blender's scene-linear texture and alpha conversion, including Non-Color. */
            IMB_colormanagement_imbuf_to_float_texture(
                &pixels_[offset].x, 0, 0, ibuf->x, ibuf->y, ibuf, false);
            info = uint4(offset, ibuf->x, ibuf->y, 0);
          }
        }
        BKE_image_release_ibuf(image, ibuf, lock);
        record.info = info;
        dirty_ = changed = true;
      }
      param.texture = record.info;
      param.texture.w = uint(tex->extension) |
                        (uint(tex->interpolation == SHD_INTERP_CLOSEST) << 2);
      if (STREQ(socket->identifier, "Alpha")) {
        param.texture.w |= 8u;
      }
      else if (!STREQ(socket->identifier, "Color")) {
        return false;
      }
      /* Blender's color-to-value conversion is the arithmetic channel average. */
      if (input->type == SOCK_FLOAT && !(param.texture.w & 8u)) {
        param.texture.w |= 16u;
      }
      return param.texture.y != 0;
    }
    if (!node.is_type("ShaderNodeRGB"_ustr) && !node.is_type("ShaderNodeValue"_ustr)) {
      return false;
    }
  }
  if (socket->type == SOCK_RGBA) {
    param.value = float4(static_cast<const bNodeSocketValueRGBA *>(socket->default_value)->value);
    if (input->type == SOCK_FLOAT) {
      param.value = float4((param.value.x + param.value.y + param.value.z) / 3.0f);
    }
  }
  else if (socket->type == SOCK_FLOAT) {
    param.value = float4(static_cast<const bNodeSocketValueFloat *>(socket->default_value)->value);
  }
  else {
    return false;
  }
  return true;
}

uint32_t RayTracingMaterials::sync(const blender::Material *material)
{
  if (!visited_.add(material)) {
    return indices_.lookup(material);
  }
  HardwareRTMaterial data{};
  bool supported = false;
  if (material && material->nodetree) {
    bNodeTree *tree = material->nodetree;
    tree->ensure_topology_cache();
    const bNode *output = ntreeShaderOutputNode(tree, SHD_OUTPUT_EEVEE);
    const bNodeSocket *surface = output ?
                                     linked_output(output->input_by_identifier("Surface"_ustr)) :
                                     nullptr;
    if (surface) {
      const bNode &node = surface->owner_node();
      auto read = [&](UString name, HardwareRTParameter &p) {
        return parameter(node.input_by_identifier(name), p);
      };
      auto constant = [&](UString name, float expected) {
        HardwareRTParameter p{};
        return read(name, p) && p.texture.y == 0 && p.value.x == expected;
      };
      data.properties = float4(1.5f, 0.5f, 1.0f, 0.0f);
      data.roughness.value = float4(1.0f);
      data.emission_strength.value = float4(1.0f);
      if (node.is_type("ShaderNodeBsdfPrincipled"_ustr)) {
        supported = read("Base Color"_ustr, data.base_color) &&
                    read("Roughness"_ustr, data.roughness) &&
                    read("Metallic"_ustr, data.metallic) &&
                    read("Emission Color"_ustr, data.emission) &&
                    read("Emission Strength"_ustr, data.emission_strength);
        HardwareRTParameter ior{}, specular{}, tint{};
        supported &= read("IOR"_ustr, ior) && !ior.texture.y &&
                     read("Specular IOR Level"_ustr, specular) && !specular.texture.y &&
                     read("Specular Tint"_ustr, tint) && !tint.texture.y &&
                     tint.value.xyz() == float3(1.0f);
        data.properties.x = ior.value.x;
        data.properties.y = specular.value.x;
        for (UString name : {"Subsurface Weight"_ustr,
                             "Transmission Weight"_ustr,
                             "Coat Weight"_ustr,
                             "Sheen Weight"_ustr,
                             "Anisotropic"_ustr,
                             "Diffuse Roughness"_ustr,
                             "Thin Film Thickness"_ustr})
        {
          supported &= constant(name, 0.0f);
        }
        supported &= constant("Alpha"_ustr, 1.0f);
      }
      else if (node.is_type("ShaderNodeBsdfDiffuse"_ustr)) {
        supported = read("Color"_ustr, data.base_color) && constant("Roughness"_ustr, 0.0f);
        data.properties.w = 1.0f;
      }
      else if (node.is_type("ShaderNodeEmission"_ustr)) {
        supported = read("Color"_ustr, data.emission) &&
                    read("Strength"_ustr, data.emission_strength);
        data.properties.w = 1.0f;
      }
      /* Normal maps, arbitrary normal/tangent inputs and layered weights need another evaluator.
       */
      for (UString name : {"Normal"_ustr, "Tangent"_ustr}) {
        if (const bNodeSocket *s = node.input_by_identifier(name)) {
          supported &= !s->is_directly_linked();
        }
      }
      /* This hidden input is initialized to zero and filled by GPU graph compilation.
       * A root closure has unit weight; its stored socket value is not a material parameter. */
      if (const bNodeSocket *weight = node.input_by_identifier("Weight"_ustr)) {
        supported &= !weight->is_directly_linked();
      }
    }
  }
  data.properties.z = supported ? 1.0f : 0.0f;
  unsupported += !supported;
  if (const uint32_t *index = indices_.lookup_ptr(material)) {
    if (memcmp(&materials_[*index], &data, sizeof(data)) != 0) {
      materials_[*index] = data;
      dirty_ = changed = true;
    }
    return *index;
  }
  const uint32_t index = materials_.size();
  materials_.append(data);
  indices_.add(material, index);
  dirty_ = changed = true;
  return index;
}

bool RayTracingMaterials::upload()
{
  if (dirty_) {
    if (materials_.size() * sizeof(HardwareRTMaterial) > size_t(GPU_max_storage_buffer_size()) ||
        pixels_.size() * sizeof(float4) > size_t(GPU_max_storage_buffer_size()))
    {
      return false;
    }
    material_buf_.resize(std::max<int64_t>(1, materials_.size()));
    pixel_buf_.resize(std::max<int64_t>(1, pixels_.size()));
    for (int i : materials_.index_range()) {
      material_buf_[i] = materials_[i];
    }
    for (int i : pixels_.index_range()) {
      pixel_buf_[i] = pixels_[i];
    }
    material_buf_.push_update();
    pixel_buf_.push_update();
    dirty_ = false;
  }
  return GPU_storagebuf_ensure_allocated(material_buf_) &&
         GPU_storagebuf_ensure_allocated(pixel_buf_);
}
}  // namespace blender::eevee
