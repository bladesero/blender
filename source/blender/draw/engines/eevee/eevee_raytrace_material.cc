/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include "eevee_raytrace_material.hh"
#include "BKE_image.hh"
#include "BKE_image_partial_update.hh"
#include "BKE_node.hh"
#include "BKE_node_runtime.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_rotation.hh"
#include "BLI_math_vector.hh"
#include "DNA_image_types.h"
#include "DNA_material_types.h"
#include "DNA_node_types.h"
#include "GPU_capabilities.hh"
#include "GPU_storage_buffer.hh"
#include "GPU_texture.hh"
#include "IMB_colormanagement.hh"
#include "IMB_imbuf_types.hh"
#include "NOD_shader.h"
#include <cmath>
#include <functional>

namespace blender::eevee {

namespace {
struct MaterialSocket {
  const bNodeSocket *socket = nullptr;
  Vector<const bNode *> groups;
  Vector<int> conversions;
};

/* Resolve a socket in its group-instance context. The same node tree can be instantiated with
 * different inputs, so a socket pointer alone is not an expression identity. */
static MaterialSocket resolve_socket(MaterialSocket ref)
{
  for (int depth = 0; ref.socket && depth < 128; depth++) {
    const bNodeSocket &socket = *ref.socket;
    if (socket.is_input()) {
      ref.conversions.append(socket.type);
      if (!socket.is_directly_linked()) {
        return ref;
      }
      const auto links = socket.directly_linked_links();
      if (links.size() != 1 || !links[0]->is_available()) {
        return {};
      }
      ref.socket = links[0]->fromsock;
      continue;
    }
    const bNode &node = socket.owner_node();
    if (node.is_reroute()) {
      ref.socket = node.input_sockets().first();
    }
    else if (node.flag & NODE_MUTED) {
      ref.socket = socket.internal_link_input();
    }
    else if (node.is_group()) {
      auto *tree = reinterpret_cast<bNodeTree *>(node.id);
      if (!tree) {
        return {};
      }
      tree->ensure_topology_cache();
      const bNode *output = tree->group_output_node();
      if (!output) {
        return {};
      }
      ref.groups.append(&node);
      ref.socket = output->input_by_identifier(UString(StringRef(socket.identifier)));
    }
    else if (node.is_group_input()) {
      if (ref.groups.is_empty()) {
        return {};
      }
      const bNode *group = ref.groups.pop_last();
      ref.socket = group->input_by_identifier(UString(StringRef(socket.identifier)));
    }
    else {
      return ref;
    }
  }
  return {};
}
}  // namespace

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
  previous_closures_ = std::move(closures_);
  closures_.clear();
  visited_.clear();
  unsupported = 0;
  diagnostics.clear();
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
  closures_.clear();
  previous_closures_.clear();
  normal_info.clear();
  uv_names.clear();
  for (auto &group : image_groups_) {
    group.images.clear();
    group.size = int2(0);
    group.texture.free();
  }
  dummy_image_.free();
  bound_images_.fill(nullptr);
  unsupported = 0;
  dirty_ = true;
}

bool RayTracingMaterials::parameter_leaf(const bNodeSocket *input,
                                         HardwareRTParameter &param,
                                         std::string *uv_name,
                                         bool ignore_coordinates)
{
  param = {};
  if (!input) {
    return false;
  }
  const bNodeSocket *socket = input;
  if (input->is_output() || input->is_directly_linked()) {
    socket = input->is_output() ? input : linked_output(input);
    if (!socket) {
      return false;
    }
    const bNode &node = socket->owner_node();
    if (node.is_type("ShaderNodeTexImage"_ustr)) {
      const auto *tex = static_cast<const NodeTexImage *>(node.storage);
      blender::Image *image = reinterpret_cast<blender::Image *>(node.id);
      const bNodeSocket *vector = node.input_by_identifier("Vector"_ustr);
      if (!tex || !image || image->type == IMA_TYPE_MULTILAYER ||
          !ELEM(image->source, IMA_SRC_FILE, IMA_SRC_GENERATED) ||
          tex->projection != SHD_PROJ_FLAT ||
          !ELEM(tex->interpolation, SHD_INTERP_LINEAR, SHD_INTERP_CLOSEST))
      {
        return false;
      }
      if (!ignore_coordinates && vector->is_directly_linked()) {
        const bNodeSocket *uv = linked_output(vector);
        if (!uv) {
          return false;
        }
        if (uv_name && uv->owner_node().is_type("ShaderNodeUVMap"_ustr)) {
          const auto *storage = static_cast<const NodeShaderUVMap *>(uv->owner_node().storage);
          if (!storage || uv->owner_node().custom1) {
            return false;
          }
          *uv_name = storage->uv_map;
        }
        else if (!uv->owner_node().is_type("ShaderNodeTexCoord"_ustr) ||
                 !STREQ(uv->identifier, "UV"))
        {
          return false;
        }
      }
      using namespace bke::image::partial_update;
      TextureRecord &record = textures_.lookup_or_add_default(image);
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
        /* Keep existing layer indices stable when a different image changes size. The vacated

         * * layer is a reusable hole; never upload its new pixels using the old dimensions. */
        if (record.info.y != 0) {
          auto &old_group = image_groups_[record.info.x >> 24];
          old_group.images[record.info.x & 0xffffffu] = nullptr;
        }
        if (ibuf && ibuf->x > 0 && ibuf->y > 0 && ibuf->x <= GPU_max_texture_size() &&
            ibuf->y <= GPU_max_texture_size() &&
            (ibuf->byte_buffer.data || (ibuf->float_buffer.data && ELEM(ibuf->channels, 1, 3, 4))))
        {
          const int2 size(ibuf->x, ibuf->y);
          const bool floating = ibuf->float_buffer.data != nullptr;
          for (int group_index = 0; group_index < image_groups_.size(); group_index++) {
            auto &group = image_groups_[group_index];
            bool empty = true;
            for (const blender::Image *entry : group.images) {
              empty &= entry == nullptr;
            }
            if (empty || (group.size == size && group.floating == floating)) {
              if (empty) {
                group.images.clear();
              }
              group.size = size;
              group.floating = floating;
              int layer = group.images.first_index_of_try(nullptr);
              if (layer < 0) {
                if (group.images.size() >= GPU_max_texture_layers()) {
                  continue;
                }
                layer = group.images.size();
                group.images.append(image);
              }
              else {
                group.images[layer] = image;
              }
              info = uint4((uint(group_index) << 24) | uint(layer), size.x, size.y, 0);
              record.image_user = tex->iuser;
              record.dirty = true;
              break;
            }
          }
          if (info.y == 0) {
            diagnostics.append(std::string(image->id.name + 2) +
                               ": secondary-image array group or layer capacity exceeded");
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
      else {
        /* Match the Image Texture node's color output, including its alpha-socket convention.

         * * Alpha parameters keep the unmodified texture channel. Apply these operations after

         * * filtering, before the fixed affine parameter transform. */
        const bool alpha_used = node.output_by_identifier("Alpha"_ustr)->is_directly_linked();
        if (ELEM(image->alpha_mode, IMA_ALPHA_IGNORE, IMA_ALPHA_CHANNEL_PACKED) ||
            IMB_colormanagement_space_name_is_data(image->colorspace_settings.name))
        {
          param.texture.w |= 64u;
        }
        else if (image->alpha_mode == IMA_ALPHA_PREMUL) {
          param.texture.w |= alpha_used ? 32u : 64u;
        }
        else {
          param.texture.w |= alpha_used ? 64u : 16u;
        }
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
      float coefficients[3];
      IMB_colormanagement_get_luminance_coefficients(coefficients);
      param.value = float4(math::dot(param.value.xyz(), float3(coefficients)));
    }
  }
  else if (socket->type == SOCK_FLOAT) {
    param.value = float4(static_cast<const bNodeSocketValueFloat *>(socket->default_value)->value);
  }
  else if (socket->type == SOCK_VECTOR) {
    param.value = float4(
        float3(static_cast<const bNodeSocketValueVector *>(socket->default_value)->value), 0.0f);
  }
  else {
    return false;
  }
  return true;
}

bool RayTracingMaterials::parameter(const bNodeSocket *input,
                                    HardwareRTParameter &param,
                                    std::string *uv_name)
{
  parameter_error_.clear();
  auto fail = [&](MaterialSocket ref, const char *reason) {
    if (parameter_error_.empty()) {
      for (const bNode *group : ref.groups) {
        parameter_error_ += std::string(group->name) + " / ";
      }
      if (ref.socket) {
        parameter_error_ += std::string(ref.socket->owner_node().name) + " / " + ref.socket->name +
                            ": ";
      }
      parameter_error_ += reason;
    }
    return false;
  };
  auto affine = [](HardwareRTParameter &p, float4 scale, float4 offset) {
    p.value = p.value * scale + offset;
    for (int i = 0; i < 4; i++) {
      p.color_transform[i] *= scale;
    }
  };
  std::function<bool(MaterialSocket, HardwareRTParameter &, int)> compile;
  compile = [&](MaterialSocket ref, HardwareRTParameter &p, int depth) {
    if (depth > 32 || !ref.socket) {
      return fail(ref, "invalid or excessively deep lightweight parameter");
    }
    ref = resolve_socket(ref);
    if (!ref.socket) {
      return fail(ref, "invalid group, muted-node or reroute connection");
    }
    const bNode &node = ref.socket->owner_node();
    p = {};
    auto arg = [&](const bNodeSocket *socket, HardwareRTParameter &out) {
      return compile({socket, ref.groups}, out, depth + 1);
    };
    if (ref.socket->is_input() || node.is_type("ShaderNodeRGB"_ustr) ||
        node.is_type("ShaderNodeValue"_ustr))
    {
      if (!parameter_leaf(ref.socket, p)) {
        return fail(ref, "unsupported constant type");
      }
    }
    else if (node.is_type("ShaderNodeTexImage"_ustr)) {
      if (!parameter_leaf(ref.socket, p, nullptr, true)) {
        return fail(ref, "unsupported or unavailable image resource");
      }
      p.value = float4(0);
      p.color_transform = float4x4::identity();
      p.uv_row0 = float4(1, 0, 0, 0);
      p.uv_row1 = float4(0, 1, 0, 0);
      std::string name;
      MaterialSocket coord = resolve_socket({node.input_by_identifier("Vector"_ustr), ref.groups});
      float4x4 transform = float4x4::identity();
      bool explicit_mapping = false;
      if (coord.socket && coord.socket->is_output() &&
          coord.socket->owner_node().is_type("ShaderNodeMapping"_ustr))
      {
        explicit_mapping = true;
        const bNode &mapping = coord.socket->owner_node();
        HardwareRTParameter location{}, rotation{}, scale{};
        if (!compile({mapping.input_by_identifier("Location"_ustr), coord.groups},
                     location,
                     depth + 1) ||
            !compile({mapping.input_by_identifier("Rotation"_ustr), coord.groups},
                     rotation,
                     depth + 1) ||
            !compile(
                {mapping.input_by_identifier("Scale"_ustr), coord.groups}, scale, depth + 1) ||
            location.texture.y || rotation.texture.y || scale.texture.y ||
            mapping.custom1 == NODE_MAPPING_TYPE_NORMAL)
        {
          return fail(coord, "Mapping requires constant affine parameters");
        }
        transform = math::from_loc_rot_scale<float4x4>(
            mapping.custom1 == NODE_MAPPING_TYPE_VECTOR ? float3(0) : location.value.xyz(),
            math::EulerXYZ(rotation.value.xyz()),
            scale.value.xyz());
        if (mapping.custom1 == NODE_MAPPING_TYPE_TEXTURE) {
          if ((scale.value.x == 0.0f || scale.value.y == 0.0f || scale.value.z == 0.0f)) {
            return fail(coord, "singular texture Mapping");
          }
          transform = math::invert(transform);
        }
        coord = resolve_socket({mapping.input_by_identifier("Vector"_ustr), coord.groups});
      }
      if (!coord.socket) {
        return fail(ref, "invalid image coordinates");
      }
      if (coord.socket->is_output()) {
        const bNode &source = coord.socket->owner_node();
        if (source.is_type("ShaderNodeUVMap"_ustr) && !source.custom1) {
          name = static_cast<const NodeShaderUVMap *>(source.storage)->uv_map;
        }
        else if (!(source.is_type("ShaderNodeTexCoord"_ustr) &&
                   STREQ(coord.socket->identifier, "UV") && !source.custom1))
        {
          return fail(coord, "lightweight images require UV coordinates");
        }
      }
      if (explicit_mapping && coord.socket->is_input()) {
        HardwareRTParameter vector{};
        if (!parameter_leaf(coord.socket, vector)) {
          return fail(coord, "invalid default Mapping vector");
        }
        const float4 mapped = transform * float4(vector.value.xyz(), 1.0f);
        p.mapping.z = 1;
        p.uv_row0 = float4(0, 0, 0, mapped.x);
        p.uv_row1 = float4(0, 0, 0, mapped.y);
      }
      else {
        int index = parameter_uvs_.first_index_of_try(name);
        if (index < 0) {
          index = parameter_uvs_.size();
          parameter_uvs_.append(name);
        }
        if (uv_name) {
          *uv_name = name;
        }
        p.mapping.x = index;
        for (int i = 0; i < 4; i++) {
          p.uv_row0[i] = transform[i][0];
          p.uv_row1[i] = transform[i][1];
        }
      }
      if (p.texture.w & 8u) {
        p.texture.w &= ~8u;
        p.color_transform = float4x4::zero();
        p.color_transform[3] = float4(1);
      }
    }
    else if (node.is_type("ShaderNodeMath"_ustr)) {
      HardwareRTParameter a{}, b{};
      if (!arg(node.input_sockets()[0], a) || !arg(node.input_sockets()[1], b)) {
        return false;
      }
      if ((a.texture.y && b.texture.y) || a.mapping.y || b.mapping.y) {
        return fail(ref, "parameter requires multiple textures or intermediate clamp");
      }
      const bool bc = b.texture.y == 0;
      if (node.custom1 == NODE_MATH_MULTIPLY) {
        p = bc ? a : b;
        affine(p, float4(bc ? b.value.x : a.value.x), float4(0));
      }
      else if (node.custom1 == NODE_MATH_ADD) {
        p = bc ? a : b;
        affine(p, float4(1), float4(bc ? b.value.x : a.value.x));
      }
      else if (node.custom1 == NODE_MATH_SUBTRACT) {
        p = bc ? a : b;
        affine(p, float4(bc ? 1 : -1), float4(bc ? -b.value.x : a.value.x));
      }
      else if (node.custom1 == NODE_MATH_DIVIDE && bc) {
        p = a;
        affine(p, float4(b.value.x == 0 ? 0 : 1 / b.value.x), float4(0));
      }
      else {
        return fail(ref, "Math is outside affine lightweight parameters");
      }
      p.mapping.y = (node.custom2 & SHD_MATH_CLAMP) != 0;
    }
    else if (node.is_type("ShaderNodeMixRGB"_ustr) && node.custom1 == MA_RAMP_MULT) {
      HardwareRTParameter factor{}, a{}, b{};
      if (!arg(node.input_sockets()[0], factor) || !arg(node.input_sockets()[1], a) ||
          !arg(node.input_sockets()[2], b))
      {
        return false;
      }
      if (factor.texture.y || (a.texture.y && b.texture.y) || a.mapping.y || b.mapping.y ||
          (node.custom2 & SHD_MIXRGB_USE_ALPHA))
      {
        return fail(ref, "Multiply requires a constant factor and one constant color");
      }
      const float f = std::clamp(factor.value.x, 0.0f, 1.0f);
      if (b.texture.y) {
        p = b;
        affine(p, a.value * f, a.value * (1 - f));
      }
      else {
        p = a;
        affine(p, float4(1 - f) + b.value * f, float4(0));
      }
      p.mapping.y = (node.custom2 & SHD_MIXRGB_CLAMP) != 0;
    }
    else if (node.is_type("ShaderNodeSeparateColor"_ustr) &&
             static_cast<const NodeCombSepColor *>(node.storage)->mode == NODE_COMBSEP_COLOR_RGB)
    {
      if (!arg(node.input_sockets()[0], p) || p.mapping.y) {
        return fail(ref, "unsupported channel input");
      }
      const int channel = STREQ(ref.socket->identifier, "Red")   ? 0 :
                          STREQ(ref.socket->identifier, "Green") ? 1 :
                                                                   2;
      p.value = float4(p.value[channel]);
      for (int i = 0; i < 4; i++) {
        p.color_transform[i] = float4(p.color_transform[i][channel]);
      }
    }
    else {
      return fail(ref, "node uses native screen/probe shading (outside lightweight parameters)");
    }
    int source = ref.socket->type;
    for (int i = int(ref.conversions.size()) - 1; i >= 0; i--) {
      const int destination = ref.conversions[i];
      if (destination == source) {
        continue;
      }
      if (!ELEM(source, SOCK_FLOAT, SOCK_VECTOR, SOCK_RGBA) ||
          !ELEM(destination, SOCK_FLOAT, SOCK_VECTOR, SOCK_RGBA) || p.mapping.y)
      {
        return fail(ref, "unsupported parameter conversion after clamp");
      }
      float3 coefficients(1.0f / 3.0f);
      if (source == SOCK_RGBA) {
        IMB_colormanagement_get_luminance_coefficients(&coefficients.x);
      }
      auto convert = [&](float4 v, bool offset) {
        if (destination == SOCK_FLOAT) {
          return float4(math::dot(v.xyz(), coefficients));
        }
        if (source == SOCK_FLOAT) {
          return float4(float3(v.x), destination == SOCK_RGBA && offset ? 1.0f : 0.0f);
        }
        return float4(v.xyz(), destination == SOCK_RGBA && offset ? 1.0f : 0.0f);
      };
      p.value = convert(p.value, true);
      for (int column = 0; column < 4; column++) {
        p.color_transform[column] = convert(p.color_transform[column], false);
      }
      source = destination;
    }
    if (!p.texture.y && p.mapping.y) {
      p.value = math::clamp(p.value, 0.0f, 1.0f);
      p.mapping.y = 0;
    }
    return true;
  };
  return compile({input, parameter_context_}, param, 0);
}

uint32_t RayTracingMaterials::sync(const blender::Material *material)
{
  if (!visited_.add(material)) {
    return indices_.lookup(material);
  }
  HardwareRTMaterial data{};
  NormalInfo normal;
  std::string normal_error;
  bool supported = false;
  parameter_context_.clear();
  parameter_uvs_.clear();
  if (material && material->nodetree) {
    bNodeTree *tree = material->nodetree;
    tree->ensure_topology_cache();
    const bNode *output = ntreeShaderOutputNode(tree, SHD_OUTPUT_EEVEE);
    MaterialSocket resolved = output ? resolve_socket(
                                           {output->input_by_identifier("Surface"_ustr), {}}) :
                                       MaterialSocket{};
    auto compile_leaf = [&](MaterialSocket leaf) {
      data = {};
      normal = {};
      supported = false;
      parameter_context_ = leaf.groups;
      const bNodeSocket *surface = leaf.socket;
      const bNode &node = surface->owner_node();
      std::string node_path;
      for (const bNode *group : leaf.groups) {
        node_path += std::string(group->name) + " / ";
      }
      node_path += node.name;
      auto read = [&](UString name, HardwareRTParameter &p) {
        const bNodeSocket *input = node.input_by_identifier(name);
        const bool valid = parameter(input, p);
        if (!valid && normal_error.empty()) {
          normal_error = parameter_error_.empty() ?
                             node_path + " / " + (input ? input->name : "<missing input>") +
                                 ": secondary-hit parameter expression is unsupported" :
                             parameter_error_;
          if (input && input->is_directly_linked()) {
            if (const bNodeSocket *source = linked_output(input)) {
              normal_error += " (from " + std::string(source->owner_node().name) + " / " +
                              source->identifier + ")";
            }
          }
        }
        return valid;
      };
      auto constant = [&](UString name, float expected) {
        HardwareRTParameter p{};
        const bool valid = read(name, p) && p.texture.y == 0 && p.value.x == expected;
        if (!valid && normal_error.empty()) {
          const bNodeSocket *input = node.input_by_identifier(name);
          normal_error = node_path + " / " + (input ? input->name : "<missing input>") +
                         ": secondary-hit closure currently requires constant " +
                         std::to_string(expected);
        }
        return valid;
      };
      data.properties = float4(1.5f, 0.5f, 1.0f, 0.0f);
      data.roughness.value = float4(1.0f);
      data.alpha.value = float4(1.0f);
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
        if (normal_error.empty() && (ior.texture.y || specular.texture.y || tint.texture.y ||
                                     tint.value.xyz() != float3(1.0f)))
        {
          normal_error = node_path +
                         ": secondary Principled requires constant IOR/Specular IOR Level and "
                         "white Specular Tint";
        }
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
        supported &= read("Alpha"_ustr, data.alpha);
      }
      else if (node.is_type("ShaderNodeBsdfDiffuse"_ustr)) {
        supported = read("Color"_ustr, data.base_color) && constant("Roughness"_ustr, 0.0f);
        data.properties.w = 1.0f;
      }
      else if (node.is_type("ShaderNodeBsdfTransparent"_ustr)) {
        supported = read("Color"_ustr, data.base_color);
        data.properties.w = 2.0f;
      }
      else if (node.is_type("ShaderNodeEmission"_ustr)) {
        supported = read("Color"_ustr, data.emission) &&
                    read("Strength"_ustr, data.emission_strength);
        data.properties.w = 1.0f;
      }
      else {
        normal_error = node_path + " (" + node.idname +
                       "): secondary-hit closure is not implemented";
      }
      if (const bNodeSocket *input = node.input_by_identifier("Normal"_ustr)) {
        if (input->is_directly_linked()) {
          const MaterialSocket normal_socket = resolve_socket({input, parameter_context_});
          const bNode *map = normal_socket.socket ? &normal_socket.socket->owner_node() : nullptr;
          if (map && map->is_type("ShaderNodeNormalMap"_ustr)) {
            const auto saved_context = parameter_context_;
            parameter_context_ = normal_socket.groups;
            const auto *storage = static_cast<const NodeShaderNormalMap *>(map->storage);
            normal.enabled = true;
            normal.tangent = storage->space == SHD_SPACE_TANGENT;
            normal.tangent_uv = storage->uv_map;
            const bool valid = parameter(map->input_by_identifier("Color"_ustr),
                                         data.normal_color) &&
                               parameter(map->input_by_identifier("Strength"_ustr),
                                         data.normal_strength);
            parameter_context_ = saved_context;
            supported &= valid;
            if (!valid) {
              normal_error = parameter_error_.empty() ?
                                 node_path + ": unsupported Normal Map Color/Strength expression" :
                                 parameter_error_;
            }
            data.normal_info = uint4(
                1, storage->space, storage->convention == SHD_NORMAL_MAP_CONVENTION_DIRECTX, 0);
            if (normal.tangent) {
              int index = parameter_uvs_.first_index_of_try(normal.tangent_uv);
              if (index < 0) {
                index = parameter_uvs_.size();
                parameter_uvs_.append(normal.tangent_uv);
              }
              data.normal_info.w = index;
              normal.tangent_uvs.append(normal.tangent_uv);
            }
          }
          else {
            supported = false;
            normal_error = node_path +
                           " / Normal: expected a Normal Map node (possibly inside a group); "
                           "Bump and arbitrary normal expressions are not implemented";
          }
        }
      }
      if (const bNodeSocket *input = node.input_by_identifier("Tangent"_ustr)) {
        supported &= !input->is_directly_linked();
      }
      if (output && output->input_by_identifier("Displacement"_ustr)->is_directly_linked()) {
        supported = false;
        normal_error = "material displacement is unsupported by secondary-hit shading";
      }
      /* This hidden input is initialized to zero and filled by GPU graph compilation.
       * A root closure has unit weight; its stored socket value is not a material parameter. */
      if (const bNodeSocket *weight = node.input_by_identifier("Weight"_ustr)) {
        supported &= !weight->is_directly_linked();
      }
      return supported;
    };
    auto multiply_weight = [&](HardwareRTParameter parent,
                               HardwareRTParameter factor,
                               bool inverse,
                               HardwareRTParameter &result) {
      if (parent.texture.y || factor.mapping.y) {
        normal_error = "nested texture-weighted Mix uses native screen/probe shading";
        return false;
      }
      result = factor;
      result.mapping.y = 1;
      const float sign = inverse ? -1.0f : 1.0f;
      result.value = parent.value * (float4(inverse ? 1.0f : 0.0f) + factor.value * sign);
      for (int i = 0; i < 4; i++) {
        result.color_transform[i] *= parent.value * sign;
      }
      return true;
    };
    const uint first_closure = closures_.size();
    NormalInfo first_normal;
    std::function<bool(MaterialSocket, HardwareRTParameter, int)> visit;
    visit = [&](MaterialSocket ref, HardwareRTParameter weight, int depth) {
      ref = resolve_socket(ref);
      if (!ref.socket || depth >= 32 || closures_.size() - first_closure >= 16) {
        normal_error = "invalid shader graph or closure capacity exceeded";
        return false;
      }
      const bNode &node = ref.socket->owner_node();
      if (node.is_type("ShaderNodeAddShader"_ustr)) {
        normal_error = std::string(node.name) + ": Add Shader uses native screen/probe shading";
        return false;
      }
      if (node.is_type("ShaderNodeMixShader"_ustr)) {
        const bool mix = true;
        MaterialSocket left = resolve_socket({node.input_sockets()[1], ref.groups});
        MaterialSocket right = resolve_socket({node.input_sockets()[2], ref.groups});
        const bool simple = depth == 0 && left.socket && right.socket &&
                            (left.socket->owner_node().is_type("ShaderNodeBsdfTransparent"_ustr) ||
                             right.socket->owner_node().is_type("ShaderNodeBsdfTransparent"_ustr));
        if (!simple) {
          normal_error = std::string(node.name) +
                         ": only a surface/Transparent Mix is lightweight";
          return false;
        }
        HardwareRTParameter a = weight, b = weight;
        if (mix) {
          HardwareRTParameter factor;
          parameter_context_ = ref.groups;
          if (!parameter(node.input_sockets()[0], factor) ||
              !multiply_weight(weight, factor, true, a) ||
              !multiply_weight(weight, factor, false, b))
          {
            if (normal_error.empty()) {
              normal_error = parameter_error_;
            }
            return false;
          }
        }
        return visit({node.input_sockets()[mix ? 1 : 0], ref.groups}, a, depth + 1) &&
               visit({node.input_sockets()[mix ? 2 : 1], ref.groups}, b, depth + 1);
      }
      if (!compile_leaf(ref)) {
        return false;
      }
      if (closures_.size() == first_closure) {
        first_normal = normal;
      }
      else {
        for (const std::string &name : normal.tangent_uvs) {
          if (!first_normal.tangent_uvs.contains(name)) {
            first_normal.tangent_uvs.append(name);
          }
        }
      }
      data.weight = weight;
      data.properties.z = 1.0f;
      const uint index = closures_.size();
      changed |= previous_closures_.size() <= index ||
                 memcmp(&previous_closures_[index], &data, sizeof(data)) != 0;
      closures_.append(data);
      dirty_ = true;
      return true;
    };
    HardwareRTParameter weight{};
    weight.value = float4(1.0f);
    supported = visit(resolved, weight, 0);
    if (supported) {
      data = closures_[first_closure];
      data.closures = uint4(first_closure, closures_.size() - first_closure, 0, 0);
      normal = first_normal;
    }
  }
  data.properties.z = supported ? 1.0f : 0.0f;
  data.capability.x = supported ? 1u : 0u;
  data.capability.y = material && material->surface_render_method == MA_SURFACE_METHOD_DEFERRED;
  if (!supported) {
    normal = {};
    parameter_uvs_.clear();
    data.closures = uint4(0);
  }
  unsupported += !supported;
  if (!supported) {
    diagnostics.append(std::string(material ? material->id.name + 2 : "<default>") + ": " +
                       (normal_error.empty() ?
                            "unsupported closure, parameter link or layered material" :
                            normal_error));
  }
  if (const uint32_t *index = indices_.lookup_ptr(material)) {
    if (memcmp(&materials_[*index], &data, sizeof(data)) != 0) {
      materials_[*index] = data;
      dirty_ = changed = true;
    }
    const NormalInfo &previous = normal_info[*index];
    changed |= previous.enabled != normal.enabled || previous.tangent != normal.tangent ||
               previous.tangent_uv != normal.tangent_uv;
    changed |= previous.tangent_uvs != normal.tangent_uvs;
    normal_info[*index] = normal;
    changed |= uv_names[*index] != parameter_uvs_;
    uv_names[*index] = parameter_uvs_;
    return *index;
  }
  const uint32_t index = materials_.size();
  materials_.append(data);
  normal_info.append(normal);
  uv_names.append(parameter_uvs_);
  indices_.add(material, index);
  dirty_ = changed = true;
  return index;
}

bool RayTracingMaterials::upload()
{
  dummy_image_.ensure_2d_array(gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 1);
  if (!dummy_image_.is_valid()) {
    return false;
  }
  dummy_image_.clear(float4(0));
  for (int group_index = 0; group_index < image_groups_.size(); group_index++) {
    auto &group = image_groups_[group_index];
    if (group.images.is_empty()) {
      bound_images_[group_index] = dummy_image_;
      continue;
    }
    const int levels = 1 + int(std::floor(std::log2(float(std::max(group.size.x, group.size.y)))));
    /* Read-only arrays use backend mip generation, preserving the bound shader's specialization.
     * The compute mip helper cannot restore that state during interactive material updates. */
    const bool resized = group.texture.ensure_2d_array(group.floating ?
                                                           gpu::TextureFormat::SFLOAT_32_32_32_32 :
                                                           gpu::TextureFormat::SFLOAT_16_16_16_16,
                                                       group.size,
                                                       group.images.size(),
                                                       GPU_TEXTURE_USAGE_SHADER_READ,
                                                       nullptr,
                                                       levels);
    if (!group.texture.is_valid()) {
      diagnostics.append("secondary-image array allocation failed");
      return false;
    }
    bound_images_[group_index] = group.texture;
    bool updated = false;
    for (int layer : group.images.index_range()) {
      if (!group.images[layer]) {
        continue;
      }
      auto &record = textures_.lookup(group.images[layer]);
      if (!resized && !record.dirty) {
        continue;
      }
      void *lock = nullptr;
      ImBuf *ibuf = BKE_image_acquire_ibuf(record.image, &record.image_user, &lock);
      if (!ibuf || ibuf->x != group.size.x || ibuf->y != group.size.y) {
        BKE_image_release_ibuf(record.image, ibuf, lock);
        diagnostics.append("secondary-image dimensions changed during upload");
        return false;
      }
      Array<float4> pixels(int64_t(ibuf->x) * ibuf->y);
      IMB_colormanagement_imbuf_to_float_texture(
          &pixels[0].x,
          0,
          0,
          ibuf->x,
          ibuf->y,
          ibuf,
          BKE_image_has_gpu_texture_premultiplied_alpha(record.image, ibuf));
      GPU_texture_update_sub(
          group.texture, GPU_DATA_FLOAT, pixels.data(), 0, 0, layer, ibuf->x, ibuf->y, 1);
      BKE_image_release_ibuf(record.image, ibuf, lock);
      record.dirty = false;
      updated = true;
    }
    if (resized || updated) {
      GPU_texture_update_mipmap_chain(group.texture);
      GPU_texture_mipmap_mode(group.texture, true, true);
    }
  }
  if (dirty_) {
    if (materials_.size() * sizeof(HardwareRTMaterial) > size_t(GPU_max_storage_buffer_size())) {
      return false;
    }
    material_buf_.resize(std::max<int64_t>(1, materials_.size()));
    for (int i : materials_.index_range()) {
      material_buf_[i] = materials_[i];
    }
    material_buf_.push_update();
    if (closures_.size() * sizeof(HardwareRTMaterial) > size_t(GPU_max_storage_buffer_size())) {
      diagnostics.append("secondary material closure buffer capacity exceeded");
      return false;
    }
    closure_buf_.resize(std::max<int64_t>(1, closures_.size()));
    for (int i : closures_.index_range()) {
      closure_buf_[i] = closures_[i];
    }
    closure_buf_.push_update();
    dirty_ = false;
  }
  return GPU_storagebuf_ensure_allocated(material_buf_) &&
         GPU_storagebuf_ensure_allocated(closure_buf_);
}
}  // namespace blender::eevee
