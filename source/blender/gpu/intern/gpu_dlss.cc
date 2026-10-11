/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GPU_dlss.hh"
#include "GPU_context.hh"
#include "GPU_platform.hh"

#ifdef WITH_NVIDIA_DLSS
#  include "vulkan/vk_dlss.hh"
#endif

namespace blender::gpu {
std::unique_ptr<DLSS> DLSS::create(std::string &error, DLSSMode mode)
{
#ifdef WITH_NVIDIA_DLSS
  if (GPU_backend_get_type() == GPU_BACKEND_VULKAN) {
    return VKDLSS::create(error, mode);
  }
  error = "DLSS requires the Vulkan backend";
#else
  (void)mode;
  error = "This build does not include NVIDIA DLSS";
#endif
  return nullptr;
}
}  // namespace blender::gpu
