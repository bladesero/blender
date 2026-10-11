/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once
#include "GPU_dlss.hh"

namespace blender::gpu {
class VKDLSS final : public DLSS {
  struct State;
  std::unique_ptr<State> state_;
  VKDLSS();

 public:
  ~VKDLSS() override;
  static std::unique_ptr<DLSS> create(std::string &error, DLSSMode mode);
  bool configure(int2 output_size, DLSSQuality quality, int2 &input_size) override;
  bool evaluate(const DLSSFrame &frame) override;
  bool failed() const override;
  std::string error() const override;
};
}  // namespace blender::gpu
