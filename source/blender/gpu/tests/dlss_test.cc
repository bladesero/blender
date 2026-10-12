/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "BLI_array.hh"
#include "BLI_math_matrix.hh"
#include "BLI_vector.hh"
#include "GPU_dlss.hh"
#include "GPU_state.hh"
#include "GPU_texture.hh"
#include "MEM_guardedalloc.h"
#include "gpu_testing.hh"

namespace blender::gpu::tests {

/* Deliberately independent of EEVEE material eligibility: these tests distinguish SDK input
 * semantics from a successful native fallback. Requires the RR runtime next to the executable. */
struct RRFixture {
  std::unique_ptr<DLSS> dlss;
  Vector<Texture *> textures;
  DLSSFrame frame;
  int2 output_size = int2(192, 128), input_size;
  std::string error;

  ~RRFixture()
  {
    /* Release an active feature before its recorded external node has necessarily executed. */
    dlss.reset();
    for (Texture *texture : textures) {
      GPU_texture_free(texture);
    }
  }

  Texture *make(const char *name, TextureFormat format, float4 value, bool output = false)
  {
    const int2 size = output ? output_size : input_size;
    Texture *texture = GPU_texture_create_2d(
        name, size.x, size.y, 1, format, GPU_TEXTURE_USAGE_GENERAL, nullptr);
    if (texture) {
      textures.append(texture);
      GPU_texture_clear(texture, GPU_DATA_FLOAT, value);
    }
    return texture;
  }

  bool init(DLSSMode mode = DLSSMode::RayReconstruction)
  {
    dlss = DLSS::create(error, mode);
    if (!dlss || !dlss->configure(output_size, DLSSQuality::Quality, input_size)) {
      if (dlss) {
        error = dlss->error();
      }
      return false;
    }
    frame.color = make("RR color", TextureFormat::SFLOAT_16_16_16_16, float4(0.25f));
    frame.output = make("RR output", TextureFormat::SFLOAT_16_16_16_16, float4(0), true);
    frame.depth = make("RR depth", TextureFormat::SFLOAT_32, float4(0.5f));
    frame.motion = make("RR motion", TextureFormat::SFLOAT_32_32, float4(0));
    frame.diffuse_albedo = make("RR diffuse", TextureFormat::SFLOAT_16_16_16_16, float4(0.5f));
    frame.specular_albedo = make("RR specular", TextureFormat::SFLOAT_16_16_16_16, float4(0.04f));
    frame.normal_roughness = make(
        "RR normal", TextureFormat::SFLOAT_16_16_16_16, float4(0, 0, 1, 0.5f));
    frame.specular_hit_distance = make("RR hit", TextureFormat::SFLOAT_32, float4(0));
    frame.color_before_transparency = make(
        "RR before", TextureFormat::SFLOAT_16_16_16_16, float4(0.25f));
    frame.transparency_radiance = make("RR layer", TextureFormat::SFLOAT_16_16_16_16, float4(0));
    frame.transparency_opacity = make("RR opacity", TextureFormat::SFLOAT_16_16_16_16, float4(0));
    frame.transparency_motion = make("RR layer motion", TextureFormat::SFLOAT_32_32, float4(0));
    frame.subsurface_guide = make("RR SSS", TextureFormat::SFLOAT_16, float4(0));
    frame.disocclusion_mask = make("RR disocclusion", TextureFormat::SFLOAT_16, float4(0));
    frame.bias_current_color_mask = make("RR current", TextureFormat::SFLOAT_16, float4(0));
    frame.view_to_clip = math::projection::perspective(-0.1f, 0.1f, -0.1f, 0.1f, 0.1f, 100.0f);
    frame.reset = true;
    frame.transparency_input = DLSSTransparencyInput::SeparateLayer;
    return textures.size() == 15;
  }

  Array<float4> read()
  {
    GPU_finish();
    Array<float4> result(int64_t(output_size.x) * output_size.y);
    float4 *pixels = static_cast<float4 *>(GPU_texture_read(frame.output, GPU_DATA_FLOAT, 0));
    if (pixels) {
      std::copy_n(pixels, result.size(), result.data());
      MEM_delete(pixels);
    }
    else {
      result.fill(float4(NAN));
    }
    return result;
  }
};

static void test_dlss_rr_optional_inputs_lifetime()
{
  RRFixture fixture;
  if (!fixture.init()) {
    GTEST_SKIP() << fixture.error;
  }
  for (int sample = 0; sample < 3; sample++) {
    /* Reuse every auxiliary resource between evaluations, including signed SSS. */
    const float4 sss = float4(sample == 1 ? -0.125f : 0.125f);
    GPU_texture_clear(fixture.frame.subsurface_guide, GPU_DATA_FLOAT, sss);
    if (sample == 2) {
      fixture.frame.bias_current_color_mask = fixture.frame.disocclusion_mask;
    }
    ASSERT_TRUE(fixture.dlss->evaluate(fixture.frame)) << fixture.dlss->error();
    fixture.frame.reset = false;
  }
  GPU_finish();
  ASSERT_FALSE(fixture.dlss->failed()) << fixture.dlss->error();
  ASSERT_TRUE(fixture.dlss->evaluate(fixture.frame));
  fixture.dlss.reset();
  for (const float4 &pixel : fixture.read()) {
    EXPECT_TRUE(std::isfinite(pixel.x) && std::isfinite(pixel.y) && std::isfinite(pixel.z));
  }
}
GPU_VULKAN_TEST(dlss_rr_optional_inputs_lifetime)

static void test_dlss_sr_bias_lifetime()
{
  RRFixture fixture;
  if (!fixture.init(DLSSMode::SuperResolution)) {
    GTEST_SKIP() << fixture.error;
  }
  for (int i = 0; i < 3; i++) {
    const float4 value(i == 1 ? 1.0f : 0.0f);
    GPU_texture_clear(fixture.frame.bias_current_color_mask, GPU_DATA_FLOAT, value);
    ASSERT_TRUE(fixture.dlss->evaluate(fixture.frame));
    fixture.frame.reset = false;
  }
  GPU_finish();
  ASSERT_FALSE(fixture.dlss->failed()) << fixture.dlss->error();
  /* Null optional inputs must clear the previous descriptor, including deferred recording. */
  fixture.frame.bias_current_color_mask = nullptr;
  ASSERT_TRUE(fixture.dlss->evaluate(fixture.frame));
  fixture.dlss.reset();
  for (const float4 &pixel : fixture.read()) {
    EXPECT_TRUE(std::isfinite(pixel.x) && std::isfinite(pixel.y) && std::isfinite(pixel.z));
  }
}
GPU_VULKAN_TEST(dlss_sr_bias_lifetime)

static void test_dlss_sr_bias_validation()
{
  RRFixture fixture;
  if (!fixture.init(DLSSMode::SuperResolution)) {
    GTEST_SKIP() << fixture.error;
  }
  fixture.frame.bias_current_color_mask = fixture.frame.output;
  EXPECT_FALSE(fixture.dlss->evaluate(fixture.frame));
  EXPECT_TRUE(fixture.dlss->failed());
}
GPU_VULKAN_TEST(dlss_sr_bias_validation)

static void test_dlss_rr_optional_inputs_validation()
{
  RRFixture fixture;
  if (!fixture.init()) {
    GTEST_SKIP() << fixture.error;
  }
  fixture.frame.transparency_opacity = nullptr;
  EXPECT_FALSE(fixture.dlss->evaluate(fixture.frame));
  EXPECT_TRUE(fixture.dlss->failed());

  RRFixture wrong_size;
  ASSERT_TRUE(wrong_size.init()) << wrong_size.error;
  wrong_size.frame.subsurface_guide = wrong_size.frame.output;
  EXPECT_FALSE(wrong_size.dlss->evaluate(wrong_size.frame));
  EXPECT_TRUE(wrong_size.dlss->failed());

  RRFixture double_count;
  ASSERT_TRUE(double_count.init()) << double_count.error;
  double_count.frame.transparency_input = DLSSTransparencyInput::CompositedColor;
  EXPECT_FALSE(double_count.dlss->evaluate(double_count.frame));

  RRFixture wrong_format;
  ASSERT_TRUE(wrong_format.init()) << wrong_format.error;
  wrong_format.frame.color_before_transparency = wrong_format.make(
      "Wrong guide format", TextureFormat::SFLOAT_32_32_32_32, float4(0));
  EXPECT_FALSE(wrong_format.dlss->evaluate(wrong_format.frame));
}
GPU_VULKAN_TEST(dlss_rr_optional_inputs_validation)

static void test_dlss_rr_composited_transparency()
{
  RRFixture fixture;
  if (!fixture.init()) {
    GTEST_SKIP() << fixture.error;
  }
  fixture.frame.transparency_input = DLSSTransparencyInput::CompositedColor;
  fixture.frame.transparency_radiance = nullptr;
  fixture.frame.transparency_opacity = nullptr;
  fixture.frame.transparency_motion = nullptr;
  /* A uniform colored layer has an analytic composite. Keep the guide independent: it must
   * not be added to the main color a second time. This tests radiometry, not moving-layer IQ. */
  const float4 background(0.2f, 0.4f, 0.6f, 1.0f);
  const float4 composite(0.38f, 0.18f, 0.4f, 1.0f);
  GPU_texture_clear(fixture.frame.color_before_transparency, GPU_DATA_FLOAT, background);
  GPU_texture_clear(fixture.frame.color, GPU_DATA_FLOAT, composite);
  for (int sample = 0; sample < 32; sample++) {
    fixture.frame.reset = sample == 0;
    ASSERT_TRUE(fixture.dlss->evaluate(fixture.frame)) << fixture.dlss->error();
  }
  const auto pixels = fixture.read();
  ASSERT_FALSE(fixture.dlss->failed()) << fixture.dlss->error();
  for (int y = 16; y < fixture.output_size.y - 16; y++) {
    for (int x = 16; x < fixture.output_size.x - 16; x++) {
      const float4 pixel = pixels[y * fixture.output_size.x + x];
      for (int c = 0; c < 3; c++) {
        EXPECT_NEAR(pixel[c], composite[c], 0.02f);
      }
    }
  }
}
GPU_VULKAN_TEST(dlss_rr_composited_transparency)

/* Opt-in semantic experiment, not a visual quality certificate. Compare a composited reference
 * with a decomposed input. Also corrupt individual auxiliary inputs to test their influence.
 * A zero delta does NOT demonstrate that a supplied input has the documented effect. */
static void test_dlss_rr_transparency_probe()
{
  const char *directory = std::getenv("BLENDER_DLSS_PROBE_DIR");
  if (!directory || !*directory) {
    GTEST_SKIP() << "Set BLENDER_DLSS_PROBE_DIR to run and export the RR transparency experiment";
  }
  std::filesystem::create_directories(directory);
  Array<float4> reference;
  std::ofstream metrics(std::filesystem::path(directory) / "transparency_probe.csv");
  metrics << "variant,width,height,max_rgb_delta,mean_rgb_delta\n";
  for (int variant = 0; variant < 8; variant++) {
    RRFixture fixture;
    ASSERT_TRUE(fixture.init()) << fixture.error;
    const int2 size = fixture.input_size;
    const bool rgb = variant >= 3 && variant <= 5;
    if (rgb) {
      /* Blender's Vulkan backend expands RGB16F/RGB32F to RGBA. R11G11B10F retains three
       * channels in the native view, letting this probe distinguish the two SDK contracts. */
      fixture.frame.transparency_radiance = fixture.make(
          "RGB layer", TextureFormat::UFLOAT_11_11_10, float4(0));
      fixture.frame.transparency_opacity = fixture.make(
          "RGB opacity", TextureFormat::UFLOAT_11_11_10, float4(0));
    }
    Array<float4> before(int64_t(size.x) * size.y), layer(before.size()), opacity(before.size());
    Array<float4> color(before.size());
    Array<float3> rgb_layer(before.size()), rgb_opacity(before.size());
    for (int sample = 0; sample < 32; sample++) {
      for (int y = 0; y < size.y; y++) {
        for (int x = 0; x < size.x; x++) {
          const int i = y * size.x + x;
          before[i] = ((x / 4 + y / 4) & 1) ? float4(0.05f, 0.1f, 0.8f, 1) :
                                              float4(0.8f, 0.1f, 0.05f, 1);
          const bool covered = x > size.x / 4 + sample / 4 && x < size.x * 3 / 4 + sample / 4 &&
                               y > size.y / 3 && y < size.y * 2 / 3;
          opacity[i] = covered ? float4(0.7f, 0.3f, 0.5f, 1) : float4(0);
          layer[i] = covered ? float4(0.28f, 0.06f, 0.1f, 0.5f) : float4(0);
          color[i] = before[i] * (float4(1) - opacity[i]) + layer[i];
          color[i].w = 1;
          if (variant == 2 || variant == 4) {
            /* Deliberately inconsistent RGB opacity, for an observable sensitivity test. */
            opacity[i] = float4(1) - opacity[i];
          }
          if (variant == 5) {
            layer[i] = float4(2, 0, 2, 1);
          }
          if (variant == 6) {
            layer[i].w = 1 - layer[i].w;
          }
          rgb_layer[i] = layer[i].xyz();
          rgb_opacity[i] = opacity[i].xyz();
        }
      }
      GPU_texture_update(fixture.frame.color,
                         GPU_DATA_FLOAT,
                         (variant == 0 || variant == 7) ? color.data() : before.data());
      GPU_texture_update(fixture.frame.color_before_transparency, GPU_DATA_FLOAT, before.data());
      GPU_texture_update(fixture.frame.transparency_radiance,
                         GPU_DATA_FLOAT,
                         rgb ? static_cast<const void *>(rgb_layer.data()) : layer.data());
      GPU_texture_update(fixture.frame.transparency_opacity,
                         GPU_DATA_FLOAT,
                         rgb ? static_cast<const void *>(rgb_opacity.data()) : opacity.data());
      DLSSFrame frame = fixture.frame;
      if (variant == 0) {
        frame.transparency_input = DLSSTransparencyInput::CompositedColor;
        frame.color_before_transparency = nullptr;
        frame.transparency_radiance = nullptr;
        frame.transparency_opacity = nullptr;
        frame.transparency_motion = nullptr;
      }
      frame.reset = sample == 0;
      ASSERT_TRUE(fixture.dlss->evaluate(frame)) << fixture.dlss->error();
    }
    Array<float4> output = fixture.read();
    ASSERT_FALSE(fixture.dlss->failed()) << fixture.dlss->error();
    if (variant == 0) {
      reference = output;
    }
    double maximum = 0, mean = 0;
    for (int64_t i : output.index_range()) {
      for (int channel = 0; channel < 3; channel++) {
        ASSERT_TRUE(std::isfinite(output[i][channel]));
        const double delta = std::abs(double(output[i][channel]) - reference[i][channel]);
        maximum = std::max(maximum, delta);
        mean += delta / (3 * output.size());
      }
    }
    metrics << variant << ',' << fixture.output_size.x << ',' << fixture.output_size.y << ','
            << maximum << ',' << mean << '\n';
    const auto path = std::filesystem::path(directory) /
                      ("variant_" + std::to_string(variant) + ".f32");
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float4));
  }
}
GPU_VULKAN_TEST(dlss_rr_transparency_probe)

}  // namespace blender::gpu::tests
