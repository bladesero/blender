# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Inspect BLENDER_DLSS_DUMP_DIR float buffers; run in Python with numpy and Pillow.

This checks signal contracts, not perceptual reconstruction quality. Use the exported
images to inspect background, reflectance, normals, hit distance and disocclusions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main(directory, static):
    stats, buffers = {}, {}
    for meta in directory.glob('*.txt'):
        width, height, channels = map(int, meta.read_text().split())
        pixels = np.fromfile(meta.with_suffix('.f32'), dtype=np.float32).reshape(height, width, channels)
        assert np.isfinite(pixels).all(), meta.name
        name = meta.stem
        buffers[name] = pixels
        stats[name] = dict(size=[width, height, channels], min=float(pixels.min()), max=float(pixels.max()),
                           mean=float(pixels.mean()), nonzero=int(np.count_nonzero(pixels)))
        rgb = pixels[:, :, :3]
        if name == 'normal_roughness':
            rgb = rgb*0.5+0.5
            assert np.max(np.abs(np.linalg.norm(pixels[:, :, :3], axis=2)-1)) < 0.004
            assert pixels[:, :, 3].min() >= 0 and pixels[:, :, 3].max() <= 1
            Image.fromarray((pixels[:, :, 3]*255).astype(np.uint8)).save(directory / 'roughness.png')
        elif name == 'depth':
            assert pixels.min() >= 0 and pixels.max() <= 1
            rgb = (rgb-rgb.min())/max(float(rgb.max()-rgb.min()), 1e-6)
        elif name == 'motion':
            if static:
                assert np.count_nonzero(pixels) == 0
            rgb = np.concatenate((rgb/32+0.5, np.full_like(rgb[:, :, :1], 0.5)), axis=2)
        elif name == 'specular_hit':
            assert pixels.min() >= 0
            rgb = np.log1p(rgb)/max(float(np.log1p(rgb).max()), 1e-6)
        else:
            rgb = np.maximum(rgb, 0)/(1+np.maximum(rgb, 0))
            rgb = rgb**(1/2.2)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(directory / (name+'.png'))
    assert {'color', 'depth', 'motion', 'output'} <= buffers.keys()
    input_shape = buffers['color'].shape[:2]
    for name, pixels in buffers.items():
        if name != 'output':
            assert pixels.shape[:2] == input_shape, name
    if 'specular' in buffers:
        assert buffers['specular'][:, :, :3].max() > 0, 'Missing reflection guide'
        assert buffers['diffuse'][:, :, :3].max() > 0, 'Missing diffuse guide'
        assert buffers['specular_hit'].max() > 0, 'No geometric reflection hits in the fixture'
    (directory / 'guide_validation.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--static', action='store_true')
    args = parser.parse_args()
    main(args.directory, args.static)
