# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Run inside Blender --debug-gpu: verify the SR opacity threshold using real guide readback."""
import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np

output = Path(sys.argv[sys.argv.index('--output') + 1])
output.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = scene.render.resolution_y = 96
scene.render.resolution_percentage = 100
scene.eevee.dlss_render_quality = 'QUALITY'
scene.eevee.taa_render_samples = 32
bpy.ops.object.camera_add(location=(0, 0, 6))
scene.camera = bpy.context.object
scene.camera.data.type = 'ORTHO'
scene.camera.data.ortho_scale = 6
scene.camera.data.clip_start = 0.1
scene.camera.data.clip_end = 100

def plane(name, z, size):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, z))
    obj = bpy.context.object
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    obj.data.materials.append(mat)
    shader = mat.node_tree.nodes['Principled BSDF']
    shader.inputs['Emission Color'].default_value = (0.2, 0.5, 0.8, 1)
    shader.inputs['Emission Strength'].default_value = 1
    return obj, mat, shader

plane('Background', 0, 5)
foreground, mat, shader = plane('Foreground', 2, 2)
mat.surface_render_method = 'BLENDED'
records = []
for alpha in (0.0, 0.4, 0.5, 0.6, 1.0):
    directory = output / str(alpha)
    os.environ['BLENDER_DLSS_DUMP_DIR'] = str(directory)
    shader.inputs['Alpha'].default_value = alpha
    bpy.ops.render.render()
    w, h, channels = map(int, (directory / 'depth.txt').read_text().split())
    depth = np.fromfile(directory / 'depth.f32', dtype=np.float32).reshape(h, w)
    assert np.isfinite(depth).all()
    value = float(depth[h // 2, w // 2])
    expected = ((6 - (2 if alpha >= 0.5 else 0)) - 0.1) / (100 - 0.1)
    assert abs(value - expected) < 1e-5, (alpha, value, expected)
    motion = np.fromfile(directory / 'motion.f32', dtype=np.float32)
    assert np.isfinite(motion).all() and np.max(abs(motion)) == 0
    rgba = np.fromfile(directory / 'rgba.f32', dtype=np.float32).reshape(96, 96, 4)
    assert np.isfinite(rgba).all() and np.min(rgba[..., 3]) >= 0 and np.max(rgba[..., 3]) <= 1
    records.append(dict(alpha=alpha, center_depth=value, expected_depth=expected))
(output / 'result.json').write_text(json.dumps(records, indent=2))
