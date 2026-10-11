# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Positive screen reflection and hardware mode switching regression.

Run in a windowed Blender (a hidden window is sufficient), not --background:
  blender --factory-startup --gpu-backend vulkan --python-exit-code 1 --python this_file.py -- --output DIR
Also run with --gpu-backend opengl to exercise hardware capability fallback.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
import gpu
from mathutils import Vector
import numpy as np


def run(output, denoise):
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_options.resolution_scale = '1'
    scene.eevee.ray_tracing_options.use_denoise = denoise
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.03, 0.03, 0.03, 1)

    def material(name, color, metallic=0, emission=0):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Base Color'].default_value = (*color, 1)
        bsdf.inputs['Metallic'].default_value = metallic
        bsdf.inputs['Roughness'].default_value = 0
        bsdf.inputs['Emission Color'].default_value = (*color, 1)
        bsdf.inputs['Emission Strength'].default_value = emission
        return mat

    bpy.ops.mesh.primitive_plane_add(size=30)
    bpy.context.object.data.materials.append(material('Mirror', (0.8, 0.8, 0.8), metallic=1))
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 2), rotation=(math.pi / 2, 0, 0))
    emitter = bpy.context.object
    emitter.scale = (1.5, 0.7, 1)
    emitter.data.materials.append(material('Red Source', (1, 0.005, 0.005), emission=4))
    bpy.ops.object.camera_add(location=(0, -6, 1))
    camera = bpy.context.object
    camera.data.lens = 32
    camera.rotation_euler = (Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera
    bpy.context.view_layer.update()
    size = 256
    area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
    space = area.spaces.active
    region = next(region for region in area.regions if region.type == 'WINDOW')
    space.shading.type = 'RENDERED'
    space.shading.use_scene_world_render = True
    space.shading.use_scene_lights_render = True
    space.overlay.show_overlays = False
    projection = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=size, y=size)
    viewport = gpu.types.GPUOffScreen(size, size, format='RGBA16F')
    images = {}
    metrics = {'backend': gpu.platform.backend_type_get(), 'denoise': denoise,
               'ray_query': gpu.capabilities.ray_query_support_get()}

    def draw(name, method):
        scene.eevee.ray_tracing_method = method
        bpy.context.view_layer.update()
        with bpy.context.temp_override(area=area, region=region):
            for _ in range(64):
                viewport.draw_view3d(scene, bpy.context.view_layer, space, region,
                                     camera.matrix_world.inverted(), projection,
                                     do_color_management=False, draw_background=True)
            with viewport.bind():
                pixels = np.array(gpu.state.active_framebuffer_get().read_color(
                    0, 0, size, size, 4, 0, 'FLOAT')).reshape(size, size, 4)
        images[name] = pixels
        # GPU readback starts at the bottom. This region contains only the mirror, while
        # the directly visible source is in the upper half and must not satisfy the test.
        reflection = pixels[:size // 2, :, :3]
        excess = np.maximum(reflection[:, :, 0] - reflection[:, :, 1], 0)
        metrics[name] = {'red_excess': float(excess.mean()),
                         'red_pixels': int((excess > 0.5).sum()),
                         'finite': bool(np.isfinite(pixels).all())}

    try:
        draw('screen_initial', 'SCREEN')
        draw('probe', 'PROBE')
        draw('screen_after_probe', 'SCREEN')
        draw('hardware', 'HARDWARE')
        draw('screen_after_hardware', 'SCREEN')
        # Remove the reflected object to ensure old hardware/screen radiance does not linger.
        bpy.data.objects.remove(emitter, do_unlink=True)
        draw('screen_deleted_source', 'SCREEN')
        checks = {'finite': all(value['finite'] for value in metrics.values() if isinstance(value, dict)),
                  'probe_has_no_source_reflection': metrics['probe']['red_excess'] < 0.005,
                  'deleted_source_has_no_reflection': metrics['screen_deleted_source']['red_excess'] < 0.005}
        for name in ('screen_initial', 'screen_after_probe', 'hardware', 'screen_after_hardware'):
            checks[name + '_has_reflection'] = metrics[name]['red_pixels'] > 100 and metrics[name]['red_excess'] > 0.05
        for name in ('screen_after_probe', 'screen_after_hardware'):
            checks[name + '_matches_initial'] = float(np.abs(
                images[name] - images['screen_initial']).mean()) < 0.02
        if not metrics['ray_query']:
            checks['hardware_fallback_matches_screen'] = float(np.abs(
                images['hardware'] - images['screen_initial']).mean()) < 0.02
        metrics['checks'] = checks
    finally:
        viewport.free()
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        # Export after rendering so Image datablock creation cannot invalidate temporal history.
        for name, pixels in images.items():
            image = bpy.data.images.new(name, width=size, height=size, alpha=True, float_buffer=True)
            image.pixels.foreach_set(pixels.ravel())
            image.save_render(str(output / (name + '.png')), scene=scene)
            bpy.data.images.remove(image)
    print('RAYTRACING_MODES', json.dumps(checks), flush=True)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError('Ray tracing mode regression: ' + ', '.join(failed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--denoise', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run(args.output, args.denoise)

    def quit_after_startup():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_startup, first_interval=0.1)
