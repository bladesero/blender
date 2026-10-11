# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Windowed viewport SR smoke/capture test (also run on OpenGL for native fallback).

blender --factory-startup --gpu-backend vulkan --debug-gpu --python this_file.py -- --output DIR

The caller must check the GPU log for successful NGX evaluations and validation errors.
Images alone cannot distinguish native fallback from DLSS. This deliberately uses real
viewport redraws: GPUOffScreen.draw_view3d is an image-render path, outside this feature.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import bpy
import gpu
from mathutils import Quaternion, Vector
import numpy as np


def main(output, hardware_raytracing=False, rr=False):
    output.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.view.show_splash = False
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.film_transparent = False
    scene.eevee.taa_samples = 32
    scene.eevee.use_raytracing = hardware_raytracing
    scene.eevee.dlss_rr = rr
    if hardware_raytracing:
        scene.eevee.ray_tracing_method = 'HARDWARE'
    scene.world.color = (0.15, 0.15, 0.15)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_plane_add(size=20)
    mat = bpy.data.materials.new('SR checker')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    checker = nodes.new('ShaderNodeTexChecker')
    checker.inputs['Scale'].default_value = 30
    if not rr:
        mat.node_tree.links.new(checker.outputs['Color'], nodes['Principled BSDF'].inputs['Base Color'])
    bpy.context.object.data.materials.append(mat)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=(0, 0, 1))
    sphere = bpy.context.object
    if rr:
        metal = mat.copy()
        metal.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.9
        metal.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.18
        sphere.data.materials.append(metal)
    bpy.ops.object.shade_smooth()
    for x in range(-4, 5):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x * 0.6, 2, 1.5))
        bpy.context.object.scale = (0.035, 0.08, 1.5)
        if rr:
            bpy.context.object.data.materials.append(mat)
    bpy.ops.object.light_add(type='AREA', location=(1, -2, 6))
    bpy.context.object.data.energy = 1800
    bpy.context.object.data.shape = 'DISK'
    bpy.context.object.data.size = 4
    area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
    space = area.spaces.active
    region = next(r for r in area.regions if r.type == 'WINDOW')
    space.shading.type = 'RENDERED'
    space.shading.use_scene_world_render = True
    space.shading.use_scene_lights_render = True
    space.shading.use_compositor = 'DISABLED'
    space.overlay.show_overlays = False
    space.show_region_toolbar = False
    rv = space.region_3d
    rv.view_location = (0, 0, 1)
    rv.view_distance = 9
    rv.view_rotation = Vector((5, -8, 5)).to_track_quat('Z', 'Y')
    rv.view_perspective = 'PERSP'
    state = {'tick': 0, 'capture': None, 'captures': {}, 'failed': False}

    def capture():
        name = state['capture']
        if not name:
            return
        try:
            path = output / (name + '.png')
            # POST_PIXEL reads only the overlay framebuffer. Read the composited window
            # through Blender's screenshot operator, after the real viewport redraw.
            with bpy.context.temp_override(area=area, region=region):
                result = bpy.ops.screen.screenshot_area(filepath=str(path))
            assert result == {'FINISHED'}, name + ': screenshot failed'
            image = bpy.data.images.load(str(path), check_existing=False)
            width, height = image.size
            array = np.array(image.pixels[:]).reshape(height, width, 4)
            rgb = array[:, :, :3]
            assert np.isfinite(rgb).all(), name + ': non-finite framebuffer'
            assert float(rgb.std()) > 0.01, name + ': empty or flat framebuffer'
            bpy.data.images.remove(image)
            state['captures'][name] = {'size': [width, height], 'mean': float(rgb.mean()),
                                       'std': float(rgb.std()), 'mode': scene.eevee.dlss_quality}
            print('DLSS_TEST_CAPTURE', name, state['captures'][name], flush=True)
        except Exception:
            traceback.print_exc()
            state['failed'] = True
        state['capture'] = None

    def tick():
        try:
            i = state['tick']
            if i == 0:
                scene.eevee.dlss_quality = 'OFF'
            elif i == 12:
                state['capture'] = '00_native'
            elif i == 14:
                scene.eevee.dlss_quality = 'QUALITY'
            elif i == 26:
                state['capture'] = '01_quality'
            elif 28 <= i < 40:
                rv.view_rotation = Quaternion((0, 0, 1), 0.035) @ rv.view_rotation
                if i == 36:
                    state['capture'] = '02_navigation'
            elif i == 42:
                state['capture'] = '03_settled'
            elif i == 44:
                rv.view_location = (1, 0, 1)
                rv.view_rotation = Quaternion((0, 0, 1), math.pi) @ rv.view_rotation
            elif i == 48:
                state['capture'] = '04_cut'
            elif i == 50:
                scene.eevee.dlss_quality = 'BALANCED'
            elif i == 56:
                state['capture'] = '05_balanced'
            elif i == 58:
                scene.eevee.dlss_quality = 'PERFORMANCE'
            elif i == 64:
                state['capture'] = '06_performance'
            elif 66 <= i < 74:
                scene.frame_set(i - 65)
                sphere.location.x = (i - 66) * 0.12
            elif i == 74:
                state['capture'] = '07_motion'
            elif i == 76:
                space.overlay.show_overlays = True
            elif i == 80:
                state['capture'] = '08_overlays'
            elif i == 82:
                scene.render.film_transparent = True
            elif i == 88:
                state['capture'] = '09_native_fallback'
            elif i == 90:
                scene.render.film_transparent = False
                scene.eevee.dlss_quality = 'QUALITY'
            elif i == 96:
                state['capture'] = '10_reenabled'
            elif i == 98:
                with bpy.context.temp_override(area=area, region=region):
                    bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
                for split_area in bpy.context.screen.areas:
                    if split_area.type == 'VIEW_3D':
                        split_space = split_area.spaces.active
                        split_space.shading.type = 'RENDERED'
                        split_space.shading.use_scene_world_render = True
                        split_space.shading.use_scene_lights_render = True
                        split_space.shading.use_compositor = 'DISABLED'
                        split_area.tag_redraw()
            elif i == 104:
                state['capture'] = '11_split_viewports'
            elif i == 106:
                scene.eevee.dlss_quality = 'OFF'
            elif i == 110:
                state['capture'] = '12_disabled'
            elif i == 111:
                # Exercise destruction with live feature handles in both viewports.
                scene.eevee.dlss_quality = 'QUALITY'
            elif rr and i == 113:
                scene.eevee.dlss_rr = False
            elif rr and i == 119:
                state['capture'] = '13_rr_to_sr'
            elif rr and i == 121:
                scene.eevee.dlss_rr = True
            elif rr and i == 127:
                state['capture'] = '14_sr_to_rr'
            elif i >= (129 if rr else 113):
                result = {'backend': gpu.platform.backend_type_get(),
                          'hardware_raytracing': hardware_raytracing,
                          'rr': rr,
                          'captures': state['captures'],
                          'passed': not state['failed'] and len(state['captures']) == (15 if rr else 13)}
                (output / 'result.json').write_text(json.dumps(result, indent=2))
                print('DLSS_TEST_RESULT', json.dumps(result), flush=True)
                bpy.ops.wm.quit_blender()
                return None
            capture()
            state['tick'] += 1
            area.tag_redraw()
            return 0.5
        except Exception:
            traceback.print_exc()
            (output / 'failure.txt').write_text(traceback.format_exc())
            bpy.ops.wm.quit_blender()
            return None

    bpy.app.timers.register(tick, first_interval=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--hardware-raytracing', action='store_true')
    parser.add_argument('--rr', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    main(args.output, args.hardware_raytracing or args.rr, args.rr)
