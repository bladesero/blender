# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Windowed render-job test. Default: cancel/recover with --debug-gpu.

Append --complete to exercise the normal F12 job, and --sr to use SR instead of RR.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from eevee_dlss_render import make_scene

arguments = sys.argv[sys.argv.index('--')+1:]
output = Path(arguments[arguments.index('--output')+1] if '--output' in arguments else arguments[0]).resolve()
output.mkdir(parents=True, exist_ok=True)
scene = make_scene()
test_cancel = '--complete' not in sys.argv
scene.eevee.dlss_render_quality = 'QUALITY'
scene.eevee.dlss_render_rr = '--sr' not in sys.argv
scene.eevee.taa_render_samples = 64 if test_cancel else 32
if '--motion-blur' in sys.argv:
    scene.render.use_motion_blur = True
    scene.eevee.motion_blur_steps = 2
if '--blended' in sys.argv:
    scene.eevee.dlss_render_rr = False
    scene.eevee.use_overscan = True
    scene.eevee.overscan_size = 3
    material = bpy.data.materials['PBR0']
    material.surface_render_method = 'BLENDED'
    material.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = 0.4
if '--normal' in sys.argv:
    tree = bpy.data.materials['PBR0'].node_tree
    normal = tree.nodes.new('ShaderNodeNormalMap')
    normal.inputs['Color'].default_value = (0.7, 0.5, 0.95, 1)
    tree.links.new(normal.outputs['Normal'], tree.nodes['Principled BSDF'].inputs['Normal'])
if '--mixed' in sys.argv:
    bpy.data.materials['PBR0'].node_tree.nodes['Principled BSDF'].inputs['Subsurface Weight'].default_value = 0.8
    tree = bpy.data.materials['PBR1'].node_tree
    glass = tree.nodes.new('ShaderNodeBsdfGlass')
    tree.links.new(glass.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
    material = bpy.data.materials['PBR3']
    material.surface_render_method = 'BLENDED'
    material.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = 0.4
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(output / 'cancelled.png')
bpy.context.preferences.view.render_display_type = 'NONE'
bpy.context.preferences.view.use_save_prompt = False
state = {'cancelled': False, 'completed': False, 'start': 0}
initial_time = (scene.frame_current, scene.frame_subframe)
if test_cancel:
    os.environ['BLENDER_DLSS_TEST_CANCEL_AT'] = '3'


def cancelled(_scene):
    state['cancelled'] = True


bpy.app.handlers.render_cancel.append(cancelled)


def completed(_scene):
    state['completed'] = True


bpy.app.handlers.render_complete.append(completed)


def poll():
    try:
        elapsed = time.monotonic()-state['start']
        if bpy.app.is_job_running('RENDER'):
            assert elapsed < 45, 'Render did not respond to cancellation'
            return 0.25
        if test_cancel:
            assert state['cancelled'], state
            assert (scene.frame_current, scene.frame_subframe) == initial_time
            assert not (output / 'cancelled.png').exists(), 'Cancelled render was saved'
            os.environ.pop('BLENDER_DLSS_TEST_CANCEL_AT', None)
            scene.eevee.taa_render_samples = 8
            scene.frame_set(2)
            scene.render.filepath = str(output / 'recovered.png')
            bpy.ops.render.render(write_still=True)
            assert (output / 'recovered.png').exists()
        else:
            assert state['completed'] and not state['cancelled'], state
            image = bpy.data.images.load(scene.render.filepath)
            assert tuple(image.size) == (640, 360)
            assert max(image.pixels[:]) > 0.1
        (output / 'result.json').write_text(json.dumps(dict(cancelled=state['cancelled'],
                                                          completed=state['completed'],
                                                          recovered=test_cancel)))
    except Exception:
        (output / 'failure.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    bpy.ops.wm.quit_blender()


def start():
    state['start'] = time.monotonic()
    bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)
    bpy.app.timers.register(poll, first_interval=0.5)


bpy.app.timers.register(start, first_interval=1.0)
