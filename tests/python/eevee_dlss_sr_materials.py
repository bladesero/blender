# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Material-independent SR fixtures and real-GPU acceptance runner.

Run with system Python and --blender PATH --output DIR to run the matrix and
validate NGX logs, or inside Blender with --case NAME. --viewport runs a 300
frame interactive sequence; --f12 uses the asynchronous render job.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

CASES = ('PBR', 'SHADER_RGB', 'EMISSION', 'NORMAL', 'SSS', 'TRANSLUCENT',
         'DITHERED', 'BLENDED', 'COLORED', 'GLASS', 'HOLDOUT', 'VOLUME',
         'SURFACE_VOLUME', 'ALPHA', 'DOF', 'MOTION_BLUR', 'COMBINED', 'MIXED')


def fixture(args):
    import bpy
    spec = importlib.util.spec_from_file_location('dlss_fixture', Path(__file__).with_name('eevee_dlss_render.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene = module.make_scene()
    scene.eevee.ray_tracing_method = 'SCREEN'
    scene.eevee.dlss_render_quality = args.quality if args.mode == 'SR' else 'OFF'
    scene.eevee.dlss_quality = scene.eevee.dlss_render_quality
    scene.eevee.dlss_render_rr = scene.eevee.dlss_rr = False
    scene.eevee.taa_render_samples = args.samples
    scene.eevee.taa_samples = 32
    if args.dof_jitter:
        scene.eevee.use_bokeh_jittered = True
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.image_settings.media_type = 'IMAGE'
    if args.orthographic:
        scene.camera.data.type = 'ORTHO'
        scene.camera.data.ortho_scale = 10
    subjects = [obj for obj in scene.objects if obj.type == 'MESH' and obj.data.materials
                and obj.data.materials[0].name.startswith('PBR')]
    if args.case in ('MIXED', 'COMBINED', 'COLORED'):
        for i in range(2):
            obj = subjects[i].copy()
            obj.data = obj.data.copy()
            obj.data.materials[0] = obj.data.materials[0].copy()
            scene.collection.objects.link(obj)
            obj.location.y += 1.0
            subjects.append(obj)

    def configure(mat, case):
        tree = mat.node_tree
        bsdf, out = tree.nodes['Principled BSDF'], tree.nodes['Material Output']
        if case == 'SHADER_RGB':
            rgb = tree.nodes.new('ShaderNodeShaderToRGB')
            emission = tree.nodes.new('ShaderNodeEmission')
            tree.links.new(bsdf.outputs[0], rgb.inputs[0])
            tree.links.new(rgb.outputs['Color'], emission.inputs['Color'])
            tree.links.new(emission.outputs[0], out.inputs['Surface'])
        elif case == 'EMISSION':
            bsdf.inputs['Emission Color'].default_value = (0.2, 0.6, 1, 1)
            bsdf.inputs['Emission Strength'].default_value = 3
        elif case == 'SSS':
            bsdf.inputs['Subsurface Weight'].default_value = 0.8
        elif case in ('NORMAL', 'PBR'):
            noise = tree.nodes.new('ShaderNodeTexNoise')
            coord = tree.nodes.new('ShaderNodeTexCoord')
            mapping = tree.nodes.new('ShaderNodeMapping')
            tree.links.new(coord.outputs['Generated'], mapping.inputs['Vector'])
            tree.links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
            tree.links.new(noise.outputs['Color'], bsdf.inputs['Base Color'])
            normal = tree.nodes.new('ShaderNodeNormalMap')
            normal.inputs['Color'].default_value = (0.65, 0.45, 0.95, 1)
            tree.links.new(normal.outputs[0], bsdf.inputs['Normal'])
            mapping.name = 'SR animated mapping'
        elif case in ('DITHERED', 'BLENDED', 'COLORED'):
            mat.surface_render_method = 'DITHERED' if case == 'DITHERED' else 'BLENDED'
            bsdf.inputs['Alpha'].default_value = 0.65
            if case == 'COLORED':
                transparent = tree.nodes.new('ShaderNodeBsdfTransparent')
                transparent.inputs['Color'].default_value = (0.8, 0.3, 0.1, 1)
                mix = tree.nodes.new('ShaderNodeMixShader')
                mix.inputs[0].default_value = 0.5
                tree.links.new(bsdf.outputs[0], mix.inputs[1])
                tree.links.new(transparent.outputs[0], mix.inputs[2])
                tree.links.new(mix.outputs[0], out.inputs['Surface'])
        elif case in ('GLASS', 'TRANSLUCENT', 'HOLDOUT'):
            node = tree.nodes.new({'GLASS': 'ShaderNodeBsdfGlass', 'TRANSLUCENT': 'ShaderNodeBsdfTranslucent',
                                   'HOLDOUT': 'ShaderNodeHoldout'}[case])
            tree.links.new(node.outputs[0], out.inputs['Surface'])
        if case == 'SURFACE_VOLUME':
            volume = tree.nodes.new('ShaderNodeVolumePrincipled')
            volume.inputs['Density'].default_value = 0.25
            tree.links.new(volume.outputs[0], out.inputs['Volume'])

    mixed = ('GLASS', 'COLORED', 'SSS', 'SHADER_RGB', 'NORMAL', 'DITHERED')
    for i, obj in enumerate(subjects):
        configure(obj.data.materials[0], mixed[i % len(mixed)] if args.case in ('MIXED', 'COMBINED') else args.case)
        # Include deformation, opposing motion and disocclusion with a static floor/background.
        obj.animation_data_clear()
        obj.keyframe_insert('location', frame=1)
        obj.location.x += (-1 if i % 2 else 1) * 1.0
        obj.keyframe_insert('location', frame=150)
        obj.location.x -= (-1 if i % 2 else 1) * 1.0
        obj.keyframe_insert('location', frame=300)
        obj.keyframe_insert('scale', frame=1)
        obj.scale.z *= 1.3
        obj.keyframe_insert('scale', frame=150)
    deform = subjects[-1]
    deform.shape_key_add(name='Basis')
    shape = deform.shape_key_add(name='SR deformation')
    for vertex in shape.data:
        vertex.co.x += 0.15 * math.sin(vertex.co.z * 5)
    for frame, value in ((1, 0), (150, 1), (300, 0)):
        shape.value = value
        shape.keyframe_insert('value', frame=frame)
    if args.case in ('VOLUME', 'MIXED'):
        tree = scene.world.node_tree
        volume = tree.nodes.new('ShaderNodeVolumePrincipled')
        volume.inputs['Density'].default_value = 0.012
        tree.links.new(volume.outputs[0], tree.nodes['World Output'].inputs['Volume'])
        volume.name = 'SR animated volume'
    scene.render.film_transparent = args.case in ('ALPHA', 'HOLDOUT', 'COMBINED')
    if scene.render.film_transparent:
        for obj in scene.objects:
            if obj.type == 'MESH' and len(obj.data.vertices) == 4:
                obj.scale = (0.025, 0.025, 0.025)
    if args.case == 'COMBINED':
        configure(subjects[0].data.materials[0], 'SURFACE_VOLUME')
    if args.case in ('DOF', 'COMBINED'):
        scene.camera.data.dof.use_dof = True
        scene.camera.data.dof.focus_distance = 10
        scene.camera.data.dof.aperture_fstop = 1.4
    if args.case in ('MOTION_BLUR', 'COMBINED'):
        scene.render.use_motion_blur = True
        scene.eevee.motion_blur_steps = 2
        scene.render.motion_blur_shutter = 0.75
    scene.frame_set(80)
    return scene


def animate(scene, frame):
    import bpy
    scene.frame_set(frame)
    for mat in bpy.data.materials:
        if mat.node_tree and (mapping := mat.node_tree.nodes.get('SR animated mapping')):
            mapping.inputs['Location'].default_value.x = frame / 75.0
    volume = scene.world.node_tree.nodes.get('SR animated volume')
    if volume:
        volume.inputs['Density'].default_value = 0.012 + 0.008 * math.sin(frame / 30.0)


def render(args):
    import bpy
    import numpy as np
    scene = fixture(args)
    animate(scene, scene.frame_current)
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / 'fixture.blend'))
    records = []
    previous_frames = {}

    def read_output(index, elapsed):
        image = bpy.data.images.load(scene.render.filepath, check_existing=False)
        assert tuple(image.size) == (args.width, args.height)
        pixels = np.array(image.pixels[:], dtype=np.float32).reshape(args.height, args.width, 4)
        assert np.isfinite(pixels).all()
        assert pixels[..., :3].std() > 0.005
        alpha = pixels[..., 3]
        if index in previous_frames:
            assert np.allclose(pixels, previous_frames[index], atol=0.002, rtol=0.002), 'Frame history leaked'
        previous_frames[index] = pixels.copy()
        assert alpha.min() >= 0 and alpha.max() <= 1
        if args.case in ('ALPHA', 'COMBINED'):
            assert alpha.min() < 0.05 and alpha.max() > 0.5
        elif args.case != 'HOLDOUT':
            assert np.max(abs(alpha - 1)) < 0.002
        np.save(args.output / f'{index:04d}.npy', pixels)
        bpy.data.images.remove(image)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '8'
        bpy.data.images['Render Result'].save_render(str(args.output / f'{index:04d}.png'), scene=scene)
        if args.case in ('ALPHA', 'HOLDOUT', 'COMBINED'):
            composite = bpy.data.images.new('SR alpha comparison', width=args.width, height=args.height, float_buffer=True)
            for name, background in (('black', 0.0), ('white', 1.0)):
                rgba = pixels.copy()
                rgba[..., :3] += background * (1.0 - rgba[..., 3:4])
                rgba[..., 3] = 1.0
                composite.pixels.foreach_set(rgba.reshape(-1))
                composite.save_render(str(args.output / f'{index:04d}_{name}.png'), scene=scene)
            bpy.data.images.remove(composite)
        records.append(dict(frame=index, seconds=elapsed, alpha_min=float(alpha.min()), alpha_max=float(alpha.max())))
        (args.output / 'result.json').write_text(json.dumps(records, indent=2))

    def setup_output(index):
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.filepath = str(args.output / f'{index:04d}.exr')

    if args.viewport:
        area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
        region = next(r for r in area.regions if r.type == 'WINDOW')
        space = area.spaces.active
        space.shading.type = 'RENDERED'
        space.shading.use_scene_world_render = space.shading.use_scene_lights_render = True
        space.shading.use_compositor = 'DISABLED'
        space.region_3d.view_perspective = 'CAMERA'
        space.overlay.show_overlays = False
        state = {'frame': 0, 'start': time.perf_counter()}

        def tick():
            try:
                frame = state['frame']
                if frame:
                    with bpy.context.temp_override(area=area, region=region):
                        bpy.ops.screen.screenshot_area(filepath=str(args.output / f'{frame:04d}.png'))
                if frame >= args.frames:
                    (args.output / 'result.json').write_text(json.dumps(dict(frames=frame, seconds=time.perf_counter()-state['start'])))
                    bpy.ops.wm.quit_blender()
                    return None
                animate(scene, frame + 1)
                scene.camera.location.x += 0.002 * math.sin(frame * 0.05)
                state['frame'] += 1
                area.tag_redraw()
                return 0.15
            except Exception:
                (args.output / 'failure.txt').write_text(traceback.format_exc())
                bpy.ops.wm.quit_blender()
                return None
        bpy.app.timers.register(tick, first_interval=3)
    elif args.f12:
        setup_output(80)
        state = {'start': 0, 'complete': False}
        bpy.app.handlers.render_complete.append(lambda _: state.update(complete=True))

        def poll():
            if bpy.app.is_job_running('RENDER'):
                return 0.2
            try:
                assert state['complete']
                read_output(80, time.perf_counter()-state['start'])
            except Exception:
                (args.output / 'failure.txt').write_text(traceback.format_exc())
            bpy.ops.wm.quit_blender()
        def start():
            state['start'] = time.perf_counter()
            bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)
            bpy.app.timers.register(poll, first_interval=0.5)
        bpy.app.timers.register(start, first_interval=2)
    else:
        for frame in map(int, args.sequence.split(',')):
            animate(scene, frame)
            setup_output(frame)
            start = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            assert scene.frame_current == frame and abs(scene.frame_subframe) < 1e-6
            read_output(frame, time.perf_counter()-start)


def run_suite(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.case == 'ALL' else (args.case,)
    records = []
    env = dict(os.environ, VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation', VK_LAYER_VALIDATE_SYNC='1')
    if args.profile:
        env['BLENDER_DLSS_PROFILE'] = '1'
    if args.inject_failure:
        env['BLENDER_DLSS_TEST_FAIL_AT'] = '0'
    for case in cases:
        output = args.output / case
        output.mkdir(exist_ok=True)
        if args.dump_inputs:
            env['BLENDER_DLSS_DUMP_DIR'] = str((output / 'inputs').resolve())
            env['BLENDER_DLSS_DUMP_SAMPLE'] = str(args.dump_sample)
        command = [str(args.blender)] + ([] if args.viewport or args.f12 else ['-b'])
        if args.viewport or args.f12:
            command += ['--window-geometry', '40', '40', '960', '720']
        command += ['--factory-startup', '--gpu-backend', 'vulkan', '--debug-gpu', '--python-exit-code', '1',
                    '--python', str(Path(__file__).resolve()), '--', '--case', case, '--output', str(output),
                    '--quality', args.quality, '--mode', args.mode, '--samples', str(args.samples),
                    '--width', str(args.width), '--height', str(args.height), '--sequence', args.sequence]
        if args.viewport:
            command += ['--viewport', '--frames', str(args.frames)]
        if args.f12:
            command += ['--f12']
        if args.orthographic:
            command += ['--orthographic']
        if args.dof_jitter:
            command += ['--dof-jitter']
        monitor = None
        with (output / 'gpu.log').open('w', encoding='utf-8') as log, (output / 'device_memory.csv').open('w') as memory:
            if args.profile:
                try:
                    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
                                                '--loop-ms=200'], stdout=memory, stderr=subprocess.DEVNULL,
                                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except FileNotFoundError:
                    pass
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=900)
            finally:
                if monitor:
                    monitor.terminate()
                    monitor.wait()
        log = (output / 'gpu.log').read_text(encoding='utf-8', errors='replace')
        calls = re.findall(r'DLSS SR: (\d+)x(\d+) -> (\d+)x(\d+), reset=\d+, NGX=0x00000001', log)
        checks = dict(exit_ok=result.returncode == 0, output=(output / 'result.json').exists(),
                      no_failure=not (output / 'failure.txt').exists(),
                      no_validation_error=not re.search(r'Validation Error|SYNC-HAZARD|Assertion failed|BLI_assert failed', log),
                      no_native_fallback='actual Native' not in log and 'Restarting with native' not in log,
                      no_output_transparency='DLSS Blended:' not in log)
        if args.inject_failure:
            checks.pop('no_native_fallback')
            checks['native_retry'] = 'Restarting with native EEVEE' in log
            checks['failed_output_discarded'] = 'discarding the reconstructed frame' in log
            checks['no_successful_ngx'] = not calls
        elif args.mode == 'SR':
            checks['ngx_executed'] = bool(calls)
            checks['lower_resolution'] = bool(calls) and all(int(a) < int(c) and int(b) < int(d) for a, b, c, d in calls)
            if case in ('MOTION_BLUR', 'COMBINED') and not args.viewport:
                checks['single_evaluation'] = len(calls) == len(args.sequence.split(','))
        else:
            checks['no_ngx'] = not calls
        record = dict(case=case, mode=args.mode, checks=checks, evaluations=len(calls), passed=all(checks.values()))
        profile = re.findall(r'DLSS GPU profile: mode=SR, evaluations=(\d+), total_ms=([\d.]+), steady_mean_ms=([\d.]+), SDK_vram_bytes=(\d+)', log)
        record['ngx_profiles'] = [dict(evaluations=int(n), total_ms=float(t), steady_mean_ms=float(s), sdk_vram_bytes=int(v))
                                  for n, t, s, v in profile]
        record['total_gpu_ms'] = None  # NGX timing does not measure the complete EEVEE pipeline.
        memory_samples = [int(v) for v in (output / 'device_memory.csv').read_text().splitlines() if v.strip().isdigit()]
        record['sampled_device_peak_mib'] = max(memory_samples) if memory_samples else None
        if args.reference and (args.reference / case).exists():
            import numpy as np
            comparisons = []
            for file in output.glob('*.npy'):
                reference = args.reference / case / file.name
                if not reference.exists():
                    continue
                actual, expected = np.load(file), np.load(reference)
                comparisons.append(dict(frame=file.stem, rgb_mae=float(np.abs(actual[..., :3]-expected[..., :3]).mean()),
                                        alpha_mae=float(np.abs(actual[..., 3]-expected[..., 3]).mean()),
                                        white_composite_mae=float(np.abs((actual[..., :3]+1-actual[..., 3:4])-
                                                                        (expected[..., :3]+1-expected[..., 3:4])).mean())))
                if args.inject_failure:
                    # Reused native lighting caches can change stochastic RGB slightly. Coverage
                    # must still match; do not hide resolution/time errors in an RGB-only metric.
                    checks['native_retry_alpha_matches'] = bool(np.allclose(actual[..., 3], expected[..., 3], atol=0.002, rtol=0))
                    checks['native_retry_rgb_mae'] = comparisons[-1]['rgb_mae'] < 0.001
            record['reference_comparison'] = comparisons
        record['passed'] = all(checks.values())
        records.append(record)
        print(json.dumps(record), flush=True)
        (args.output / 'summary.json').write_text(json.dumps(records, indent=2))
    if not all(r['passed'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES + ('ALL',), default='ALL')
    parser.add_argument('--mode', choices=('SR', 'NATIVE'), default='SR')
    parser.add_argument('--quality', choices=('QUALITY', 'BALANCED', 'PERFORMANCE'), default='QUALITY')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--sequence', default='80')
    parser.add_argument('--viewport', action='store_true')
    parser.add_argument('--f12', action='store_true')
    parser.add_argument('--orthographic', action='store_true')
    parser.add_argument('--dof-jitter', action='store_true')
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--dump-inputs', action='store_true')
    parser.add_argument('--dump-sample', type=int, default=1, help='Successful SR evaluation to dump (1 for shutter integration)')
    parser.add_argument('--inject-failure', action='store_true', help='Fail the first NGX evaluation and require a complete native retry')
    parser.add_argument('--reference', type=Path)
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    run_suite(args) if args.blender else render(args)
