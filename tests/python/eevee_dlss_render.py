# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Real GPU reconstruction acceptance runner. API success must also be checked in GPU logs.

blender -b --factory-startup --gpu-backend vulkan --debug-gpu --python this_file.py -- --output DIR
Use --mode RR and --sequence 1,2,1 for reconstruction and independent-frame checks.
Without -b the same operator exercises the windowed final-render path.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector
import numpy as np


def make_scene():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 360
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_method = 'HARDWARE'
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.15, 0.2, 0.3, 1)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    def material(name, color, metal, rough):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Base Color'].default_value = (*color, 1)
        bsdf.inputs['Metallic'].default_value = metal
        bsdf.inputs['Roughness'].default_value = rough
        return mat

    ground = material('Ground', (0.3, 0.3, 0.3), 0, 0.35)
    bpy.ops.mesh.primitive_plane_add(size=200)
    bpy.context.object.data.materials.append(ground)
    spheres = []
    for i, (color, metal, rough) in enumerate([
            ((0.8, 0.035, 0.02), 0, 0.65),
            ((0.8, 0.45, 0.1), 1, 0.22),
            ((0.8, 0.8, 0.8), 1, 0.02),
            ((0.02, 0.12, 0.8), 0, 0.2)]):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=((i-1.5)*1.6, 0, 0.75), radius=0.75)
        obj = bpy.context.object
        obj.data.materials.append(material('PBR' + str(i), color, metal, rough))
        bpy.ops.object.shade_smooth()
        spheres.append(obj)
    for i in range(15):
        bpy.ops.mesh.primitive_cube_add(size=1, location=((i-7)*0.4, 2, 1.2))
        bpy.context.object.scale = (0.025, 0.05, 1.2)
        bpy.context.object.data.materials.append(ground)
    bpy.ops.object.light_add(type='AREA', location=(-3, -4, 6))
    bpy.context.object.data.energy = 1600
    bpy.context.object.data.size = 3
    bpy.ops.object.camera_add(location=(4, -10, 5))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, 0.7))-camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = 45
    scene.camera = camera
    spheres[0].keyframe_insert('location', frame=1)
    spheres[0].location.z += 0.5
    spheres[0].keyframe_insert('location', frame=2)
    return scene


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene if args.use_existing_scene else make_scene()
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.eevee.taa_render_samples = args.samples
    scene.eevee.dlss_render_quality = 'OFF' if args.mode == 'NATIVE' else args.quality
    scene.eevee.dlss_render_rr = args.mode == 'RR'
    if args.fallback == 'TRANSPARENT':
        scene.render.film_transparent = True
    elif args.fallback == 'DEPTH':
        bpy.context.view_layer.use_pass_z = True
    elif args.fallback == 'DOF':
        scene.camera.data.dof.use_dof = True
    elif args.fallback == 'BORDER':
        scene.render.use_border = True
    elif args.fallback == 'AOV':
        bpy.context.view_layer.aovs.add().name = 'Unsupported AOV'
    elif args.fallback == 'MOTION_BLUR':
        scene.render.use_motion_blur = True
    elif args.fallback == 'CRYPTOMATTE':
        bpy.context.view_layer.use_pass_cryptomatte_object = True
    elif args.fallback == 'TRANSMISSION':
        bpy.data.materials['PBR0'].node_tree.nodes['Principled BSDF'].inputs['Transmission Weight'].default_value = 0.8
    elif args.fallback == 'NORMAL_MAP':
        tree = bpy.data.materials['PBR0'].node_tree
        normal = tree.nodes.new('ShaderNodeNormalMap')
        tree.links.new(normal.outputs['Normal'], tree.nodes['Principled BSDF'].inputs['Normal'])
    elif args.fallback == 'VOLUME':
        tree = scene.world.node_tree
        volume = tree.nodes.new('ShaderNodeVolumePrincipled')
        volume.inputs['Density'].default_value = 0.01
        tree.links.new(volume.outputs['Volume'], tree.nodes['World Output'].inputs['Volume'])
    elif args.fallback == 'POINTCLOUD':
        cloud = bpy.data.objects.new('Unsupported points', bpy.data.pointclouds.new('Points'))
        bpy.context.collection.objects.link(cloud)
        group = bpy.data.node_groups.new('Point fixture', 'GeometryNodeTree')
        group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        points = group.nodes.new('GeometryNodePoints')
        points.inputs['Count'].default_value = 4
        points.inputs['Position'].default_value = (0, 0, 2)
        points.inputs['Radius'].default_value = 0.2
        group_output = group.nodes.new('NodeGroupOutput')
        group.links.new(points.outputs['Points'], group_output.inputs['Geometry'])
        cloud.modifiers.new('Generate points', 'NODES').node_group = group
    elif args.fallback == 'MULTIVIEW':
        scene.render.use_multiview = True
    if args.composite:
        tree = bpy.data.node_groups.new('Combined postprocessing', 'CompositorNodeTree')
        scene.compositing_node_group = tree
        tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        layers = tree.nodes.new('CompositorNodeRLayers')
        exposure = tree.nodes.new('CompositorNodeExposure')
        exposure.inputs['Exposure'].default_value = 1.0
        output_node = tree.nodes.new('NodeGroupOutput')
        tree.links.new(layers.outputs['Image'], exposure.inputs['Image'])
        tree.links.new(exposure.outputs['Image'], output_node.inputs['Image'])
    records, frames = [], {}
    if args.save_fixture:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_fixture))
    for index, frame in enumerate(map(int, args.sequence.split(','))):
        scene.frame_set(frame)
        name = '%02d_frame%04d' % (index, frame)
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.filepath = str(args.output / (name + '.exr'))
        print('DLSS_RENDER_BEGIN', name, args.mode, flush=True)
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        elapsed = time.perf_counter()-start
        image_path = scene.render.filepath
        if args.fallback == 'MULTIVIEW':
            image_path = str(args.output / (name + '_L.exr'))
            assert (args.output / (name + '_R.exr')).exists()
        image = bpy.data.images.load(image_path, check_existing=False)
        width, height = image.size
        pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        assert (width, height) == (args.width, args.height), image.size[:]
        assert np.isfinite(pixels).all(), 'Non-finite pixels'
        assert pixels[:, :, :3].std() > 0.01, 'Blank output'
        if args.fallback != 'TRANSPARENT':
            assert np.max(np.abs(pixels[:, :, 3]-1)) < 1e-4, 'Invalid opaque alpha'
        record = dict(name=name, frame=frame, seconds=elapsed, size=[width, height], mean=float(pixels.mean()))
        if frame in frames:
            record['repeat_max_error'] = float(np.max(np.abs(pixels-frames[frame])))
            assert np.allclose(pixels, frames[frame], rtol=0.002, atol=0.002), record
        frames[frame] = pixels
        np.save(args.output / (name + '.npy'), pixels)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '8'
        if args.fallback == 'MULTIVIEW':
            # The native render writes individual stereo EXRs; export a mono preview of the left view.
            scene.render.use_multiview = False
            image.save_render(str(args.output / (name + '.png')), scene=scene)
            scene.render.use_multiview = True
        else:
            bpy.data.images['Render Result'].save_render(str(args.output / (name + '.png')), scene=scene)
        bpy.data.images.remove(image)
        records.append(record)
        print('DLSS_RENDER_END', json.dumps(record), flush=True)
    (args.output / 'results.json').write_text(json.dumps(dict(mode=args.mode, fallback=args.fallback, records=records), indent=2))
    if not bpy.app.background:
        # Startup scripts run before the first event loop; defer shutdown until it exists.
        def quit_after_startup():
            bpy.ops.wm.quit_blender()
        bpy.app.timers.register(quit_after_startup, first_interval=1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['SR', 'RR', 'NATIVE'], default='SR')
    parser.add_argument('--quality', choices=['QUALITY', 'BALANCED', 'PERFORMANCE'], default='QUALITY')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--sequence', default='1')
    parser.add_argument('--use-existing-scene', action='store_true')
    parser.add_argument('--save-fixture', type=Path)
    parser.add_argument('--composite', action='store_true')
    parser.add_argument('--fallback', choices=['NONE', 'TRANSPARENT', 'DEPTH', 'DOF', 'BORDER',
                        'AOV', 'CRYPTOMATTE', 'MOTION_BLUR', 'TRANSMISSION', 'NORMAL_MAP', 'VOLUME',
                        'POINTCLOUD', 'MULTIVIEW'], default='NONE')
    main(parser.parse_args(sys.argv[sys.argv.index('--')+1:]))
