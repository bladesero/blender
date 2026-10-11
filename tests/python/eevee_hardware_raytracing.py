# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Vulkan hardware RT acceptance scenes, using a persistent GPUOffScreen viewport.

Run in a windowed Blender (the window may be hidden), not --background:
  blender --factory-startup --gpu-backend vulkan --python-exit-code 1 --python this_file.py -- --output DIR
Images and metrics are written to DIR. No external scene assets are required.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

import bpy
import gpu
from mathutils import Vector
import numpy as np


def run(output, denoise=False, profile_render=False):
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_method = 'HARDWARE'
    scene.eevee.ray_tracing_options.resolution_scale = '1'
    scene.eevee.ray_tracing_options.use_denoise = denoise
    scene.render.resolution_x = scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.03, 0.03, 0.03, 1)

    def material(name, color, metallic=0, roughness=1, emission=0):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        node = mat.node_tree.nodes['Principled BSDF']
        node.inputs['Base Color'].default_value = (*color, 1)
        node.inputs['Metallic'].default_value = metallic
        node.inputs['Roughness'].default_value = roughness
        node.inputs['Emission Color'].default_value = (*color, 1)
        node.inputs['Emission Strength'].default_value = emission
        return mat

    bpy.ops.mesh.primitive_plane_add(size=30)
    floor = bpy.context.object
    floor.name = 'HardwareRTReceiver'
    receiver = material('Receiver', (0.8, 0.8, 0.8), metallic=1, roughness=0)
    floor.data.materials.append(receiver)
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 2), rotation=(math.pi / 2, 0, 0))
    emitter = bpy.context.object
    emitter.name = 'HardwareRTEmitter'
    emitter.scale = (1.5, 0.7, 1)
    source = material('Source', (1, 0.005, 0.005), emission=4)
    emitter.data.materials.append(source)
    bpy.ops.object.camera_add(location=(0, -6, 1))
    camera = bpy.context.object
    camera.data.lens = 60
    scene.camera = camera

    if profile_render:
        camera.rotation_euler = (Vector((0, 0, 0.5)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
        scene.render.resolution_x = scene.render.resolution_y = 512
        scene.eevee.taa_render_samples = 32
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(str(output / 'profile_render.png'), scene=scene)
        return

    area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
    space = area.spaces.active
    region = next(region for region in area.regions if region.type == 'WINDOW')
    space.shading.type = 'RENDERED'
    space.shading.use_scene_world_render = True
    space.shading.use_scene_lights_render = True
    space.overlay.show_overlays = False
    size = 256
    metrics = {'backend': gpu.platform.backend_type_get(), 'device': gpu.platform.renderer_get(),
               'ray_query': gpu.capabilities.ray_query_support_get()}
    timings = {}
    images = []
    viewport = gpu.types.GPUOffScreen(size, size, format='RGBA16F')

    def aim(target):
        camera.rotation_euler = (Vector(target) - camera.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()

    def draw(target, frames=24):
        aim(target)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        projection = camera.calc_matrix_camera(depsgraph, x=size, y=size)
        started = time.perf_counter()
        with bpy.context.temp_override(area=area, region=region):
            for _ in range(frames):
                viewport.draw_view3d(scene, bpy.context.view_layer, space, region,
                                     camera.matrix_world.inverted(), projection,
                                     do_color_management=False, draw_background=True)
            with viewport.bind():
                fb = gpu.state.active_framebuffer_get()
                pixels = np.array(fb.read_color(0, 0, size, size, 4, 0, 'FLOAT')).reshape(size, size, 4)
        elapsed = (time.perf_counter() - started) / frames
        return pixels, elapsed

    def save(name, target, frames=24):
        pixels, elapsed = draw(target, frames)
        timings[name] = elapsed * 1000
        rgb = pixels[:, :, :3]
        metrics[name] = {
            'mean_rgb': rgb.mean(axis=(0, 1)).tolist(),
            'red_excess': float(np.maximum(rgb[:, :, 0] - rgb[:, :, 1], 0).mean()),
            'max': float(rgb.max()),
            'finite': bool(np.isfinite(rgb).all()),
        }
        # Creating Image datablocks changes the depsgraph and invalidates material caches.
        # Keep readbacks on the host and export only after all temporal checks are complete.
        images.append((name, pixels))
        return pixels

    warm = (0, 0, 2)
    offscreen = (0, 0, -0.6)
    try:
        save('mirror_warm_view', warm)
        save('mirror_cached_offscreen', offscreen)
        # A separate viewport is a genuine cold surface cache at the same camera position.
        viewport.free()
        viewport = gpu.types.GPUOffScreen(size, size, format='RGBA16F')
        save('mirror_cold_offscreen', offscreen)
        save('mirror_rewarm', warm)
        save('mirror_stable_early', offscreen)
        save('mirror_stable_late', offscreen, 96)
        emitter.location.x = 5
        bpy.context.view_layer.update()
        save('mirror_moved_source', offscreen)
        emitter.location.x = 0
        bpy.context.view_layer.update()
        save('mirror_restored_source', warm)
        for vertex in emitter.data.vertices:
            vertex.co.x += 20
        emitter.data.update()
        bpy.context.view_layer.update()
        save('mirror_deformed_source', offscreen)
        for vertex in emitter.data.vertices:
            vertex.co.x -= 20
        emitter.data.update()
        bpy.context.view_layer.update()
        save('mirror_deformation_rewarm', warm)
        # Remove the original two triangles entirely, then restore them with a different diagonal.
        mesh = emitter.data
        positions = [tuple(vertex.co) for vertex in mesh.vertices]
        mesh.clear_geometry()
        mesh.from_pydata([(x + 20, y, z) for x, y, z in positions], [], [(0, 1, 2), (0, 2, 3)])
        mesh.update()
        bpy.context.view_layer.update()
        save('mirror_topology_changed', offscreen)
        mesh.clear_geometry()
        mesh.from_pydata(positions, [], [(0, 1, 3), (1, 2, 3)])
        mesh.update()
        bpy.context.view_layer.update()
        save('mirror_topology_rewarm', warm)
        save('mirror_topology_restored', offscreen)
        scene.eevee.ray_tracing_method = 'SCREEN'
        for vertex in mesh.vertices:
            vertex.co.x += 20
        mesh.update()
        bpy.context.view_layer.update()
        save('screen_edit_geometry', offscreen)
        scene.eevee.ray_tracing_method = 'HARDWARE'
        save('hardware_after_screen_edit', offscreen)
        for vertex in mesh.vertices:
            vertex.co.x -= 20
        mesh.update()
        bpy.context.view_layer.update()
        save('mirror_mode_rewarm', warm)
        # Shared mesh, distinct per-instance material assignment and radiance.
        source_x, source_scale = emitter.location.x, emitter.scale.x
        emitter.location.x, emitter.scale.x = -1.2, 0.6
        twin = emitter.copy()
        twin.data = emitter.data
        bpy.context.collection.objects.link(twin)
        twin.location.x = 1.2
        twin.material_slots[0].link = 'OBJECT'
        twin.material_slots[0].material = material('InstanceBlue', (0.005, 0.005, 1), emission=4)
        viewport.free()
        viewport = gpu.types.GPUOffScreen(size, size, format='RGBA16F')
        instance_pixels = save('instance_materials', offscreen, 64)
        metrics['instance_channels'] = {
            'red_pixels': int(((instance_pixels[:, :, 0] > 0.5) & (instance_pixels[:, :, 2] < 0.2)).sum()),
            'blue_pixels': int(((instance_pixels[:, :, 2] > 0.5) & (instance_pixels[:, :, 0] < 0.2)).sum()),
        }
        bpy.data.objects.remove(twin, do_unlink=True)
        emitter.location.x, emitter.scale.x = source_x, source_scale
        nt = source.node_tree
        output_node = nt.nodes.get('Material Output')
        source_bsdf = nt.nodes['Principled BSDF']
        emission_node = nt.nodes.new('ShaderNodeEmission')
        emission_node.inputs['Color'].default_value = (0, 1, 0, 1)
        emission_node.inputs['Strength'].default_value = 4
        nt.links.new(emission_node.outputs[0], output_node.inputs['Surface'])
        save('standalone_emission', offscreen, 64)
        nt.nodes.remove(emission_node)
        nt.links.new(source_bsdf.outputs[0], output_node.inputs['Surface'])
        uv = mesh.uv_layers.new(name='UVMap')
        for loop in mesh.loops:
            co = mesh.vertices[loop.vertex_index].co
            uv.data[loop.index].uv = ((co.x + 1) * 0.5, (co.y + 1) * 0.5)
        mesh.update()
        texture_image = bpy.data.images.new('RTTexture', width=2, height=2, float_buffer=True)
        texture_image.pixels.foreach_set([1, 0, 0, 1, 0, 1, 0, 1] * 2)
        tex = source.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = texture_image
        tex.interpolation = 'Closest'
        link = source.node_tree.links.new(tex.outputs['Color'],
                                         source.node_tree.nodes['Principled BSDF'].inputs['Emission Color'])
        viewport.free()
        viewport = gpu.types.GPUOffScreen(size, size, format='RGBA16F')
        texture_pixels = save('texture_cold', offscreen, 64)
        metrics['texture_channels'] = {
            'red_pixels': int(((texture_pixels[:, :, 0] > 0.5) & (texture_pixels[:, :, 1] < 0.2)).sum()),
            'green_pixels': int(((texture_pixels[:, :, 1] > 0.5) & (texture_pixels[:, :, 0] < 0.2)).sum()),
        }
        texture_image.pixels.foreach_set([0, 0, 1, 1] * 4)
        texture_image.update()
        save('texture_changed', offscreen, 64)
        source.node_tree.links.remove(link)
        source.node_tree.nodes.remove(tex)
        bpy.data.images.remove(texture_image)
        bpy.context.view_layer.update()
        source.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 0
        bpy.context.view_layer.update()
        save('mirror_emission_disabled', offscreen)
        bpy.ops.object.light_add(type='POINT', location=(0, -2, 2))
        lamp = bpy.context.object
        lamp.data.energy = 2000
        viewport.free()
        viewport = gpu.types.GPUOffScreen(size, size, format='RGBA16F')
        save('direct_light_cold', offscreen, 64)
        diffuse_node = nt.nodes.new('ShaderNodeBsdfDiffuse')
        diffuse_node.inputs['Color'].default_value = (1, 0.005, 0.005, 1)
        nt.links.new(diffuse_node.outputs[0], output_node.inputs['Surface'])
        save('standalone_diffuse', offscreen, 64)
        nt.nodes.remove(diffuse_node)
        nt.links.new(source_bsdf.outputs[0], output_node.inputs['Surface'])
        # This blocker lies between the lamp and source, above the mirror's ray to the source.
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -1, 2))
        blocker = bpy.context.object
        blocker.scale = (4, 0.1, 0.9)
        blocker.data.materials.append(material('Blocker', (0, 0, 0)))
        save('direct_light_shadowed', offscreen, 64)
        blocker.visible_shadow = False
        save('direct_light_non_caster', offscreen, 64)
        bpy.data.objects.remove(blocker, do_unlink=True)
        save('direct_light_warm', warm)
        save('direct_light_cached', offscreen)
        lamp.data.energy = 0
        bpy.context.view_layer.update()
        save('direct_light_changed', offscreen)
        lamp.data.energy = 2000
        bpy.context.view_layer.update()
        save('direct_light_rewarm', warm)
        lamp.rotation_euler = (emitter.location - lamp.location).to_track_quat('-Z', 'Y').to_euler()
        for light_type in ['AREA', 'SPOT', 'SUN']:
            lamp.data.type = light_type
            lamp.data.energy = 2 if light_type == 'SUN' else 2000
            if light_type == 'SPOT':
                lamp.data.spot_size = math.radians(100)
            save('light_' + light_type.lower(), offscreen, 64)
        bpy.data.objects.remove(lamp, do_unlink=True)
        bpy.context.view_layer.update()
        save('direct_light_deleted', offscreen)
        source.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 4
        receiver.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.6
        bpy.context.view_layer.update()
        save('rough_warm', warm)
        save('rough_cached_offscreen', offscreen, 96)
        receiver.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0
        receiver.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 1
        bpy.context.view_layer.update()
        save('diffuse_warm', warm)
        save('diffuse_cached_offscreen', offscreen, 96)
        if metrics['ray_query']:
            # Fresh viewport: neither this cache nor surface cache ever saw the source.
            viewport.free()
            viewport = gpu.types.GPUOffScreen(size, size, format='RGBA16F')
            bpy.app.debug_value = 22
            save('world_irradiance', offscreen, 128)
            save('world_irradiance_stable', offscreen, 256)
            bpy.app.debug_value = 0
        bpy.data.objects.remove(emitter, do_unlink=True)
        bpy.context.view_layer.update()
        deleted = save('diffuse_deleted_source', offscreen, 96)
        if metrics['ray_query']:
            bpy.app.debug_value = 22
            save('world_irradiance_deleted', offscreen, 64)
            bpy.app.debug_value = 0
        scene.eevee.ray_tracing_method = 'SCREEN'
        screen = save('screen_regression', offscreen, 96)
        metrics['cpu_draw_including_gpu_wait_ms'] = timings
        metrics['checks'] = {
            'finite': all(value.get('finite', True) for value in metrics.values() if isinstance(value, dict)),
            'instance_material_identity': metrics['instance_channels']['red_pixels'] > 10 and
                                          metrics['instance_channels']['blue_pixels'] > 10,
            'standalone_emission': metrics['standalone_emission']['mean_rgb'][1] >
                                   metrics['standalone_emission']['mean_rgb'][0] + 0.01,
            'standalone_diffuse': metrics['standalone_diffuse']['red_excess'] > 0.005,
            'area_spot_sun': all(metrics['light_' + t]['red_excess'] > 0.005 for t in ['area', 'spot', 'sun']),
            'shadow_visibility_flag': metrics['direct_light_non_caster']['red_excess'] >
                                      metrics['direct_light_shadowed']['red_excess'] + 0.02,
            'texture_barycentric_uv': metrics['texture_channels']['red_pixels'] > 10 and
                                      metrics['texture_channels']['green_pixels'] > 10,
            'texture_invalidation': metrics['texture_changed']['mean_rgb'][2] >
                                    metrics['texture_changed']['mean_rgb'][0] + 0.01,
            'never_visible_emission': metrics['mirror_cold_offscreen']['red_excess'] > 0.005,
            'cold_matches_warm': abs(metrics['mirror_cached_offscreen']['red_excess'] -
                                     metrics['mirror_cold_offscreen']['red_excess']) < 0.02,
            'emission_invalidation': metrics['mirror_emission_disabled']['red_excess'] <
                                     metrics['mirror_cached_offscreen']['red_excess'] * 0.2,
            'deletion_invalidation': metrics['diffuse_deleted_source']['red_excess'] <
                                     metrics['diffuse_cached_offscreen']['red_excess'] * 0.2,
            'diffuse_hardware_gi': metrics['diffuse_cached_offscreen']['red_excess'] > 0.005,
            'rough_reflection': metrics['rough_cached_offscreen']['red_excess'] > 0.005,
            'no_energy_growth': abs(metrics['mirror_stable_late']['red_excess'] -
                                    metrics['mirror_stable_early']['red_excess']) < 0.01,
            'transform_invalidation': metrics['mirror_moved_source']['red_excess'] < 0.005,
            'deformation_invalidation': metrics['mirror_deformed_source']['red_excess'] < 0.005,
            'topology_invalidation': metrics['mirror_topology_changed']['red_excess'] < 0.005,
            'topology_restored': metrics['mirror_topology_restored']['red_excess'] > 0.005,
            'mode_switch_geometry': metrics['hardware_after_screen_edit']['red_excess'] < 0.005,
            'never_visible_direct_light': metrics['direct_light_cold']['red_excess'] > 0.005,
            'ray_query_shadow': metrics['direct_light_shadowed']['red_excess'] <
                                 metrics['direct_light_cold']['red_excess'] * 0.5,
            'direct_light_cache': metrics['direct_light_cached']['red_excess'] > 0.005,
            'light_change_invalidation': metrics['direct_light_changed']['red_excess'] < 0.005,
            'light_delete_invalidation': metrics['direct_light_deleted']['red_excess'] < 0.005,
        }
        if metrics['ray_query']:
            metrics['checks']['independent_world_irradiance'] = metrics['world_irradiance']['red_excess'] > 0.005
            metrics['checks']['world_irradiance_energy'] = abs(
                metrics['world_irradiance_stable']['red_excess'] -
                metrics['world_irradiance']['red_excess']) < max(0.01, metrics['world_irradiance']['red_excess'] * 0.2)
            metrics['checks']['world_irradiance_invalidation'] = metrics['world_irradiance_deleted']['red_excess'] < 0.002
        if not metrics['ray_query']:
            metrics['checks'] = {'finite': metrics['checks']['finite'],
                                 'screen_fallback_matches': float(np.abs(deleted-screen).mean()) < 0.003}
    finally:
        viewport.free()
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        for name, pixels in images:
            image = bpy.data.images.new(name, width=size, height=size, alpha=True, float_buffer=True)
            image.pixels.foreach_set(pixels.ravel())
            image.filepath_raw = str(output / (name + '.exr'))
            image.file_format = 'OPEN_EXR'
            image.save()
            image.save_render(str(output / (name + '.png')), scene=scene)
            bpy.data.images.remove(image)
    print('HARDWARE_RT_ACCEPTANCE', json.dumps(metrics['checks']), flush=True)
    failed = [name for name, passed in metrics['checks'].items() if not passed]
    if failed:
        raise AssertionError('Hardware RT acceptance failed: ' + ', '.join(failed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--debug-value', type=int, default=0)
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--profile-render', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    bpy.app.debug_value = args.debug_value
    try:
        run(args.output, args.denoise, args.profile_render)
    except Exception:
        traceback.print_exc()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    if not bpy.app.background:
        # Command-line scripts run before the first UI event loop. Schedule exit after startup.
        def quit_after_startup():
            bpy.ops.wm.quit_blender()
            return None
        bpy.app.timers.register(quit_after_startup, first_interval=0.1)
