# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""GPU fixtures for lightweight PBR, mixed native/HWRT RR, SR and Normal Map support.

Run with --factory-startup --gpu-backend vulkan --debug-gpu. Check NGX and validation
logs as well as the output: successful native fallback is not successful reconstruction.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Vector


def main(args):
    if not args.existing:
        spec = importlib.util.spec_from_file_location('fixture', Path(__file__).with_name('eevee_dlss_render.py'))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        fixture.make_scene()
    scene = bpy.context.scene
    bpy.app.debug_value = args.debug_value
    scene.render.image_settings.media_type = 'IMAGE'
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.eevee.dlss_render_quality = 'OFF' if args.mode == 'NATIVE' else args.quality
    scene.eevee.dlss_render_rr = args.mode == 'RR'
    scene.eevee.taa_render_samples = args.samples
    if args.denoise != 'DEFAULT':
        scene.eevee.ray_tracing_options.use_denoise = args.denoise != 'NONE'
        scene.eevee.ray_tracing_options.denoise_spatial = True
        scene.eevee.ray_tracing_options.denoise_temporal = args.denoise in {'TEMPORAL', 'FULL'}
        scene.eevee.ray_tracing_options.denoise_bilateral = args.denoise == 'FULL'
    if args.orthographic and scene.camera:
        scene.camera.data.type = 'ORTHO'
        scene.camera.data.ortho_scale = 10
    scene.eevee.use_overscan = args.overscan > 0
    scene.eevee.overscan_size = args.overscan
    if not args.keep_passes:
        for layer in scene.view_layers:
            for prop in layer.bl_rna.properties:
                if prop.identifier.startswith('use_pass_') and not prop.is_readonly:
                    setattr(layer, prop.identifier, prop.identifier == 'use_pass_combined')
    if not args.existing:
        material = bpy.data.materials['PBR0']
        tree = material.node_tree
        bsdf = tree.nodes['Principled BSDF']
        if args.case in {'LEGACY_PASSES', 'LEGACY_WITH_DEPTH', 'LEGACY_WITH_TRANSPARENT'}:
            # Saved Cycles flags hidden in EEVEE's UI must not disable final reconstruction.
            for name in ('use_pass_diffuse_indirect', 'use_pass_glossy_indirect',
                         'use_pass_transmission_direct', 'use_pass_transmission_indirect',
                         'use_pass_transmission_color'):
                setattr(scene.view_layers[0], name, True)
        if args.case in {'MULTIPASS', 'LEGACY_WITH_DEPTH'}:
            scene.view_layers[0].use_pass_z = True
        if args.case == 'LEGACY_WITH_TRANSPARENT':
            scene.view_layers[0].eevee.use_pass_transparent = True
        if args.case == 'F12_MIXED':
            bsdf.inputs['Subsurface Weight'].default_value = 0.8
            target = bpy.data.materials['PBR1'].node_tree
            glass = target.nodes.new('ShaderNodeBsdfGlass')
            target.links.new(glass.outputs[0], target.nodes['Material Output'].inputs['Surface'])
            target_material = bpy.data.materials['PBR3']
            target_material.surface_render_method = 'BLENDED'
            target_material.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = 0.4
        if args.case in {'AFFINE', 'AFFINE_REFERENCE'}:
            color, tint, factor = (0.1, 0.2, 0.3, 0.5), (0.4, 0.7, 0.9, 1), 0.6
            bsdf.inputs['Base Color'].default_value = tuple(
                color[i] * (1 - factor + factor * tint[i]) for i in range(3)) + (1,)
            bsdf.inputs['Roughness'].default_value = 0.3
            if args.case == 'AFFINE':
                image = bpy.data.images.new('Affine input', 1, 1, float_buffer=True)
                image.colorspace_settings.name = 'Non-Color'
                image.alpha_mode = 'CHANNEL_PACKED'
                image.pixels[:] = color
                image.pack()
                tex = tree.nodes.new('ShaderNodeTexImage')
                tex.image = image
                uv = tree.nodes.new('ShaderNodeUVMap')
                mapping = tree.nodes.new('ShaderNodeMapping')
                mapping.inputs['Location'].default_value = (0.25, -0.125, 0)
                mapping.inputs['Scale'].default_value = (2, 0.5, 1)
                tree.links.new(uv.outputs['UV'], mapping.inputs['Vector'])
                tree.links.new(mapping.outputs['Vector'], tex.inputs['Vector'])
                multiply = tree.nodes.new('ShaderNodeMixRGB')
                multiply.blend_type = 'MULTIPLY'
                multiply.inputs[0].default_value = factor
                multiply.inputs[2].default_value = tint
                tree.links.new(tex.outputs['Color'], multiply.inputs[1])
                tree.links.new(multiply.outputs[0], bsdf.inputs['Base Color'])
                scale = tree.nodes.new('ShaderNodeMath')
                scale.operation = 'MULTIPLY'
                scale.inputs[1].default_value = 0.4
                offset = tree.nodes.new('ShaderNodeMath')
                offset.operation = 'ADD'
                offset.inputs[1].default_value = 0.1
                tree.links.new(tex.outputs['Alpha'], scale.inputs[0])
                tree.links.new(scale.outputs[0], offset.inputs[0])
                tree.links.new(offset.outputs[0], bsdf.inputs['Roughness'])
        if args.case in {'GLASS', 'MIXED'}:
            target = bpy.data.materials['PBR1'].node_tree
            glass = target.nodes.new('ShaderNodeBsdfGlass')
            glass.inputs['Color'].default_value = (0.6, 0.9, 1, 1)
            glass.inputs['Roughness'].default_value = 0.15
            target.links.new(glass.outputs[0], target.nodes['Material Output'].inputs['Surface'])
        if args.case == 'MIXED':
            bsdf.inputs['Subsurface Weight'].default_value = 0.8
            bsdf.inputs['Subsurface Scale'].default_value = 0.05
            target = bpy.data.materials['PBR3'].node_tree
            noise = target.nodes.new('ShaderNodeTexNoise')
            target.links.new(noise.outputs['Color'], target.nodes['Principled BSDF'].inputs['Base Color'])
            for index in range(2):
                bpy.ops.mesh.primitive_plane_add(size=2, location=(-1 + index * 2, -1.0, 1.1),
                                                rotation=(math.pi / 2, 0, 0))
                hair = bpy.context.object
                hair.name = 'Mixed transparent layer %d' % index
                mat = bpy.data.materials.new(hair.name)
                mat.use_nodes = True
                mat.surface_render_method = 'BLENDED' if index else 'DITHERED'
                shader = mat.node_tree.nodes['Principled BSDF']
                shader.inputs['Base Color'].default_value = (0.05, 0.4, 0.2, 1)
                shader.inputs['Alpha'].default_value = 0.35
                hair.data.materials.append(mat)
        if args.case == 'EXPRESSION_CONTEXT':
            for obj in scene.objects:
                if obj.type == 'MESH':
                    obj.data.uv_layers.new(name='ContextUV', do_init=True)
            for mat in bpy.data.materials:
                if not mat.node_tree:
                    continue
                nt = mat.node_tree
                diffuse = nt.nodes.new('ShaderNodeBsdfDiffuse')
                coords = nt.nodes.new('ShaderNodeTexCoord')
                mapping = nt.nodes.new('ShaderNodeMapping')
                mapping.vector_type = 'TEXTURE'
                mapping.inputs['Rotation'].default_value = (0.1, 0.2, 0.3)
                mapping.inputs['Scale'].default_value = (1.5, 0.75, 2)
                noise = nt.nodes.new('ShaderNodeTexNoise')
                noise.noise_dimensions = '3D'
                noise.noise_type = 'FBM'
                geometry = nt.nodes.new('ShaderNodeNewGeometry')
                tangent = nt.nodes.new('ShaderNodeTangent')
                tangent.direction_type = 'UV_MAP'
                tangent.uv_map = 'ContextUV'
                cross = nt.nodes.new('ShaderNodeVectorMath')
                cross.operation = 'CROSS_PRODUCT'
                transform = nt.nodes.new('ShaderNodeVectorTransform')
                transform.vector_type = 'VECTOR'
                transform.convert_from = 'WORLD'
                transform.convert_to = 'OBJECT'
                fresnel = nt.nodes.new('ShaderNodeFresnel')
                mix = nt.nodes.new('ShaderNodeMixRGB')
                mix.blend_type = 'MULTIPLY'
                mix.inputs[2].default_value = (0.2, 0.5, 0.8, 1)
                nt.links.new(coords.outputs['Object'], mapping.inputs['Vector'])
                nt.links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
                nt.links.new(geometry.outputs['Normal'], cross.inputs[0])
                nt.links.new(tangent.outputs['Tangent'], cross.inputs[1])
                nt.links.new(cross.outputs['Vector'], transform.inputs['Vector'])
                nt.links.new(transform.outputs['Vector'], fresnel.inputs['Normal'])
                nt.links.new(fresnel.outputs[0], mix.inputs[0])
                nt.links.new(noise.outputs['Color'], mix.inputs[1])
                nt.links.new(mix.outputs[0], diffuse.inputs['Color'])
                nt.links.new(diffuse.outputs[0], nt.nodes['Material Output'].inputs['Surface'])
        if args.case in {'CLOSURES', 'CLOSURES_WEIGHT_EXPR', 'CLOSURES_NORMALS', 'CLOSURES_REFERENCE'}:
            a, b, c = (0.1, 0.2, 0.5, 1), (0.8, 0.1, 0.05, 1), (0.02, 0.02, 0.02, 1)
            if args.case != 'CLOSURES_REFERENCE':
                leaves = [tree.nodes.new('ShaderNodeBsdfDiffuse') for _ in range(3)]
                for leaf, color in zip(leaves, (a, b, c)):
                    leaf.inputs['Color'].default_value = color
                mix = tree.nodes.new('ShaderNodeMixShader')
                mix.inputs[0].default_value = 0.25
                if args.case == 'CLOSURES_WEIGHT_EXPR':
                    # Force program concatenation and register remapping in both Mix branches.
                    value = tree.nodes.new('ShaderNodeValue')
                    value.outputs[0].default_value = 0.5
                    squared = tree.nodes.new('ShaderNodeMath')
                    squared.operation = 'POWER'
                    squared.inputs[1].default_value = 2
                    tree.links.new(value.outputs[0], squared.inputs[0])
                    tree.links.new(squared.outputs[0], mix.inputs[0])
                if args.case == 'CLOSURES_NORMALS':
                    for obj in scene.objects:
                        if obj.type == 'MESH' and material in obj.data.materials[:]:
                            uv = obj.data.uv_layers.new(name='LayerNormalUV', do_init=True)
                            for item in uv.data:
                                item.uv.x = 1 - item.uv.x
                    for index, leaf in enumerate(leaves):
                        normal = tree.nodes.new('ShaderNodeNormalMap')
                        normal.uv_map = 'LayerNormalUV' if index == 1 else ''
                        normal.inputs['Color'].default_value = (
                            0.5 + 0.15 * (index - 1), 0.5, 0.95, 1)
                        tree.links.new(normal.outputs[0], leaf.inputs['Normal'])
                add = tree.nodes.new('ShaderNodeAddShader')
                tree.links.new(leaves[0].outputs[0], mix.inputs[1])
                tree.links.new(leaves[1].outputs[0], mix.inputs[2])
                tree.links.new(mix.outputs[0], add.inputs[0])
                tree.links.new(leaves[2].outputs[0], add.inputs[1])
                output = add.outputs[0]
            else:
                leaf = tree.nodes.new('ShaderNodeBsdfDiffuse')
                leaf.inputs['Color'].default_value = tuple(
                    0.75 * a[i] + 0.25 * b[i] + c[i] for i in range(3)) + (1,)
                output = leaf.outputs[0]
            tree.links.new(output, tree.nodes['Material Output'].inputs['Surface'])
        if args.case in {'EXPRESSION', 'EXPRESSION_REFERENCE'}:
            # Exercise nested group boundaries with a lossy color->float->color conversion,
            # muted-node routing, reroutes, and per-instance inputs. Compare against constants
            # in a separate run; HWRT must evaluate this graph for secondary hits as well.
            for index, name in enumerate(('PBR0', 'PBR1')):
                target = bpy.data.materials[name].node_tree
                target_bsdf = target.nodes['Principled BSDF']
                color = (0.15 + index * 0.3, 0.3, 0.6, 1)
                # Factory color management uses linear Rec.709 luminance coefficients.
                value = sum(c * w for c, w in zip(color[:3], (0.2126, 0.7152, 0.0722)))
                target_bsdf.inputs['Base Color'].default_value = (value, value, value, 1)
                target_bsdf.inputs['Roughness'].default_value = 0.25
                if args.case == 'EXPRESSION_REFERENCE':
                    continue
                if index == 0:
                    inner = bpy.data.node_groups.new('Scalar boundary', 'ShaderNodeTree')
                    inner.interface.new_socket(name='Value', in_out='INPUT', socket_type='NodeSocketFloat')
                    inner.interface.new_socket(name='Color', in_out='OUTPUT', socket_type='NodeSocketColor')
                    ii = inner.nodes.new('NodeGroupInput')
                    io = inner.nodes.new('NodeGroupOutput')
                    inner.links.new(ii.outputs['Value'], io.inputs['Color'])
                    group = bpy.data.node_groups.new('Expression PBR', 'ShaderNodeTree')
                    group.interface.new_socket(name='Tint', in_out='INPUT', socket_type='NodeSocketColor')
                    group.interface.new_socket(name='Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
                    gi = group.nodes.new('NodeGroupInput')
                    go = group.nodes.new('NodeGroupOutput')
                    shader = group.nodes.new('ShaderNodeBsdfPrincipled')
                    boundary = group.nodes.new('ShaderNodeGroup')
                    boundary.node_tree = inner
                    reroute = group.nodes.new('NodeReroute')
                    muted = group.nodes.new('ShaderNodeGamma')
                    muted.mute = True
                    group.links.new(gi.outputs['Tint'], boundary.inputs['Value'])
                    group.links.new(boundary.outputs['Color'], muted.inputs['Color'])
                    group.links.new(muted.outputs['Color'], reroute.inputs[0])
                    group.links.new(reroute.outputs[0], shader.inputs['Base Color'])
                    power = group.nodes.new('ShaderNodeMath')
                    power.operation = 'POWER'
                    power.inputs[0].default_value = 0.5
                    power.inputs[1].default_value = 2
                    group.links.new(power.outputs[0], shader.inputs['Roughness'])
                    group.links.new(shader.outputs[0], go.inputs['Shader'])
                instance = target.nodes.new('ShaderNodeGroup')
                instance.node_tree = group
                instance.inputs['Tint'].default_value = color
                # Match the shared program's Principled defaults in the reference graph.
                target_bsdf.inputs['Metallic'].default_value = 0
                target.links.new(instance.outputs['Shader'], target.nodes['Material Output'].inputs['Surface'])
            if args.case == 'EXPRESSION_REFERENCE':
                for name in ('PBR0', 'PBR1'):
                    bpy.data.materials[name].node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0
        if args.case == 'IMAGE_ARRAYS':
            # Four original-resolution images exceed the former single 64 MiB float4 pool.
            # PBR3 shares PBR0's image through a distinct Image Texture node.
            array_images = []
            for index, name in enumerate(('PBR0', 'PBR1', 'PBR2', 'Ground', 'PBR3')):
                target_tree = bpy.data.materials[name].node_tree
                image = array_images[0] if index == 4 else bpy.data.images.new(
                    'ArrayImage%d' % index, 2048, 2048)
                if index != 4:
                    image.generated_color = (0.1 + index * 0.15, 0.2, 0.4, 1)
                    array_images.append(image)
                texture = target_tree.nodes.new('ShaderNodeTexImage')
                texture.image = image
                target_tree.links.new(texture.outputs['Color'],
                                      target_tree.nodes['Principled BSDF'].inputs['Base Color'])
        if args.case == 'SSS':
            bsdf.inputs['Subsurface Weight'].default_value = 0.8
            bsdf.inputs['Subsurface Scale'].default_value = args.sss_scale
        if args.case == 'TRANSLUCENT':
            node = tree.nodes.new('ShaderNodeBsdfTranslucent')
            tree.links.new(node.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
        if args.case.startswith('NORMAL'):
            image = bpy.data.images.new('NormalPixels', 32, 32)
            image.colorspace_settings.name = 'Non-Color'
            pixels = []
            for y in range(32):
                for x in range(32):
                    nx = 0.0 if args.case == 'NORMAL_NEUTRAL' else 0.65 * math.sin(x * math.pi / 4)
                    ny = 0.0 if args.case == 'NORMAL_NEUTRAL' else 0.35 * math.cos(y * math.pi / 4)
                    pixels.extend((nx * 0.5 + 0.5, ny * 0.5 + 0.5,
                                   math.sqrt(max(0, 1-nx*nx-ny*ny)) * 0.5 + 0.5, 1))
            image.pixels[:] = pixels
            image.pack()
            texture = tree.nodes.new('ShaderNodeTexImage')
            texture.image = image
            node = tree.nodes.new('ShaderNodeNormalMap')
            node.space = args.normal_space
            node.convention = args.convention
            node.inputs['Strength'].default_value = args.strength
            tree.links.new(texture.outputs['Color'], node.inputs['Color'])
            tree.links.new(node.outputs['Normal'], bsdf.inputs['Normal'])
            if args.case == 'NORMAL_STRENGTH':
                tree.links.new(texture.outputs['Alpha'], node.inputs['Strength'])
            if args.case == 'NORMAL_BUMP':
                bump = tree.nodes.new('ShaderNodeBump')
                tree.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
            targets = [o for o in scene.objects if o.type == 'MESH' and material in o.data.materials[:]]
            for obj in targets:
                if args.case == 'NORMAL_MISSING_UV':
                    for uv in list(obj.data.uv_layers):
                        obj.data.uv_layers.remove(uv)
                obj.scale = (-1.0, 1.2, 0.8) if args.case == 'NORMAL_NEGATIVE' else obj.scale
                if args.case == 'NORMAL_DEGENERATE':
                    for value in obj.data.uv_layers.active.data:
                        value.uv = (0, 0)
                if args.case == 'NORMAL_DEFORM':
                    obj.shape_key_add(name='Basis')
                    shape = obj.shape_key_add(name='Deform')
                    for vertex in shape.data:
                        vertex.co.z += 0.2 * math.sin(vertex.co.x * 4)
                    shape.value = 0
                    shape.keyframe_insert('value', frame=1)
                    shape.value = 1
                    shape.keyframe_insert('value', frame=2)
                if args.case in {'NORMAL_UV', 'NORMAL_UV_INDEPENDENT'}:
                    uv = obj.data.uv_layers.new(name='NormalUV', do_init=True)
                    for value in uv.data:
                        value.uv.x = 1-value.uv.x
                    node.uv_map = 'NormalUV'
                    if args.case == 'NORMAL_UV':
                        coord = tree.nodes.new('ShaderNodeUVMap')
                        coord.uv_map = 'NormalUV'
                        tree.links.new(coord.outputs['UV'], texture.inputs['Vector'])
        if args.case == 'NORMAL_SECONDARY':
            target = targets[0]
            for obj in list(scene.objects):
                if obj.type == 'MESH' and obj != target:
                    bpy.data.objects.remove(obj, do_unlink=True)
            target.animation_data_clear()
            target.location = (-1, -6, 2)
            target.scale = (1, 1, 1)
            camera = scene.camera
            camera.location = (0, -3, 2)
            camera.rotation_euler = Vector((0, 1, 0)).to_track_quat('-Z', 'Y').to_euler()
            camera.data.lens = 35
            bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 2), rotation=(math.pi/2, 0, 0))
            mirror = bpy.data.materials.new('Secondary-hit mirror')
            mirror.use_nodes = True
            principled = mirror.node_tree.nodes['Principled BSDF']
            principled.inputs['Base Color'].default_value = (1, 1, 1, 1)
            principled.inputs['Metallic'].default_value = 1
            principled.inputs['Roughness'].default_value = 0.02
            bpy.context.object.data.materials.append(mirror)
            for obj in scene.objects:
                if obj.type == 'LIGHT':
                    obj.location = (-2, -5, 5)
                    obj.data.energy = 2000
                    obj.rotation_euler = (Vector(target.location)-obj.location).to_track_quat('-Z', 'Y').to_euler()
        if args.case in {'DITHERED', 'BLENDED', 'COLORED'}:
            material.surface_render_method = 'BLENDED' if args.case != 'DITHERED' else 'DITHERED'
            bsdf.inputs['Alpha'].default_value = 0.4
            if args.case == 'COLORED':
                transparent = tree.nodes.new('ShaderNodeBsdfTransparent')
                transparent.inputs['Color'].default_value = (0.15, 0.7, 0.95, 1)
                mix = tree.nodes.new('ShaderNodeMixShader')
                mix.inputs[0].default_value = 0.5
                tree.links.new(bsdf.outputs[0], mix.inputs[1])
                tree.links.new(transparent.outputs[0], mix.inputs[2])
                tree.links.new(mix.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
    args.output.mkdir(parents=True, exist_ok=True)
    if args.save:
        scene.eevee.dlss_quality = args.quality
        scene.eevee.dlss_rr = args.mode == 'RR'
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output / 'fixture.blend'))
    results = []
    for index, frame in enumerate(map(int, args.sequence.split(','))):
        scene.frame_set(frame)
        if not args.existing and args.case == 'MIXED':
            phase = frame * math.tau / 100
            scene.camera.location.x = 4 + math.sin(phase) * 0.5
            scene.camera.rotation_euler = (Vector((0, 0, 0.7)) - scene.camera.location).to_track_quat('-Z', 'Y').to_euler()
            for layer in range(2):
                obj = bpy.data.objects['Mixed transparent layer %d' % layer]
                obj.location.z = 1.1 + 0.2 * math.sin(phase + layer)
                obj.rotation_euler.z = 0.1 * math.cos(phase)
                for vertex in obj.data.vertices:
                    vertex.co.z = 0.1 * math.sin(phase + vertex.index)
                obj.data.update()
                obj.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = (
                    0.35 + 0.25 * math.sin(phase + layer))
        if not args.existing and args.case == 'IMAGE_ARRAYS':
            size = 1024 if frame == 2 else 2048
            if array_images[0].size[0] != size:
                array_images[0].scale(size, size)
        if not args.existing and args.case == 'NORMAL_EDIT':
            node.inputs['Strength'].default_value = 1 if frame == 1 else 2
        if not args.existing and args.case == 'DITHERED':
            bsdf.inputs['Alpha'].default_value = 0.4 if frame == 1 else 0.7
        name = '%02d_frame%04d' % (index, frame)
        if args.dump_state:
            bpy.context.view_layer.update()
            dependency_graph = bpy.context.evaluated_depsgraph_get()
            state = {}
            for obj in scene.objects:
                evaluated = obj.evaluated_get(dependency_graph)
                state[obj.name] = {'matrix': [list(row) for row in evaluated.matrix_world]}
                if obj.name.startswith('Mixed transparent'):
                    state[obj.name]['vertices'] = [list(v.co) for v in evaluated.data.vertices]
                    state[obj.name]['alpha'] = obj.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value
            (args.output / (name + '.state.json')).write_text(json.dumps(state, indent=2))
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.filepath = str(args.output / (name + '.exr'))
        print('COMPAT_BEGIN', args.case, args.mode, name, flush=True)
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(scene.render.filepath, check_existing=False)
        pixels = np.array(image.pixels[:], dtype=np.float32).reshape(args.height, args.width, 4)
        assert np.isfinite(pixels).all()
        assert np.max(np.abs(pixels[..., 3] - 1)) < 1e-4
        np.save(args.output / (name + '.npy'), pixels)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '8'
        bpy.data.images['Render Result'].save_render(str(args.output / (name + '.png')), scene=scene)
        bpy.data.images.remove(image)
        results.append({'frame': frame, 'seconds': time.perf_counter()-start,
                        'mean': float(pixels[..., :3].mean()), 'size': [args.width, args.height]})
        print('COMPAT_END', json.dumps(results[-1]), flush=True)
    (args.output / 'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sss-scale', type=float, default=0.05)
    parser.add_argument('--case', default='BASE')
    parser.add_argument('--mode', choices=['SR', 'RR', 'NATIVE'], default='SR')
    parser.add_argument('--quality', default='QUALITY')
    parser.add_argument('--normal-space', default='TANGENT')
    parser.add_argument('--convention', default='OPENGL')
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--debug-value', type=int, default=0)
    parser.add_argument('--overscan', type=float, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--denoise', choices=['DEFAULT', 'NONE', 'SPATIAL', 'TEMPORAL', 'FULL'], default='DEFAULT')
    parser.add_argument('--sequence', default='1')
    parser.add_argument('--existing', action='store_true')
    parser.add_argument('--keep-passes', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--dump-state', action='store_true')
    parser.add_argument('--orthographic', action='store_true')
    main(parser.parse_args(sys.argv[sys.argv.index('--')+1:]))
