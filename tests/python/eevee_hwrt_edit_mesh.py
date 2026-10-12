# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Windowed HWRT edit-mesh regression (requires Vulkan ray-query support).

blender --factory-startup --gpu-backend vulkan --debug-gpu --python this_file.py -- --output DIR

Check the GPU log for valid HardwareRT scenes and validation errors as well as result.json.
Timers leave real viewport redraws between edits; background rendering cannot test this path.
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

import bmesh
import bpy
import gpu


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.view.show_splash = False
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_method = 'HARDWARE'
    scene.eevee.dlss_quality = 'OFF'
    scene.eevee.taa_samples = 8
    cube = bpy.data.objects['Cube']
    bpy.context.view_layer.objects.active = cube
    cube.select_set(True)
    material = bpy.data.materials.new('Edit mesh tangent regression')
    material.use_nodes = True
    tree = material.node_tree
    normal = tree.nodes.new('ShaderNodeNormalMap')
    tree.links.new(normal.outputs['Normal'], tree.nodes['Principled BSDF'].inputs['Normal'])
    cube.data.materials.clear()
    cube.data.materials.append(material)
    area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    area.spaces.active.shading.type = 'RENDERED'
    area.spaces.active.shading.use_compositor = 'DISABLED'
    state = {'step': 0, 'stages': []}

    def edit():
        bpy.ops.object.mode_set(mode='EDIT')

    def move():
        bm = bmesh.from_edit_mesh(cube.data)
        for v in bm.verts:
            v.co.z *= 1.4
        bmesh.update_edit_mesh(cube.data, loop_triangles=True, destructive=False)

    def subdivide():
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide(number_cuts=1)

    def empty():
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete(type='VERT')

    def refill():
        bpy.ops.mesh.primitive_cube_add()

    def object_mode():
        bpy.ops.object.mode_set(mode='OBJECT')

    def deform():
        modifier = cube.modifiers.new('Edit deformation', 'SIMPLE_DEFORM')
        modifier.angle = 0.4
        modifier.show_in_editmode = True
        modifier.show_on_cage = True
        edit()

    def evaluated():
        object_mode()
        cube.modifiers.clear()
        modifier = cube.modifiers.new('Edit solidify', 'SOLIDIFY')
        modifier.thickness = 0.15
        modifier.show_in_editmode = True
        edit()

    def subdivision():
        object_mode()
        cube.modifiers.clear()
        modifier = cube.modifiers.new('Edit subdivision', 'SUBSURF')
        modifier.levels = 1
        modifier.show_in_editmode = True
        edit()

    stages = [
        ('object', lambda: None),
        ('enter_edit', edit),
        ('move_vertices', move),
        ('change_topology', subdivide),
        ('empty_mesh', empty),
        ('refill_mesh', refill),
        ('exit_edit', object_mode),
        ('reenter_edit', edit),
        ('exit_again', object_mode),
        ('deformed_edit', deform),
        ('deformed_move', move),
        ('solidify_edit', evaluated),
        ('solidify_move', move),
    ]
    skipped = []
    if bpy.app.build_options.opensubdiv:
        stages.extend([('subdivision_edit', subdivision), ('subdivision_move', move)])
    else:
        skipped.append('Subdivision Surface: build has no OpenSubdiv')
    stages.append(('final_object', object_mode))

    def tick():
        try:
            i = state['step']
            with bpy.context.temp_override(area=area, region=region):
                if i:
                    name = stages[i - 1][0]
                    bpy.ops.screen.screenshot_area(filepath=str(output / (name + '.png')))
                    state['stages'].append(name)
                    print('HWRT_EDIT_COMPLETE', name, flush=True)
                if i == len(stages):
                    result = {'passed': True, 'backend': gpu.platform.backend_type_get(),
                              'stages': state['stages'], 'skipped': skipped}
                    (output / 'result.json').write_text(json.dumps(result, indent=2))
                    bpy.ops.wm.quit_blender()
                    return None
                name, action = stages[i]
                print('HWRT_EDIT_BEGIN', name, flush=True)
                action()
            state['step'] += 1
            area.tag_redraw()
            return 2.0
        except Exception:
            traceback.print_exc()
            (output / 'failure.txt').write_text(traceback.format_exc())
            bpy.ops.wm.quit_blender()
            return None

    bpy.app.timers.register(tick, first_interval=5.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    main(args.output)
