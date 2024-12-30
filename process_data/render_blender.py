# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Tested with Blender 2.9
#

import argparse, sys, os, math, re
import bpy
from glob import glob
import numpy as np
import json

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/imgs',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.0,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='16',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--resolution', type=int, default=1024,
                    help='Resolution of the images.')
parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = args.color_depth # ('8', '16')
render.image_settings.file_format = args.format # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
render.film_transparent = True

scene.use_nodes = True
scene.view_layers["View Layer"].use_pass_normal = True
scene.view_layers["View Layer"].use_pass_diffuse_color = True
scene.view_layers["View Layer"].use_pass_object_index = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new('CompositorNodeRLayers')

# Create id map output nodes
id_file_output = nodes.new(type="CompositorNodeOutputFile")
id_file_output.label = 'ID Output'
id_file_output.base_path = ''
id_file_output.file_slots[0].use_node_format = True
id_file_output.format.file_format = args.format
id_file_output.format.color_depth = args.color_depth

if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])
else:
    id_file_output.format.color_mode = 'BW'

    divide_node = nodes.new(type='CompositorNodeMath')
    divide_node.operation = 'DIVIDE'
    divide_node.use_clamp = False
    divide_node.inputs[1].default_value = 2**int(args.color_depth)

    links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
    links.new(divide_node.outputs[0], id_file_output.inputs[0])

# Delete default cube
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')

bpy.ops.import_scene.obj(filepath=args.obj)

obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj

# Possibly disable specular shading
for slot in obj.material_slots:
    node = slot.material.node_tree.nodes['Principled BSDF']
    node.inputs['Specular'].default_value = 0.05

if args.scale != 1:
    bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
    bpy.ops.object.transform_apply(scale=True)

if args.edge_split:
    bpy.ops.object.modifier_add(type='EDGE_SPLIT')
    context.object.modifiers["EdgeSplit"].split_angle = 1.32645
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

# Set objekt IDs
obj.pass_index = 1

# Make light just directional, disable shadows.
light = bpy.data.lights['Light']
light.type = 'SUN'
light.use_shadow = False
# Possibly disable specular shading:
light.specular_factor = 1.0
light.energy = 5.0


# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.light_add(type='SUN')
light2 = bpy.data.lights['Sun']
light2.use_shadow = False
light2.specular_factor = 1.0
light2.energy = 5.0
bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
bpy.data.objects['Sun'].rotation_euler[0] += 180

bpy.ops.object.light_add(type='SUN')
light3 = bpy.data.lights['Sun.001']
light3.use_shadow = False
light3.specular_factor = 1.0
light3.energy = 5.0
bpy.data.objects['Sun.001'].rotation_euler = bpy.data.objects['Light'].rotation_euler
bpy.data.objects['Sun.001'].rotation_euler[1] += 180

bpy.ops.object.light_add(type='SUN')
light4 = bpy.data.lights['Sun.002']
light4.use_shadow = False
light4.specular_factor = 1.0
light4.energy = 10.0
bpy.data.objects['Sun.002'].rotation_euler = bpy.data.objects['Light'].rotation_euler
bpy.data.objects['Sun.002'].rotation_euler[2] += 180

# Place camera
cam = scene.objects['Camera']
cam.location = (0, 1, 0.6)
cam.location = (0, 1.5, 0.9)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty

stepsize = 360.0 / args.views
rotation_mode = 'XYZ'

model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
fp = os.path.join(os.path.abspath(args.output_folder), 'images', 'images')

poses = []
filename = []
for i in range(0, args.views+1):
    print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))
    render_file_path = fp + '_r2_{0:03d}'.format(int(i * stepsize))
    ###
    filename.append('images/'+os.path.basename(render_file_path)+'.png')
    ###
    scene.render.filepath = render_file_path
    # depth_file_output.file_slots[0].path = render_file_path + "_depth"
    # normal_file_output.file_slots[0].path = render_file_path + "_normal"
    # albedo_file_output.file_slots[0].path = render_file_path + "_albedo"
    id_file_output.file_slots[0].path = render_file_path + "_id"

    bpy.ops.render.render(write_still=True)  # render still
    ### TODO: output camera matrix
    bpy.context.scene.view_layers['View Layer'].update()
    c2w = np.array(cam.matrix_world)
    poses.append(c2w)
    ###
    cam_empty.rotation_euler[2] += math.radians(stepsize)

### TODO: 增加更多旋转角度
for i in range(0, args.views + 1):
    print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))
    render_file_path = fp + '_r1_{0:03d}'.format(int(i * stepsize))
    ###
    filename.append('images/' + os.path.basename(render_file_path) + '.png')
    ###
    scene.render.filepath = render_file_path
    # depth_file_output.file_slots[0].path = render_file_path + "_depth"
    # normal_file_output.file_slots[0].path = render_file_path + "_normal"
    # albedo_file_output.file_slots[0].path = render_file_path + "_albedo"
    id_file_output.file_slots[0].path = render_file_path + "_id"

    bpy.ops.render.render(write_still=True)  # render still
    ### TODO: output camera matrix
    bpy.context.scene.view_layers['View Layer'].update()
    c2w = np.array(cam.matrix_world)
    poses.append(c2w)
    ###
    cam_empty.rotation_euler[1] += math.radians(stepsize)

for i in range(0, args.views + 1):
    print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))
    render_file_path = fp + '_r0_{0:03d}'.format(int(i * stepsize))
    ###
    filename.append('images/' + os.path.basename(render_file_path) + '.png')
    ###
    scene.render.filepath = render_file_path
    # depth_file_output.file_slots[0].path = render_file_path + "_depth"
    # normal_file_output.file_slots[0].path = render_file_path + "_normal"
    # albedo_file_output.file_slots[0].path = render_file_path + "_albedo"
    id_file_output.file_slots[0].path = render_file_path + "_id"

    bpy.ops.render.render(write_still=True)  # render still
    ### TODO: output camera matrix
    bpy.context.scene.view_layers['View Layer'].update()
    c2w = np.array(cam.matrix_world)
    poses.append(c2w)
    ###
    cam_empty.rotation_euler[0] += math.radians(stepsize)



# focal = cam.data.lens*args.resolution/cam.data.sensor_width

out = {
    "camera_angle_x": cam.data.angle,
    "camera_angle_y": cam.data.angle,
    "cx": render.resolution_x/2,
    "cy": render.resolution_y/2,
    "w": render.resolution_x,
    "h": render.resolution_y,
    "frames": [],
}
for i in range(len(filename)):
    name = filename[i]
    c2w = poses[i]
    frame = {"file_path": name[:-4], "transform_matrix": c2w}
    out["frames"].append(frame)
for f in out["frames"]:
    f["transform_matrix"] = f["transform_matrix"].tolist()
with open(os.path.join(os.path.abspath(args.output_folder), 'transforms_train.json'), "w") as outfile:
    json.dump(out, outfile, indent=2)

with open(os.path.join(os.path.abspath(args.output_folder), 'transforms_test.json'), "w") as outfile:
    json.dump(out, outfile, indent=2)
    