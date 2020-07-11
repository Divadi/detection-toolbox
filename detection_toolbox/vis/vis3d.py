import numpy as np
from tqdm import tqdm
import time

'''
pc is n x (at least 3)
'''
def draw_lidar(
	pc, 
	ptcolor=(1, 1, 1), 
	fig=None,
	bgcolor=(0, 0, 0), 
	fig_size=(8000, 4000),
	draw_range_squares=False #! if you want to also see square boxes at 40, 80, 120 meters
):
	from mayavi import mlab
	mlab.options.offscreen = True

	if fig is None:
		fig = mlab.figure(bgcolor=bgcolor, size=fig_size)

	#! Draw origin & axes
	mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=1)

	axis_len = 2.0
	#! Each row is the "ending point" of each axis: x, y, z, starting from 0
	axes = np.array([
		[axis_len, 0., 0.], 
		[0., axis_len, 0.],
		[0., 0., axis_len],
	], dtype=np.float32)
	
	for axis_ind, axis in enumerate(["x", "y", "z"]):
		mlab.plot3d(
			[0, axes[axis_ind, 0]], 
			[0, axes[axis_ind, 1]], 
			[0, axes[axis_ind, 2]], 
			color=tuple([int(c) for c in axes[axis_ind] / axis_len]), #!(1,0,0), (0,1,0), (0,0,1),
			line_width=4,
			tube_radius=None,
			figure=fig
		)
		mlab.text3d(
			axes[axis_ind, 0],
			axes[axis_ind, 1],
			axes[axis_ind, 2],
			text=axis,
			color=tuple([int(c) for c in axes[axis_ind] / axis_len]),
			figure=fig,
			scale=(0.5, 0.5, 0.5)
		)

	if draw_range_squares:
		for i in [40, 80, 120]:
			mlab.plot3d(
				[i, i, -i, -i, i],
				[i, -i, -i, i, i],
				[0, 0, 0, 0, 0],
				color=(0.2, 0.2, 0.2),
				line_width=6,
				tube_radius=None,
				figure=fig
			)
	
	#! Draw point cloud
	mlab.points3d(
		pc[:,0], 
		pc[:,1], 
		pc[:,2], 
		color=ptcolor, 
		mode='point', 
		colormap='gnuplot', 
		scale_factor=1, 
		figure=fig
	)

	return fig

'''
Input:
	objects: KittiLabel object
	calib: Calibration object
	fig: mayavi figure object
	
	The goal of this function is to display only specific boxes in objects and be able to control what color each box is.
	If color_func is None, all boxes are displayed & default_color is used for all.
	if color_func is not None, it should be a function that takes in (index in objects.labels, label: SingleLabel) and
		returns either "None" or a tuple denoting the color the object should be displayed with.
		If it returns None for a label, the box is not displayed for that object/label
	text_func has the same inputs, but should output a string or None
'''
def draw_3d_boxes_from_objects_advanced(
	objects,
	calib,
	fig,
	default_color=(0, 1, 0), #! default color is green
	color_func=None,
	text_func=None
):
	from mayavi import mlab
	mlab.options.offscreen = True

	color_dict = dict()

	for label_ind, label in enumerate(objects):
		if color_func is not None:
			color = color_func(label_ind, label)
			if color is None:
				continue
		else:
			color = default_color

		if text_func is not None:
			text = text_func(label_ind, label)
			if text is None:
				text = ""
		else:
			text = ""

		corners_3d_rect = label.compute_box_3d() #! Gets corners of 3d box
		corners_3d_velo = calib.project_rect_to_velo(corners_3d_rect)

		if color not in color_dict.keys():
			color_dict[color] = {
				"boxes": [],
				"texts": [],
				"tmp": []
			}
		
		color_dict[color]['boxes'].append(corners_3d_velo)
		color_dict[color]['texts'].append(text)
		color_dict[color]['tmp'].append(label)

	for color, val in color_dict.items():
		draw_boxes_3d(val['boxes'], fig, box_color=color, text_color=color, text_list=val['texts'])

	return fig


'''
#! objects is type KittiLabel
#! calib is type Calibration
#! gt is boolean - whether it's gt boxes or not
gt is drawn in green, dt is drawn in red. dt also writes score
'''
def draw_boxes_from_objects(
	objects,
	calib,
	fig,
	occ_thresh=0.7,
	categories=["Car", "Pedestrian", "Cyclist", "Motorcycle", "Undefined"],
	text_func=None
):
	from mayavi import mlab
	mlab.options.offscreen = True

	default_connections = [
		(0, 1), (4, 5), (0, 4),
		(1, 2), (5, 6), (1, 5),
		(2, 3), (6, 7), (2, 6),
		(3, 0), (7, 4), (3, 7)
	] #! If the 8 corners were 0 indexed, these are the connections between them

	all_boxes = []
	all_text = []
	real_index = -1 #! keep track of & later display index of object in label file, so we can zoom in later

	for label in objects:
		real_index += 1
		if label.occlusion > occ_thresh or label.type not in categories:
			continue
		corners_3d_rect = label.compute_box_3d() #! Gets corners of 3d box
		corners_3d_velo = calib.project_rect_to_velo(corners_3d_rect)

		all_boxes.append(corners_3d_velo)

		if objects.gt:
			if text_func is None:
				all_text.append("v{}_{}".format(objects.view, real_index))
			else:
				all_text.append(str(text_func(label)))
		else:
			if text_func is None:
				all_text.append("{:.2f}".format(label.score))
			else:
				all_text.append(str(text_func(label)))

	if objects.gt:
		box_color = text_color = (0, 1, 0) #! green gt boxes
		# all_text = None
	else:
		box_color = text_color = (1, 0, 0) #! red dt boxes
	
	draw_boxes_3d(all_boxes, fig, box_color=box_color, text_color=text_color, text_list=all_text)

	return fig

def draw_boxes_3d(box3d, fig, box_color=(1,1,1), text_color=(1,0,0), text_scale=(.5,.5,.5), text_list=None):
	from mayavi import mlab
	mlab.options.offscreen = True

	default_connections = [
		(0, 1), (4, 5), (0, 4),
		(1, 2), (5, 6), (1, 5),
		(2, 3), (6, 7), (2, 6),
		(3, 0), (7, 4), (3, 7)
	] #! If the 8 corners were 0 indexed, these are the connections between them

	all_connections = []

	for box_index in range(len(box3d)):
		b = box3d[box_index]
		if text_list is not None:
			text_tmp = text_list[box_index]
			mlab.text3d(b[4,0], b[4,1], b[4,2], str(text_tmp), scale=text_scale, color=text_color, figure=fig)

		all_connections += default_connections #! Put in connections
		default_connections = [(a + 8, b + 8) for (a, b) in default_connections] #! Increment default by 8 

	
	all_corners_3d_velo = np.array(box3d).reshape(-1, 3) # just make it a list of points
	pts = mlab.points3d(
		all_corners_3d_velo[:, 0], 
		all_corners_3d_velo[:, 1], 
		all_corners_3d_velo[:, 2], 
		color=box_color, 
		mode="point", 
		scale_factor=1
	)
	pts.mlab_source.dataset.lines = np.array(all_connections)
	tube = mlab.pipeline.tube(pts, tube_radius=0.05)
	tube.filter.radius_factor = 1.
	mlab.pipeline.surface(tube, color=box_color)

	return fig


def set_view(fig, azimuth, elevation, distance):
	from mayavi import mlab
	mlab.options.offscreen = True
	#! view
	focalpoint = [0, 0, 0]
	mlab.view(
		azimuth=azimuth,
		elevation=elevation,
		distance=distance,
		focalpoint=focalpoint,
		figure=fig
	)
	return fig

#! Zooms-in the view to zoom_idx-th object
def zoom_view(objects, calib, fig, zoom_idx):
	from mayavi import mlab
	mlab.options.offscreen = True

	zoom_object = objects.labels[zoom_idx] #! Object to focus on
	x, y, z = zoom_object.t #! Center of object in rect camera coord.
	x_velo, y_velo, z_velo = calib.project_rect_to_velo(np.array([[x, y, z]]))[0]

	curr_azimuth, curr_elevation, curr_distance, _ = mlab.view()
	curr_x, curr_y, curr_z = spherical_to_cartesian(curr_azimuth, curr_elevation, curr_distance)

	ratio = 10 
	new_x = curr_x / ratio + x_velo * (ratio - 1) / ratio
	new_y = curr_y / ratio + y_velo * (ratio - 1) / ratio
	new_z = curr_z / ratio + z_velo * (ratio - 1) / ratio

	new_azimuth, new_elevation, new_distance = cartesian_to_spherical(new_x, new_y, new_z)

	mlab.view(
		azimuth=new_azimuth,
		elevation=new_elevation,
		distance=new_distance,
		focalpoint=[x_velo, y_velo, z_velo],
		figure=fig
	)

	return fig

def spherical_to_cartesian(azimuth, elevation, distance):
	pi_over_180 = np.pi / 180.0
	x = distance * np.sin(elevation * pi_over_180) * np.cos(azimuth * pi_over_180)
	y = distance * np.sin(elevation * pi_over_180) * np.sin(azimuth * pi_over_180)
	z = distance * np.cos(elevation * pi_over_180)

	return x, y, z

def cartesian_to_spherical(x, y, z):
	pi_below_180 = 180.0 / np.pi
	distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
	azimuth = pi_below_180 * np.arctan(y / x)
	elevation = pi_below_180 * np.arctan(np.sqrt(x ** 2 + y ** 2) / z)

	return azimuth, elevation, distance