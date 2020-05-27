import numpy as np
from tqdm import tqdm
import time
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

'''
For details, reference vis3d draw_3d_boxes_from_objects_advanced
'''
def draw_2d_boxes_from_objects_advanced(
	objects,
	calib,
	img,
	default_color=(0, 1, 0), #! default color is green
	color_func=None,
	text_func=None
):
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
			if text is "":
				text = None
		else:
			text = None

		bbox = label.get_imgaug_bbox()
		bbox.label = text

		if color not in color_dict.keys():
			color_dict[color] = {
				"boxes": []
			}
		
		color_dict[color]['boxes'].append(bbox)

	for color, val in color_dict.items():
		bboxes = BoundingBoxesOnImage(val['boxes'], shape=img.shape[:2])
		#! flip color tuple b/c img is bgr and provided color is rgb
		img = bboxes.draw_on_image(img, color=tuple(int(i * 255) for i in color)[::-1], size=3) 

	return img