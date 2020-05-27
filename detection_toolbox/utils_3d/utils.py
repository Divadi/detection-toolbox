import os
import sys
import numpy as np
import cv2

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

'''
pc is point cloud, box3d are corners
'''
def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


def get_lidar_in_image_fov(pc_rect, calib, view, xmin, ymin, xmax, ymax,
                        return_more=False, clip_distance=.1):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_rect_to_image(pc_rect)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)

    clip_filter = None
    if view == 0:
        clip_filter = pc_rect[:, 2] < -clip_distance
    elif view == 1:
        clip_filter = pc_rect[:, 0] < -clip_distance
    elif view == 2:
        clip_filter = pc_rect[:, 2] > clip_distance
    elif view == 4:
        clip_filter = pc_rect[:, 0] > clip_distance

    
    fov_inds = fov_inds & clip_filter
    imgfov_pc_rect = pc_rect[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo