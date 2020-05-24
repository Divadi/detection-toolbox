import numpy as np
import cv2
import os

from .calibration import Calibration
from .kitti_label import KittiLabel


class Kitti(object):
    '''
    calib_dir: directory with all the calib files.
    image_dir: directory with folders image_0, image_1, ..., each with images
    gt_label_dir: directory with folders label_0, label_1, ..., each with gt labels
    dt_label_dir: Either:
        1) directory with folders label_0, label_1, ..., each with dt labels
        2) directory with dt labels 000000.txt, ..., 
    lidar_dir: directory with lidar files
    depthmap_dir: directory with folders depth_0, depth_1, ..., each with depthmaps
    '''
    def __init__(
        self,
        calib_dir=None,
        image_dir=None,
        gt_label_dir=None,
        dt_label_dir=None,
        lidar_dir=None,
        depthmap_dir=None
    ):
        self.calib_dir = calib_dir
        self.image_dir = image_dir
        self.gt_label_dir = gt_label_dir
        self.dt_label_dir = dt_label_dir
        self.lidar_dir = lidar_dir
        self.depthmap_dir = depthmap_dir

    #! Returns Calibration Object
    def get_calib(self, idx):
        if self.calib_dir is None:
            raise Exception("calib_dir not provided")
        else:
            return Calibration(os.path.join(self.calib_dir, str(idx).zfill(6) + ".txt"))

    def get_gt_label(self, view, idx):
        if self.gt_label_dir is None:
            raise Exception("gt_label_dir not provided")
        else:
            return KittiLabel(
                os.path.join(
                    os.path.join(self.gt_label_dir, "label_{}".format(view)), 
                    str(idx).zfill(6) + ".txt"
                ),
                view=view,
                gt=True
            )

    '''
    Either pass in a view = 0, 1, 2, 3, 4, then it goes to label_{view} inside self.dt_label_dir
    or pass in view = None, then it directly looks for {idx}.txt inside self.dt_label_dir
    '''
    def get_dt_label(self, view, idx):
        if self.dt_label_dir is None:
            raise Exception("dt_label_dir not provided")
        elif view is None:
            return KittiLabel(os.path.join(self.dt_label_dir, str(idx).zfill(6) + ".txt"))
        else:
            return KittiLabel(
                os.path.join(
                    os.path.join(self.dt_label_dir, "label_{}".format(view)), 
                    str(idx).zfill(6) + ".txt"
                ),
                view=view,
                gt=False
            )

    #! BGR Format
    def get_image(self, view, idx):
        if self.image_dir is None:
            raise Exception("image_dir not provided")
        else:
            return cv2.imread(
                os.path.join(
                    os.path.join(self.image_dir, "image_{}".format(view)), 
                    str(idx).zfill(6) + ".png"
                )
            )

    #! Returns lidar in velodyne format, n x 4
    def get_lidar(self, idx):
        if self.lidar_dir is None:
            raise Exception("lidar_dir not provided")
        else:
            return np.fromfile(os.path.join(self.lidar_dir, str(idx).zfill(6) + ".bin"), dtype=np.float32).reshape((-1, 4))

    #! Returns in BGR Format. Likely not ideal
    def get_depthmap(self, view, idx):
        if self.depthmap_dir is None:
            raise Exception("depthmap_dir not provided")
        else:
            return cv2.imread(
                os.path.join(
                    os.path.join(self.depthmap_dir, "image_{}".format(view)), 
                    str(idx).zfill(6) + ".png"
                )
            )
