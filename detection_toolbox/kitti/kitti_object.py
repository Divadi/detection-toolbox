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

    def get_gt_label(self, view, idx, filter_truncation_1=True):
        if self.gt_label_dir is None:
            raise Exception("gt_label_dir not provided")
        else:
            return KittiLabel(
                os.path.join(
                    os.path.join(self.gt_label_dir, "label_{}".format(view)), 
                    str(idx).zfill(6) + ".txt"
                ),
                view=view,
                gt=True,
                idx=idx,
                filter_truncation_1=filter_truncation_1
            )

    '''
    Either pass in a view = 0, 1, 2, 3, 4, then it goes to label_{view} inside self.dt_label_dir
    or pass in view = None, then it directly looks for {idx}.txt inside self.dt_label_dir
    '''
    def get_dt_label(self, view, idx):
        if self.dt_label_dir is None:
            raise Exception("dt_label_dir not provided")
        elif view is None:
            return KittiLabel(os.path.join(self.dt_label_dir, str(idx).zfill(6) + ".txt"), idx=idx)
        else:
            return KittiLabel(
                os.path.join(
                    os.path.join(self.dt_label_dir, "label_{}".format(view)), 
                    str(idx).zfill(6) + ".txt"
                ),
                view=view,
                gt=False,
                idx=idx
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

    #! Returns gt inds
    def get_gt_inds(self):
        if self.gt_label_dir is None:
            raise Exception("gt_label_dir not provided")
        else:
            gt_label_sub_dirs = os.listdir(self.gt_label_dir) #! label_0, label_1, ...
            gt_label_sub_dirs = list(filter(lambda s: "label_" in s, gt_label_sub_dirs))
            each_ind_set = []
            for gt_label_sub_dir in gt_label_sub_dirs:
                inds = set(list(map(lambda s: s[:-4], os.listdir(os.path.join(self.gt_label_dir, gt_label_sub_dir)))))
                each_ind_set.append(inds)
            for i in range(len(each_ind_set) - 1): #! Check that each sub dir has the same inds
                assert (each_ind_set[i] == each_ind_set[i + 1])
            
            return sorted(each_ind_set[0])
    
    #! Returns dt inds
    def get_dt_inds(self):
        if self.dt_label_dir is None:
            raise Exception("dt_label_dir not provided")
        else:
            dt_label_sub_dirs = os.listdir(self.dt_label_dir) #! label_0, label_1, ...
            dt_label_sub_dirs = list(filter(lambda s: "label_" in s, dt_label_sub_dirs))
            each_ind_set = []
            for dt_label_sub_dir in dt_label_sub_dirs:
                inds = set(list(map(lambda s: s[:-4], os.listdir(os.path.join(self.dt_label_dir, dt_label_sub_dir)))))
                each_ind_set.append(inds)
            for i in range(len(each_ind_set) - 1): #! Check that each sub dir has the same inds
                assert (each_ind_set[i] == each_ind_set[i + 1])
            
            return sorted(each_ind_set[0])

    def get_gt_annotated_image(self, view, idx):
        img = self.get_image(view, idx)
        gt = self.get_gt_label(view, idx)
        return gt.get_imgaug_bboxes().draw_on_image(img, size=3)

    def get_dt_annotated_image(self, view, idx):
        img = self.get_image(view, idx)
        dt = self.get_dt_label(view, idx)
        return dt.get_imgaug_bboxes().draw_on_image(img, size=3)

    '''
    Calculates num_points for each non-garbage label in GT view 
    Saves in save_dir/label_{view}/000000.txt, ...

    Only does so for inds in inds argument. If inds argument is None, gets them from get_dt_inds. 
    If self.dt_label_dir was not provided, goes through everything in gt.

    nonempty_ok should generally be false unless the views are being done in parallel.
    filter_truncation_1 should be True for regular kitti data.
    filter_truncation_1 should be False for simulation data, where each label file contains annotations for all objects
        with correct 3d coords but potentially wrong 2d bbox
    '''
    def generate_and_save_gt_num_points(self, view, save_dir, inds=None, tqdm=False, nonempty_ok=False, filter_truncation_1=True):
        if tqdm: from tqdm import tqdm
        from detection_toolbox.std import makedirs

        save_view_dir = os.path.join(save_dir, "label_{}".format(view))
        makedirs(save_view_dir, exist_ok=True, nonempty_ok=nonempty_ok)

        if inds is None:
            if self.dt_label_dir is None:
                inds = self.get_gt_inds()
            else:
                inds = self.get_dt_inds()

        if tqdm: inds = tqdm(inds)
        import time
        
        for ind in inds:
            gt = self.get_gt_label(view, ind, filter_truncation_1=filter_truncation_1)
            lidar_rect = self.get_calib(ind).project_velo_to_rect(self.get_lidar(ind)[:, :3])
            gt.compute_num_points_inside_3d_box(lidar_rect)
            gt.write_num_points_to_file(os.path.join(save_view_dir, str(ind).zfill(6) + ".txt"))

    '''
    Returns a tuple of gt_annos, dt_annos that can be put into "get_official_eval_result"
    Only considers/returns labels corresponding to view argument & inds in dt_label_dir/label_{view}
    '''
    def get_eval_annos(self, view, gt_filter_truncation_1=True, tqdm=False):
        if tqdm: from tqdm import tqdm
        dt_inds = self.get_dt_inds()
        
        gt_annos = []
        dt_annos = []

        if tqdm: dt_inds = tqdm(dt_inds)
        for dt_ind in dt_inds:
            gt_annos.append(self.get_gt_label(view, dt_ind, filter_truncation_1=gt_filter_truncation_1).get_annotation_dict())
            dt_annos.append(self.get_dt_label(view, dt_ind).get_annotation_dict())
        
        return gt_annos, dt_annos

    def get_eval_extra_info(self, view, gt_filter_truncation_1=True, num_points_dir=None, tqdm=False):
        if tqdm: from tqdm import tqdm
        dt_inds = self.get_dt_inds()

        gt_extra_info = []
        dt_extra_info = []
        
        if tqdm: dt_inds = tqdm(dt_inds)
        for dt_ind in dt_inds:
            gt = self.get_gt_label(view, dt_ind, filter_truncation_1=gt_filter_truncation_1)
            if num_points_dir is not None:
                num_points_file_path = os.path.join(num_points_dir, "label_0/{}.txt".format(dt_ind))
                gt._read_num_points_file_path(num_points_file_path=num_points_file_path)

            gt_extra_info.append(gt.get_extra_info())
            dt_extra_info.append(self.get_dt_label(view, dt_ind).get_extra_info())

        return gt_extra_info, dt_extra_info