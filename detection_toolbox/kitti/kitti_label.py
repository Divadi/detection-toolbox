import numpy as np
import imgaug
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class SingleLabel(object):
    # __slots__ = ['gt', 'garbage', 'type', 'truncation', 'occlusion', 'alpha',
    #              'xmin', 'ymin', 'xmax', 'ymax', 'box2d',
    #              'h', 'w', 'l', 't', 'ry', 'score',
    #              'distance', 'num_points']
    def __init__(self, label_line, gt):
        self.gt = gt
        self._process_label_line(label_line)
    
    def _process_label_line(self, label_line):
        self.garbage = False
        split = label_line.split(" ")
        if len(split) == 15:
            assert self.gt
        elif len(split) == 16:
            assert not self.gt
        else:
            self.garbage = True #! Garbage
            return
        
        
        split[1:] = [float(x) for x in split[1:]]

        self.type = split[0] # 'Car', 'Pedestrian', ...
        self.truncation = split[1] # truncated pixel ratio [0..1]
        self.occlusion = split[2] # For KITTI dataset, int 0, 1, 2, 3. For other, just float
        self.alpha = split[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = split[4] # left
        self.ymin = split[5] # top
        self.xmax = split[6] # right
        self.ymax = split[7] # bottom
        self.height2d = self.ymax - self.ymin
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = split[8] # box height
        self.w = split[9] # box width
        self.l = split[10] # box length (in meters)
        self.t = (split[11],split[12],split[13]) # location (x,y,z) in rect. camera coord.
        self.ry = split[14] # yaw angle (around Y-axis in rect. camera coordinates) [-pi..pi]

        if len(split) == 16:
            self.score = split[15] #! If DT, include score as well.

        self.distance = np.sqrt(self.t[0] ** 2 + self.t[1] ** 2 + self.t[2] ** 2)
        self.num_points = None #! has to be calculated later

    '''
    Returns corners of 3d bbox in rect camera coords. (8, 3) np array.
    '''
    def compute_box_3d(self):
        # compute rotational matrix around yaw axis
        R = roty(self.ry)    

        # 3d bounding box dimensions
        l = self.l
        w = self.w
        h = self.h
        
        # 3d bounding box corners
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
        
        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))

        corners_3d[0,:] = corners_3d[0,:] + self.t[0]
        corners_3d[1,:] = corners_3d[1,:] + self.t[1]
        corners_3d[2,:] = corners_3d[2,:] + self.t[2]

        return corners_3d.T

    '''
    Converts to line (original label) form, without newline character
    '''
    def to_line(self):
        line = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
            self.type, self.truncation, self.occlusion, self.alpha,
            self.xmin, self.ymin, self.xmax, self.ymax,
            self.h, self.w, self.l,
            self.t[0], self.t[1], self.t[2],
            self.ry
        )
        if hasattr(self, "score"):
            line += " {}".format(self.score)
        return line

    '''
    Returns 2d bbox in imgaug BoundingBox format
    '''
    def get_imgaug_bbox(self):
        return BoundingBox(
            x1=self.xmin, y1=self.ymin, x2=self.xmax, y2=self.ymax
        )


class KittiLabel(object):
    '''
    view is an int (could also be None if gt is False)
    gt is boolean
    idx is the {idx}.txt this label came from

    num_points_file_path is optional - it has # of points of the non-garbage labels, one on each line
    #! Careful: the order/number of this is dependent on which labels were actually inside KittiLabel when this file
    #! was generated. And which labels were inside KittiLabel depends on _read_label_from_file.
    #! NMS might remove some objects, but num_points only matters for gt, and we don't run NMS on gt. 
    '''
    def __init__(self, label_file_path, view, gt, idx, filter_truncation_1=True, num_points_file_path=None):
        self.view = view
        self.gt = gt
        self.idx = idx
        self._read_label_from_file(label_file_path, filter_truncation_1)

        if num_points_file_path is not None:
            self._read_label_from_file(num_points_file_path)
        
    #! if filter_truncation_1 is True, gets rid of labels with truncation = 1 (100%)
    def _read_label_from_file(self, label_file_path, filter_truncation_1=True):
        labels = [SingleLabel(line.strip(), self.gt) for line in open(label_file_path, "r").readlines()]

        labels = list(filter(lambda l: (not l.garbage) and (not filter_truncation_1 or l.truncation != 1), labels))

        self.labels = labels
    
    def _read_num_points_file_path(self, num_points_file_path):
        num_points = [int(line.strip()) for line in open(num_points_file_path, "r").readlines()]

        self.add_label_attribute("num_points", num_points)
        
        # assert len(num_points) == len(self.labels)

        # for label_ind, label in enumerate(self.labels):
        #     label.num_points = num_points[label_ind]

    #! remove labels with score < score_thresh
    #! Returns num removed
    def filter_score(self, score_thresh):
        assert not self.gt
        prev_num = len(self.labels)
        self.labels = list(filter(lambda l: l.score >= score_thresh, self.labels))
        return len(self.labels) - prev_num

    '''
    Writes contents of label to file
    Assumes that relevant folders are created
    returns passed in write_file_path
    '''
    def write_to_file(self, write_file_path):
        with open(write_file_path, "w+") as f:
            for label in self.labels:
                f.write(label.to_line() + "\n")
        return write_file_path

    '''
    Writes contents of label.num_points to file, one on each line.
    '''
    def write_num_points_to_file(self, write_file_path):
        with open(write_file_path, "w+") as f:
            for label in self.labels:
                f.write(str(label.num_points) + "\n")
        return write_file_path

    '''
    Computes # of points in each 3d box and saves them into self.labels' num_points attribute
    pc_rect is n x (3 or 4) numpy array of point cloud in rect. coords.
    '''
    def compute_num_points_inside_3d_box(self, pc_rect):
        from detection_toolbox.utils_3d.utils import extract_pc_in_box3d
        if pc_rect.shape[-1] != 3:
            pc_rect = pc_rect[:, :3]
        
        for label in self.labels:
            corners_3d_rect = label.compute_box_3d()
            pc_in_box, _ = extract_pc_in_box3d(pc_rect, corners_3d_rect)
            label.num_points = pc_in_box.shape[0]


    '''
    Returns a dict in format 
    {
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
    }
    '''
    def get_annotation_dict(self):
        res = dict()

        res['name'] = np.array([label.type for label in self.labels])
        res['truncated'] = np.array([label.truncation for label in self.labels])
        res['occluded'] = np.array([label.occlusion for label in self.labels])
        res['alpha'] = np.array([label.alpha for label in self.labels])
        res['bbox'] = np.array([label.box2d for label in self.labels]).reshape(-1, 4)
        #? l, h, w is not read-in order
        res['dimensions'] = np.array([[label.l, label.h, label.w] for label in self.labels]).reshape(-1, 3) 
        res['location'] = np.array([label.t for label in self.labels]).reshape(-1, 3)
        res['rotation_y'] = np.array([label.ry for label in self.labels])

        if not self.gt:
            res['score'] = np.array([label.score for label in self.labels])

        return res

    '''
    if gt:
        {
            'distance': [],
            'num_points': [], # fills in with 100k if they haven't been calculated yet
        }
    '''
    def get_extra_info(self):
        res = dict()
        if self.gt:
            res['distance'] = np.array([label.distance for label in self.labels])
            res['num_points'] = np.array([
                (label.num_points if label.num_points is not None else 100000)
                for label in self.labels
            ])
            return res
        else:
            res['distance'] = np.array([label.distance for label in self.labels])
            return res

    '''
    Returns bboxes in imgaug BoundingBoxesOnImage format
    TODO: have some mechanisms for gt vs not gt, filtering, labels, etc
    '''
    def get_imgaug_bboxes(self, img_shape=(720, 1920)):
        bboxes = [label.get_imgaug_bbox() for label in self.labels]
        return BoundingBoxesOnImage(bboxes, shape=img_shape)

    #! Adds attribute of attribute name to each label
    #! Attribute vals should be a list
    def add_label_attribute(self, attribute_name, attribute_vals):
        assert len(self.labels) == len(attribute_vals)

        for label, attribute_val in zip(self.labels, attribute_vals):
            setattr(label, attribute_name, attribute_val)


    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
