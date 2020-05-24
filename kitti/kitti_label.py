import numpy as np

class SingleLabel(object):
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
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = split[8] # box height
        self.w = split[9] # box width
        self.l = split[10] # box length (in meters)
        self.t = (split[11],split[12],split[13]) # location (x,y,z) in rect. camera coord.
        self.ry = split[14] # yaw angle (around Y-axis in rect. camera coordinates) [-pi..pi]

        if len(split) == 16:
            self.score = split[15] #! If DT, include score as well.

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
        

class KittiLabel(object):
    '''
    view is an int (could also be None if gt is False)
    gt is boolean
    '''
    def __init__(self, label_file_path, view, gt):
        self.view = view
        self.gt = gt
        self._read_label_from_file(label_file_path)
        
    #! if filter_truncation_1 is True, gets rid of labels with truncation = 1 (100%)
    def _read_label_from_file(self, label_file_path, filter_truncation_1=True):
        labels = [SingleLabel(line.strip(), self.gt) for line in open(label_file_path, "r").readlines()]

        labels = list(filter(lambda l: (not l.garbage) and (not filter_truncation_1 or l.truncation != 1), labels))

        self.labels = labels

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
