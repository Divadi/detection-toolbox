import numpy as np

class Calibration(object):
    def __init__(self, calib_file_path):
        self._read_calib_from_file(calib_file_path)


    def _read_calib_from_file(self, calib_file_path):
        lines = open(calib_file_path, "r").readlines()
        for line in lines:
            line = line.strip()
            key, val = line.split(":", 1)
            val = np.array(val.strip().split(" "), dtype=np.float32)

            if "P" == key[0]:
                setattr(self, key, val.reshape(3, 4))
            elif key == "Tr_velo_to_p2" or key == "Tr_velo_to_cam":
                self.V2C = val.reshape(3, 4)
                self.C2V = inverse_rigid_trans(self.V2C)
            elif key == "Tr_imu_to_velo":
                self.I2V = val.reshape(3, 4)
            elif key == "R0_rect":
                self.R0 = val.reshape(3, 3)
            else:
                raise Exception("Undefined key in calib file: {}, {}".format(key, calib_file_path))

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
 
    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, self.C2V.T)

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.dot(np.linalg.inv(self.R0), pts_3d_rect.T).T
    
    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
 
    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        ''' 
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect, view):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(getattr(self, "P" + str(view)))) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]
    
    def project_velo_to_image(self, pts_3d_velo, view):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # # =========================== 
    # # ------- 2d to 3d ---------- 
    # # =========================== 
    # def project_image_to_rect(self, uv_depth):
    #     ''' Input: nx3 first two channels are uv, 3rd channel
    #                is depth in rect camera coord.
    #         Output: nx3 points in rect camera coord.
    #     '''
    #     n = uv_depth.shape[0]
    #     x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
    #     y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
    #     pts_3d_rect = np.zeros((n,3))
    #     pts_3d_rect[:,0] = x
    #     pts_3d_rect[:,1] = y
    #     pts_3d_rect[:,2] = uv_depth[:,2]
    #     return pts_3d_rect
    

    # #! From Xinshuo's file
    # def img_to_rect(self, u, v, depth_rect):
    #     """
    #     :param u: (N)
    #     :param v: (N)
    #     :param depth_rect: (N)
    #     :return:
    #     """

    #     # split the extrinsics from the projection matrix
    #     proj_matrix = self.P.astype('float64')
    #     ref_proj = self.P2.astype('float64')
    #     intrinsics = ref_proj[:, :3]
    #     if self.view == 5: intrinsics[1, 2] = intrinsics[0, 2]
    #     extrinsics = np.matmul(np.linalg.inv(intrinsics), proj_matrix)          # 3 x 4

    #     # invert the extrinsics
    #     extrin = np.concatenate((extrinsics, np.array([0, 0, 0, 1]).reshape((1, 4))), axis=0)     # 4 x 4
    #     extrin = np.linalg.inv(extrin)
    #     extrin = extrin[:3, :]
        
    #     # project the points back to the 3D coordinate with respect to P2
    #     data_cam = self.get_intrisics_extrinsics(ref_proj)
    #     x = ((u - data_cam['cu']) * depth_rect) / data_cam['fu']
    #     y = ((v - data_cam['cv']) * depth_rect) / data_cam['fv']
    #     num_pts = x.shape[0]
    #     pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), \
    #         depth_rect.reshape(-1, 1), np.ones((num_pts, 1), dtype='float64')), axis=1)         # N x 4

    #     # rotate and translate to the 3D coordinate with respect to any camera
    #     pts_rect = np.matmul(pts_rect, extrin.transpose())

    #     return pts_rect
    
    # #! From Xinshuo's file
    # #! Changed: box specifies coordinates of passed depth_map in original image
    # #! used for when we took only a 2dbbox part of the image out
    # def depthmap_to_rect(self, depth_map, segmap=None, box=None, depth_limit=120):
    #     """
    #     :param depth_map: (H, W), depth_map
    #     :return:
    #     """
    #     if box is not None:
    #         xmin, ymin = box
    #     else:
    #         xmin = 0
    #         ymin = 0
        
    #     x_range = np.arange(0, depth_map.shape[1])
    #     y_range = np.arange(0, depth_map.shape[0])

    #     x_idxs, y_idxs = np.meshgrid(x_range, y_range)
    #     x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
    #     depth = depth_map[y_idxs, x_idxs]

    #     # remove the depth point which does not reflect back and has the maximum depth range
    #     valid_index = np.where(depth < depth_limit)[0].tolist()
    #     x_idxs, y_idxs, depth = x_idxs[valid_index], y_idxs[valid_index], depth[valid_index]

    #     x_idxs += xmin
    #     y_idxs += ymin #! Scale to proper positio in original image

    #     pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
    #     return pts_rect, x_idxs, y_idxs

    # def project_image_to_velo(self, uv_depth):
    #     pts_3d_rect = self.project_image_to_rect(uv_depth)
    #     return self.project_rect_to_velo(pts_3d_rect)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr