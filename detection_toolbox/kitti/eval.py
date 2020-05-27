import io as sysio
import os
import time

import numba
import numpy as np
from scipy.interpolate import interp1d

# from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval
import math
from pathlib import Path

import numba
import numpy as np
from numba import cuda
from detection_toolbox.std import dprint


@cuda.jit(device=True, inline=True)
def iou_device(a, b):
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    width = max(right - left + 1, 0.)
    height = max(bottom - top + 1, 0.)
    interS = width * height
    Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return interS / (Sa + Sb - interS)


@cuda.jit()
def nms_kernel_v2(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(
        shape=(threadsPerBlock, 5), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if (tx < col_size):
        block_boxes[tx, 0] = dev_boxes[dev_box_idx, 0]
        block_boxes[tx, 1] = dev_boxes[dev_box_idx, 1]
        block_boxes[tx, 2] = dev_boxes[dev_box_idx, 2]
        block_boxes[tx, 3] = dev_boxes[dev_box_idx, 3]
        block_boxes[tx, 4] = dev_boxes[dev_box_idx, 4]
    cuda.syncthreads()
    if (cuda.threadIdx.x < row_size):
        cur_box_idx = threadsPerBlock * row_start + cuda.threadIdx.x
        # cur_box = dev_boxes + cur_box_idx * 5;
        i = 0
        t = 0
        start = 0
        if (row_start == col_start):
            start = tx + 1
        for i in range(start, col_size):
            if (iou_device(dev_boxes[cur_box_idx], block_boxes[i]) >
                    nms_overlap_thresh):
                t |= 1 << i
        col_blocks = ((n_boxes) // (threadsPerBlock) + (
            (n_boxes) % (threadsPerBlock) > 0))
        dev_mask[cur_box_idx * col_blocks + col_start] = t


@cuda.jit()
def nms_kernel(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if (tx < col_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if (tx < row_size):
        cur_box_idx = threadsPerBlock * row_start + tx
        # cur_box = dev_boxes + cur_box_idx * 5;
        t = 0
        start = 0
        if (row_start == col_start):
            start = tx + 1
        for i in range(start, col_size):
            iou = iou_device(dev_boxes[cur_box_idx * 5:cur_box_idx * 5 + 4],
                             block_boxes[i * 5:i * 5 + 4])
            if (iou > nms_overlap_thresh):
                t |= 1 << i
        col_blocks = ((n_boxes) // (threadsPerBlock) + (
            (n_boxes) % (threadsPerBlock) > 0))
        dev_mask[cur_box_idx * col_blocks + col_start] = t


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@numba.jit(nopython=True)
def nms_postprocess(keep_out, mask_host, boxes_num):
    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    remv = np.zeros((col_blocks), dtype=np.uint64)
    num_to_keep = 0
    for i in range(boxes_num):
        nblock = i // threadsPerBlock
        inblock = i % threadsPerBlock
        mask = np.array(1 << inblock, dtype=np.uint64)
        if not (remv[nblock] & mask):
            keep_out[num_to_keep] = i
            num_to_keep += 1
            # unsigned long long *p = &mask_host[0] + i * col_blocks;
            for j in range(nblock, col_blocks):
                remv[j] |= mask_host[i * col_blocks + j]
                # remv[j] |= p[j];
    return num_to_keep


def nms_gpu(dets, nms_overlap_thresh, device_id=0):
    """nms in gpu. 
    
    Args:
        dets ([type]): [description]
        nms_overlap_thresh ([type]): [description]
        device_id ([type], optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """

    boxes_num = dets.shape[0]
    keep_out = np.zeros([boxes_num], dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    boxes_host = dets[order, :]

    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    cuda.select_device(device_id)
    mask_host = np.zeros((boxes_num * col_blocks, ), dtype=np.uint64)
    blockspergrid = (div_up(boxes_num, threadsPerBlock),
                     div_up(boxes_num, threadsPerBlock))
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes_host.reshape([-1]), stream)
        mask_dev = cuda.to_device(mask_host, stream)
        nms_kernel[blockspergrid, threadsPerBlock, stream](
            boxes_num, nms_overlap_thresh, boxes_dev, mask_dev)
        mask_dev.copy_to_host(mask_host, stream=stream)
    # stream.synchronize()
    num_out = nms_postprocess(keep_out, mask_host, boxes_num)
    keep = keep_out[:num_out]
    return list(order[keep])


@cuda.jit(device=True, inline=True)
def trangle_area(a, b, c):
    return (
        (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@cuda.jit(device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                         int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@cuda.jit(device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = cuda.local.array((2, ), dtype=numba.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = cuda.local.array((2, ), dtype=numba.float32)
        vs = cuda.local.array((16, ), dtype=numba.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@cuda.jit(
    device=True,
    inline=True)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = cuda.local.array((2, ), dtype=numba.float32)
    B = cuda.local.array((2, ), dtype=numba.float32)
    C = cuda.local.array((2, ), dtype=numba.float32)
    D = cuda.local.array((2, ), dtype=numba.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@cuda.jit(
    device=True,
    inline=True)
def line_segment_intersection_v1(pts1, pts2, i, j, temp_pts):
    a = cuda.local.array((2, ), dtype=numba.float32)
    b = cuda.local.array((2, ), dtype=numba.float32)
    c = cuda.local.array((2, ), dtype=numba.float32)
    d = cuda.local.array((2, ), dtype=numba.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]

    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)

    if area_abc * area_abd >= 0:
        return False

    area_cda = trangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd

    if area_cda * area_cdb >= 0:
        return False
    t = area_cda / (area_abd - area_abc)

    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy
    return True


@cuda.jit(device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    eps = -1e-6
    return abab - abap >= eps and abap >= eps and adad - adap >= eps and adap >= eps
    


@cuda.jit(device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2, ), dtype=numba.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@cuda.jit(device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4, ), dtype=numba.float32)
    corners_y = cuda.local.array((4, ), dtype=numba.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i +
                1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@cuda.jit(device=True, inline=True)
def inter(rbbox1, rbbox2):
    corners1 = cuda.local.array((8, ), dtype=numba.float32)
    corners2 = cuda.local.array((8, ), dtype=numba.float32)
    intersection_corners = cuda.local.array((16, ), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])

    return area(intersection_corners, num_intersection)


@cuda.jit(device=True, inline=True)
def devRotateIoU(rbox1, rbox2):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    return area_inter / (area1 + area2 - area_inter)


@cuda.jit()
def rotate_nms_kernel(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 6, ), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if (tx < col_size):
        block_boxes[tx * 6 + 0] = dev_boxes[dev_box_idx * 6 + 0]
        block_boxes[tx * 6 + 1] = dev_boxes[dev_box_idx * 6 + 1]
        block_boxes[tx * 6 + 2] = dev_boxes[dev_box_idx * 6 + 2]
        block_boxes[tx * 6 + 3] = dev_boxes[dev_box_idx * 6 + 3]
        block_boxes[tx * 6 + 4] = dev_boxes[dev_box_idx * 6 + 4]
        block_boxes[tx * 6 + 5] = dev_boxes[dev_box_idx * 6 + 5]
    cuda.syncthreads()
    if (tx < row_size):
        cur_box_idx = threadsPerBlock * row_start + tx
        # cur_box = dev_boxes + cur_box_idx * 5;
        t = 0
        start = 0
        if (row_start == col_start):
            start = tx + 1
        for i in range(start, col_size):
            iou = devRotateIoU(dev_boxes[cur_box_idx * 6:cur_box_idx * 6 + 5],
                               block_boxes[i * 6:i * 6 + 5])
            # print('iou', iou, cur_box_idx, i)
            if (iou > nms_overlap_thresh):
                t |= 1 << i
        col_blocks = ((n_boxes) // (threadsPerBlock) + (
            (n_boxes) % (threadsPerBlock) > 0))
        dev_mask[cur_box_idx * col_blocks + col_start] = t


def rotate_nms_gpu(dets, nms_overlap_thresh, device_id=0):
    """nms in gpu. WARNING: this function can provide right result 
    but its performance isn't be tested
    
    Args:
        dets ([type]): [description]
        nms_overlap_thresh ([type]): [description]
        device_id ([type], optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """
    dets = dets.astype(np.float32)
    boxes_num = dets.shape[0]
    keep_out = np.zeros([boxes_num], dtype=np.int32)
    scores = dets[:, 5]
    order = scores.argsort()[::-1].astype(np.int32)
    boxes_host = dets[order, :]

    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    cuda.select_device(device_id)
    # mask_host shape: boxes_num * col_blocks * sizeof(np.uint64)
    mask_host = np.zeros((boxes_num * col_blocks, ), dtype=np.uint64)
    blockspergrid = (div_up(boxes_num, threadsPerBlock),
                     div_up(boxes_num, threadsPerBlock))
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes_host.reshape([-1]), stream)
        mask_dev = cuda.to_device(mask_host, stream)
        rotate_nms_kernel[blockspergrid, threadsPerBlock, stream](
            boxes_num, nms_overlap_thresh, boxes_dev, mask_dev)
        mask_dev.copy_to_host(mask_host, stream=stream)
    num_out = nms_postprocess(keep_out, mask_host, boxes_num)
    keep = keep_out[:num_out]
    return list(order[keep])


@cuda.jit('(int64, int64, float32[:], float32[:], float32[:])', fastmath=False)
def rotate_iou_kernel(N, K, dev_boxes, dev_query_boxes, dev_iou):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    if (tx < col_size):
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if (tx < row_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i
            dev_iou[offset] = devRotateIoU(block_qboxes[i * 5:i * 5 + 5],
                                           block_boxes[tx * 5:tx * 5 + 5])


def rotate_iou_gpu(boxes, query_boxes, device_id=0):
    """rotated box iou running in gpu. 500x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation).
    
    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims, 
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): [description]
        device_id (int, optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def devRotateIoUEval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


@cuda.jit(
    '(int64, int64, float32[:], float32[:], float32[:], int32)',
    fastmath=False)
def rotate_iou_kernel_eval(N,
                           K,
                           dev_boxes,
                           dev_query_boxes,
                           dev_iou,
                           criterion=-1):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    if (tx < col_size):
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if (tx < row_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i
            dev_iou[offset] = devRotateIoUEval(block_qboxes[i * 5:i * 5 + 5],
                                               block_boxes[tx * 5:tx * 5 + 5],
                                               criterion)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """rotated box iou running in gpu. 8x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation).
    
    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims, 
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): [description]
        device_id (int, optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)


def get_mAP(prec):
    sums = 0
    for i in range(0, len(prec), 4):
        sums += prec[i]
    return sums / 11 * 100

#! scores is a 1d array of scores of matched dts.
#! num_gt is the total number of valid gt boxes in the dataset
#? Honestly, i'm not sure. It looks like it divies the space of scores into num_sample_pts parts
#? And returns the scores at each of the parts as the thresholds.
#? The scores I think are in decreasing order.
#? So it's not necessarily that the returned thresholds are 1.0, 0.9, 0.8, ... 0 if um_sample_pts = 11
#? But it's scores[len(scores) * 0 / 10], scores[len(scores) * 1 /10], .... i think...
#? Well all that really matters is that the thresholds returns at the end are length num_sample pts sorted decreasing
@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1] #! scores are in decreasing order.
    current_recall = 0
    thresholds = []

    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        #! I literally have no clue
        #! if the current recall is closer to right (bigger) recall than left, skip.
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))): 
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    # print(len(thresholds), len(scores), num_gt)
    return thresholds

#! gt_anno is anno for single image
#! current class is an index
#! difficulty is 0, 1, or 2
#? Note that images can be in multiple difficulty groups
#? I think Easy \subset Moderate \subset Hard
def clean_data(gt_anno, dt_anno, current_class, difficulty, extra_info_single):

    gt_extra_info_single, dt_extra_info_single, general_extra_info = extra_info_single

    MIN_HEIGHT = general_extra_info['MIN_HEIGHT']
    MAX_OCCLUSION = general_extra_info['MAX_OCCLUSION']
    MAX_TRUNCATION = general_extra_info['MAX_TRUNCATION']
    MAX_DISTANCE = general_extra_info['MAX_DISTANCE']
    MIN_POINTS_THRESHOLD = general_extra_info['MIN_POINTS_THRESHOLD'] #! int
    CLASS_NAMES = list(map(lambda s: s.lower(), general_extra_info['CLASS_NAMES']))
    #! Added later in eval.py, no need for user to specify
    curr_metric = general_extra_info['curr_metric'] #! 0 or 1 or 2











    # CLASS_NAMES = [
    #     'car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'car',
    #     'tractor', 'trailer'
    # ]
    # if os.environ["KITTI_EVAL_CHANGES"] == "0":
    #     MIN_HEIGHT = [40, 25, 25]
    #     MAX_OCCLUSION = [0, 1, 2]
    #     MAX_TRUNCATION = [0.15, 0.3, 0.5]
    
    # elif os.environ["KITTI_EVAL_CHANGES"] == "1":
    #     MAX_TRUNCATION = [0.99, 0.99, 0.99] # filter out stuff bigger than this.
    #     #! This is mostly to filter out all the truncation = 1s
    #     #! Note that there are indeed some things that have truncation < 1 and occlusion = 1. So, occlusion threshold
    #     #! starts at 0.99 to get rid of all fully occluded things (in diagnostics.py)
    #     #? For sanity, when we had occlusion = 0 and integer occlusions we ended up removing all the things with occlusion = 1


    #     if "," in os.environ["KITTI_EVAL_MIN_HEIGHT"]: #! if we passed in something like 40,20,0
    #         split = os.environ["KITTI_EVAL_MIN_HEIGHT"].split(",")
    #         MIN_HEIGHT = [int(s) for s in split]
    #         assert len(MIN_HEIGHT) == 3
    #     else: #! otherwise, just use a single value
    #         MIN_HEIGHT = [int(os.environ["KITTI_EVAL_MIN_HEIGHT"])] * 3
        
    #     max_occlusion = float(os.environ["KITTI_EVAL_MAX_OCCLUSION"])
    #     MAX_OCCLUSION = [max_occlusion] * 3

    # #! Special: includes max distance
    # elif os.environ["KITTI_EVAL_CHANGES"] == "2":
    #     # MAX_TRUNCATION = [0.99, 0.99, 0.99] # filter out stuff bigger than this.
    #     split = os.environ["KITTI_EVAL_MAX_TRUNCATION"].split(",")
    #     MAX_TRUNCATION = [float(s) for s in split]
    #     assert len(MAX_TRUNCATION) == 3

    #     split = os.environ["KITTI_EVAL_MIN_HEIGHT"].split(",")
    #     MIN_HEIGHT = [int(s) for s in split]
    #     assert len(MIN_HEIGHT) == 3

    #     split = os.environ["KITTI_EVAL_MAX_OCCLUSION"].split(",")
    #     MAX_OCCLUSION = [float(s) for s in split]
    #     assert len(MAX_OCCLUSION) == 3

    #     split = os.environ["KITTI_EVAL_MAX_DISTANCE"].split(",")
    #     MAX_DISTANCE = [int(s) for s in split]
    #     assert len(MAX_DISTANCE) == 3

    #     if CLASS_NAMES[current_class] == "cyclist" and os.environ["KITTI_EVAL_CYC_MAX_OCCLUSION"] != "": 
    #         split = os.environ["KITTI_EVAL_CYC_MAX_OCCLUSION"].split(",") #? Separate for cyclists
    #         MAX_OCCLUSION = [float(s) for s in split]
    #         assert len(MAX_OCCLUSION) == 3

    # else:
    #     raise Exception("Unsupported kitti eval changes")


    dc_bboxes, ignored_gt, ignored_dt = [], [], []


    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"]) #! number of boxes


    num_valid_gt = 0 #! Keeps the number of boxes that perfecty match the current class and fit the current difficulty

    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]


        valid_class = -1
        if (gt_name == current_cls_name): #! This bbox corresponds with the class we're doing rn
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name): #
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Undefined".lower() == gt_name): #! don't treat undefined as fp for cars
            valid_class = 0
        elif (current_cls_name == "Cyclist".lower() and "Motorcycle".lower() == gt_name):
            valid_class = 0
        else: #! no relationship with current class
            valid_class = -1


        ignore = False
        if (curr_metric == 0 or curr_metric == 1): #! 2d bbox or bev
            if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])
                or (gt_extra_info_single["distance"][i] > MAX_DISTANCE[difficulty])):

                ignore = True

        else: #! 3d
            if ((gt_extra_info_single["distance"][i] > MAX_DISTANCE[difficulty])
                or (gt_extra_info_single["num_points"][i] < MIN_POINTS_THRESHOLD)):
                # or gt_anno["occluded"][i] == 1): #? GET RID OF THIS TODO:

                ignore = True

        # if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
        #     or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
        #     or (height <= MIN_HEIGHT[difficulty])
        #     or (gt_extra_info_single["distance"][i] > MAX_DISTANCE[difficulty])
        #     or (curr_metric == 2 and gt_extra_info_single["num_points"] < MIN_POINTS_THRESHOLD)):

        #     ignore = True


        # if os.environ["KITTI_EVAL_CHANGES"] == "0" or os.environ["KITTI_EVAL_CHANGES"] == "1":
        #     # if gt_anno["occluded"][i] > 1.0:
        #     #     print(gt_anno["occluded"][i])
        #     if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
        #             or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
        #             or (height <= MIN_HEIGHT[difficulty])):
        #         # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
        #         ignore = True #! out of this difficult, ignore
        # #! Includes distance
        # elif os.environ["KITTI_EVAL_CHANGES"] == "2":
        #     if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
        #             or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
        #             or (height <= MIN_HEIGHT[difficulty])
        #             or (gt_anno["distance"][i] > MAX_DISTANCE[difficulty])):
        #         ignore = True
        # else:
        #     raise Exception("Unsupported kitti eval changes")

        

        #? Ignored_gt: 0 -> keep, don't ignore. 1 -> Ignore, but don't treat as FP. -1 -> Ignore, treat as FP.
        if valid_class == 1 and not ignore: #! all good to go, keep
            ignored_gt.append(0)
            num_valid_gt += 1
        #! Don't treat as false positive.
        #! Translation: If we have a detection that detections this, don't treat as part of the denominator for AP
        #! Two cases: If valid_class == 0, one of special FP classes
        #! ignore and valid_class == 1: If same class but harder. So if the model ends up predicting a harder box
        #! it's not penalized for it.
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        #! Unrelated 
        else:
            ignored_gt.append(-1)


        #! store don't care boxes so we can ignore detections in this area
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])


    for i in range(num_dt):
        #! Filter out irrelevant detection classes
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])

        #! This is a detection that's smaller than min_height
        #! This is the "new" change. They say it's because:
        '''
        ! suppose we're doing evalulation for easy. Apparently, if we have a bbox of size 39 pixels, we don't want
        ! it to be a FP for the 40 pixel easy box.
        ? Frankly I have no clue why.
        ! Bottom line is all detection boxes smaller than the current GT difficulty height are cut out.
        ! Note that this does still include detections of other classes
        '''
        # or (curr_metric == 2 and dt_extra_info_single["distance"][i] > MAX_DISTANCE[difficulty])
        if height < MIN_HEIGHT[difficulty] or (curr_metric == 2 and dt_extra_info_single["distance"][i] > MAX_DISTANCE[difficulty]) :
            ignored_dt.append(1)
        #! detection matches class, keep
        elif valid_class == 1:
            ignored_dt.append(0)
        #! mismatch, toss.
        else:
            ignored_dt.append(-1)


    '''
    ! num_valid_gt are the number of gt boxes in this image that: 1) are of the current class 2) fit the difficulty req.
    ! ignored_gt is a list of -1, 0, 1 length total # of GT boxes in this image.
    !   Ignored_gt: 0 -> keep, don't ignore. 1 -> Ignore, but don't treat as FP. -1 -> Ignore, treat as FP.
    ! ignored_dt is a list of -1, 0, 1 length total # of DT boxes in the image
    !   Ignored_dt: 0 -> keep (matches height and class), 1 -> Doesn't match height (too small), -1 -> class mismatch
    ! dc_bboxes: list of bounding boxes that are DontCare. These have value -1 in ignored_gt
    '''
    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
#! boxes: gt. query_boxes: detections.
#! returns an N x K matrix of ious.
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0] #! total number of gt boxes
    K = query_boxes.shape[0] #! total number of detections
    overlaps = np.zeros((N, K), dtype=boxes.dtype) #! type np float
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1])) #! area of the k-th dt box
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua #! yada yada i'm pretty sure this is just iou
                    #? Why does this calculate iou between boxes from different images too? 
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


# @numba.jit(nopython=True, parallel=True)
@numba.jit(nopython=True, parallel=False)
def d3_box_overlap_kernel(boxes,
                          qboxes,
                          rinc,
                          criterion=-1,
                          z_axis=1,
                          z_center=1.0):
    """
        z_axis: the z (height) axis.
        z_center: unified z (height) center of box.
    """
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                min_z = min(
                    boxes[i, z_axis] + boxes[i, z_axis + 3] * (1 - z_center),
                    qboxes[j, z_axis] + qboxes[j, z_axis + 3] * (1 - z_center))
                max_z = max(
                    boxes[i, z_axis] - boxes[i, z_axis + 3] * z_center,
                    qboxes[j, z_axis] - qboxes[j, z_axis + 3] * z_center)
                iw = min_z - max_z
                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1, z_axis=1, z_center=1.0):
    """kitti camera format z_axis=1.
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)
    rinc = rotate_iou_gpu_eval(boxes[:, bev_axes], qboxes[:, bev_axes], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion, z_axis, z_center)
    return rinc


#? It appears that if we're computing recall thresholds, we set compute_fp to be False.
@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    # gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size #! probably storing whether each detection was assigned to a gt.
    ignored_threshold = [False] * det_size #! array storing if detection score was below thresh
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True


    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0 #! Likely used for computing thresholds?
    delta = np.zeros((gt_size, ))
    delta_idx = 0


    #! My own code ----
    #! For each gt box, store whether it was -1 (ignored), 0 (false negative (unmatched)), 1 (true positive (matched))
    gt_box_type = np.full((gt_size, ), -1)
    #! For each dt box, store whether it was -1 (irrelevant), 0 (false positive (unmatched)), 1 (true positive (matched))
    #! Note that -1 could mean it was in don't care territory, was of a different class, etc
    dt_box_type = np.full((det_size, ), -1)

    #! loop over gt boxes
    for i in range(gt_size):
        if ignored_gt[i] == -1: #! Don't match completely irrelevant gt boxes
            continue
        
        det_idx = -1 #! the best detection for this gt stored
        valid_detection = NO_DETECTION #! Stores the max score so far of the detection.
        max_overlap = 0 #! The overlap for the best detection. "best" is highest overlap
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1): #! Don't match with completely irrelevant dt boxes
                continue
            if (assigned_detection[j]): #! if dt was already assigned, skip (assigned to a better gt)
                continue
            if (ignored_threshold[j]): #! if dt score is below threhsold, skip
                continue
            
            
            overlap = overlaps[j, i] #! Current overlap between this dt and this gt.
            dt_score = dt_scores[j] #! score of current dt

            #! If compute_fp is false, this is the only part that matters.
            #! Just finds the detection with sufficient overlap and highest score.
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score

            #! compute_fp is true. This means we're acutally doing the metric and not making thresholds.
            #! If overlap is sufficient, (better than previous overlap or previous was a det we don't care about)
            #!  and the current det is something we care about,
            #! Assign. Update overlap, det_idx. Note that we 1-out valid-detection since we dont rank by score.
            #! we 1-out it to show that we have assigned a det we care about.
            #! When compute_fp is true, we choose based on overlap
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False

            #! compute_fp is true.
            #! if overlap is sufficient, nothing was assigned yet, and it's a detection we don't care about
            #! We assign it. Note that we leave max_overlap as default so anything can overwrite this.
            #? One curious thing is that of the dets we don't care about, if we assign the first one, the enxt one
            #?  can't overwrite it because valid_detection != NO_DETECTION
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        #! If we couldn't match this gt to anything and it's something we care about, it's a false negative.
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
            gt_box_type[i] = 0

        #! If we did match this gt to something and
        #!  (gt is something we don't care about or det is something we don't care about)
        #! We assign it. Why? probably because if we don't assign it, it'll be a false positive later (unassigned det)
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1 )):
            assigned_detection[det_idx] = True
            gt_box_type[i] = -1
            dt_box_type[det_idx] = -1

        #! If we did match this gt to something and
        #!  the remaining condition is: (gt is something we care about and det is something we care about)
        #! It's a good match! true positive. 
        #! Here, we also (basically) append the det score to the end of thersholds
        #! Then, assign detection True
        elif valid_detection != NO_DETECTION:
            # only a tp add a threshold.
            gt_box_type[i] = 1
            dt_box_type[det_idx] = 1
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
        #! This should be when there is no detection and  gt is something we don't care about
        else:
            gt_box_type[i] = -1


    
    #? Note that so far, we have not used dc boxes. This is because they are only used for false positive calculation
    #?  as we haven't looked at unmatched detections yet. If an unmatched is inside don't care, we dont' count it as FP
    if compute_fp:
        #! loop through detections
        for i in range(det_size):
            #! When is a detection a false positive? Well, it is a false positive if it is:
            #!  NOT assigned to a gt, and
            #!  NOT of a different class, and
            #!  NOT of the same class but of a different size, and
            #!  NOT below the score threshold.
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
                dt_box_type[i] = 0 #! false positive!

        #! I believe this is the number of detections we harvest from don't care regions. We'll subtract it from fp.
        nstuff = 0
        #! Metric == 0 is 2d bbox
        if metric == 0:
            #! ious between dt boxes and dc boxes.
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    #! skip stuff that doesn't add to fp right above
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    #! if the overlap between the two is bigger than min_overlap
                    #! assign the detection to dc and add it to somethign we take away from fp.
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
                        dt_box_type[j] = -1 #! nvm, don't care about this one
        #! take nstuff away from fp.
        fp -= nstuff

        #TODO: annotate this
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1

    '''
    ! Let's have a conditionsl here. 
    ! if compute_fp:
    !   tp and fn are here. fp and similarity is nonsense.
    !   thresholds[:thresh_idx] == thresholds
    !   So basically, we have to tools to calculate recall and the scores of the matched dts.
    '''
    return tp, fp, fn, similarity, thresholds[:thresh_idx], (gt_box_type, dt_box_type)


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


# @numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             gt_box_types,
                             dt_box_types,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0

    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num +
                               gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _, (gt_box_type, dt_box_type) = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh, #! note that we pass in threshold we generated
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity

            if t == len(thresholds) - 1: #! just do for last threshold, since last one is smallest
                gt_box_types.append(gt_box_type)
                dt_box_types.append(dt_box_type)
                # print(thresh)
                # print(gt_box_type)
                # print(dt_box_type)
                # print(tp)
                # print(fp)
                # print(fn)
                # print(thresh == 0.05203958600759506)
                # assert 1 == 2
                # print(thresh)
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos,
                         dt_annos,
                         metric,
                         num_parts=50,
                         z_axis=1,
                         z_center=1.0):
    """fast iou algorithm. this function can be used independently to
    do result analysis. 
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py #! Actually a list of dicts
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
        z_axis: height axis. kitti camera use 1, lidar use 2.

    annos = [
        {
            'name': np.array(["Car", "Pedestrian", "Car", ...]),
            'truncated': np.array([0.1, 0.5, 1.0, ...]),
            'occluded': np.array([0, 1, 2, 3, 0, ...]),
            'alpha': np.array([-3.14, 3.14, 0.0, ...]),
            'bbox': np.array([
                [x1, y1, x2, y2],
                [left, top, right, bot],
                [0.0, 0.0, 385.0, 1280.0],
                ...
            ]), #! N x 4
            'dimensions': Don't care for now
            'location': Don't care for now
            'rotation_y': Don't care for now
            'score': np.array([
                0.1,
                0.3,
                ...
            ]) #! or all 0s for gt
        }
    ]
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0) #! a list of number of annotations in each file
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)

    num_examples = len(gt_annos)
    #! returns a list of numbers, which is num_examples split up into num_parts, with a remainder at the end.
    #! So (13, 2) would return [6, 6, 1] or something
    split_parts = get_split_parts(num_examples, num_parts) 

    parted_overlaps = []
    example_idx = 0
    bev_axes = list(range(3))
    bev_axes.pop(z_axis)
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part] #! basically chop up dataset into parts and iterate
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0: #! This is the 2D bbox part
            #! appears like it concats ALL the bounding boxes in the entire dataset into a super tall array
            #? Correction: PART of the dataset gt_annos_part and dt_annos_part
            #! shape (total number of bboxes, 4)
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            #! returns np array of shape (total # of gt boxes, total # of dt boxes)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes) 
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, bev_axes] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, bev_axes] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            loc = np.concatenate(
                [a["location"][:, bev_axes] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, bev_axes] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = bev_box_overlap(gt_boxes,
                                           dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = d3_box_overlap(
                gt_boxes, dt_boxes, z_axis=z_axis,
                z_center=z_center).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        
        parted_overlaps.append(overlap_part) #! ends up being a list of iou matrices b/n parts of the dataset
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part] #! these two aren't used...
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx +
                                   gt_box_num, dt_num_idx:dt_num_idx +
                                   dt_box_num]) #! slice out the part that corresponds to a single image
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    #! In the end, overlaps becomes a list of matrices. The list is length len(dt_annos) == len(gt_annos) (number of images)
    #! In each index is a iou matrix shape (number of gt boxes in that image, number of dt boxes in that image)
    #! parted_overlaps is overlap matrices over parts of dataset
    #! total_gt_num is list of number of boxes in each image.

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty, extra_info=None):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0

    gt_extra_info, dt_extra_info, general_extra_info = extra_info

    #! Loop through each image
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty, \
            extra_info_single=(gt_extra_info[i], dt_extra_info[i], general_extra_info))

        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        #! Ends up being a list of ignored_gts. etc...

        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        #! dc_boxes is a np array shape (# of don't care boxes IN THIS IMAGE, 4)
        #! Each row is a Don't Care bbox
        total_dc_num.append(dc_bboxes.shape[0])
        #! Number of don't care boxes. total_dc_num is a list of # of dc_boxes for each iamge
        dontcares.append(dc_bboxes)
        #! list of list of dc_boxes for each image

        total_num_valid_gt += num_valid_gt
        #! counter of total number of valid gt boxes

        #! bbox index is N x 4
        #! alpha index is N -> with the np.newaxis, it's N x 1
        #! So concat makes it an N x 5 with the "5" dimension being [x1, y1, x2, y2, alpha]
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)

        #! Similarly, N x 6. "6" dimension is [x1, y1, x2, y2, alpha, score]
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)

        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
        #! list lists of boxes
    #! I don't know why they do this instead of np.array. This just makes a length # of images array of 
    #! number of dontcare boxes in each image.
    total_dc_num = np.stack(total_dc_num, axis=0)
    '''
    ? All the arrays here have length = # of images in dataset
    ! gt_datas_list: list of (N x 5 arrays)
    ! dt_datas_list: list of (N x 6 arrays)
    ! ignored_gts: list of (length N array (vals -1, 0, or 1))
    ! ignored_dets: list of (length N array (vals -1, 0, or 1))
    ! dontcares: list of (# of don't care boxes in image x 4 arrays)
    ! total_dc_num: list of (# of don'tcare boxes in image value)
    ! total_num_valid_gt: total number of valid gts (int)
    '''
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
                  dt_annos,
                  current_classes,
                  difficultys,  # ! Is a tuple (0, 1, 2)
                  metric,       #! is 0 (bbox), 1 (bev), or 2 (3d)
                  min_overlaps, #! I believe this is shape (2, 3, num_classes) where:
                                #! 2 is just moderate thresholds, easy thresholds. DIFFERENT FROM BBOX DIFFICULTY
                                #! 3 is the different metrics (bbox, bev, 3d), 
                                #! num_classes is  for threshold for each class
                  compute_aos=False,
                  z_axis=1,
                  z_center=1.0,
                  num_parts=50,
                  extra_info=None):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist
        difficulty: int. eval difficulty, 0: easy, 1: normal, 2: hard # ! No, actually a tuple (0, 1, 2)
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlap: float, min overlap. official: 
            [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]] 
            format: [metric, class]. choose one from matrix above.
        num_parts: int. a parameter for fast calculate algorithm
        extra_info: a tuple (gt_extra_info, dt_extra_info, general_extra_info). Check get_kitti_eval for more details

    Returns:
        dict of recall, precision and aos
    """
    # print(len(gt_annos))
    # print(len(dt_annos))
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(
        dt_annos,
        gt_annos,
        metric,
        num_parts,
        z_axis=z_axis,
        z_center=z_center)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    #! In the end, overlaps becomes a list of matrices. The list is length len(dt_annos) == len(gt_annos) (number of images)
    #! In each index is a iou matrix shape (number of gt boxes in that image, number of dt boxes in that image)
    #! parted_overlaps is overlap matrices over parts of dataset
    #! total_gt_num is list of number of boxes in each image.

    N_SAMPLE_PTS = 41

    num_minoverlap = len(min_overlaps) #! moderate, or easy
    num_class = len(current_classes)
    num_difficulty = len(difficultys)

    #! A single point would be the precision for a class, a certain bbox difficulty, the type of threhsolds (mod or easy)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    #! To store gt_box_types, dt_box_types for each class, per difficulty, per num_minoverlap
    gt_box_typess = np.full((num_class, num_difficulty, num_minoverlap), None, dtype=object)
    dt_box_typess = np.full((num_class, num_difficulty, num_minoverlap), None, dtype=object)


    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    all_thresholds = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])


    #! Per class
    for m, current_class in enumerate(current_classes):
        #! Per difficulty
        for l, difficulty in enumerate(difficultys):
            gt_extra_info, dt_extra_info, general_extra_info = extra_info
            general_extra_info['curr_metric'] = metric #! pass on which metric we're doing
            extra_info = (gt_extra_info, dt_extra_info, general_extra_info)

            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty, extra_info=extra_info)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            '''
            ? All the arrays here have length = # of images in dataset
            ! gt_datas_list: list of (N x 5 arrays)
            ! dt_datas_list: list of (N x 6 arrays)
            ! ignored_gts: list of (length N array (vals -1, 0, or 1))
            ! ignored_dets: list of (length N array (vals -1, 0, or 1))
            ! dontcares: list of (# of don't care boxes in image x 4 arrays)
            ! total_dc_num: list of (# of don'tcare boxes in image value)
            ! total_num_valid_gt: total number of valid gts (int)
            '''

            #! Runs twice, first for moderate overall difficulty setting, then easy.
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []

                #! Loop over images in dataset. So single image at a time.
                for i in range(len(gt_annos)):

                    rets = compute_statistics_jit(
                        overlaps[i], #! iou values b/n gt and dt for single image
                        gt_datas_list[i], #! N x 5 array
                        dt_datas_list[i], #! N x 6 array
                        ignored_gts[i], #! Length N array of -1, 0, 1
                        ignored_dets[i], #! Length N array of -1, 0, 1
                        dontcares[i], #! Length number of don't care boxes x 4
                        metric, #! 0, 1, or 2 (bbox, bev, 3d)
                        min_overlap=min_overlap, #! float minimum iou threshold for positive
                        thresh=0.0, #! ignore dt with scores below this.
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds, _ = rets #! Don't carea bout gt_box_type, dt_box_type here
                    thresholdss += thresholds.tolist()
                #! A 1d array of scores of matched dts.
                thresholdss = np.array(thresholdss)

                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                #! N_SAMPLE_PTS length array of scores, decreasing. these are the thresholds

                all_thresholds[m, l, k, :len(thresholds)] = thresholds
                #! Threshold for each combo
                #! storing 4 "things" for each threshold.
                #? [tp, fp, fn, similarity]
                pr = np.zeros([len(thresholds), 4])
                
                #! My addition - stores information about gt/dt boxes (whether ignored, fn, tn, fp)
                #! ends up being a list of np arrays
                gt_box_types = []
                dt_box_types = []
                #! Again, we're splitting up the dataset into parts and running it in.
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        gt_box_types=gt_box_types,
                        dt_box_types=dt_box_types,
                        compute_aos=compute_aos)
                    idx += num_part

                gt_box_typess[m, l, k] = gt_box_types
                dt_box_typess[m, l, k] = dt_box_types
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1]) #! true pos / (true pos + false pos)
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2]) #! true pos / (true pos + false neg)
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1) #? INTERPOLATES AND FLIPS THE ORDER!!! 
                                                         #? NOW ITS IN ORDER OF INCREASING RECALL
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)

    ret_dict = {
        "recall": recall, # [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
        "precision": precision, #? Order of INCREASING RECALL, so precision DECREASES (as we would expect in a graph)
        "orientation": aos,
        "thresholds": all_thresholds,
        "min_overlaps": min_overlaps,
        "gt_box_typess": gt_box_typess,
        "dt_box_typess": dt_box_typess
    }
    return ret_dict


def get_mAP_v2(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def do_eval_v2(gt_annos,
               dt_annos,
               current_classes,
               min_overlaps,
               compute_aos=False,
               difficultys=(0, 1, 2),
               z_axis=1,
               z_center=1.0):
    # min_overlaps: [num_minoverlap, metric, num_class]
    ret = eval_class(
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        0,
        min_overlaps,
        compute_aos,
        z_axis=z_axis,
        z_center=z_center)
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    mAP_bbox = get_mAP_v2(ret["precision"])
    mAP_aos = None
    if compute_aos:
        mAP_aos = get_mAP_v2(ret["orientation"])
    ret = eval_class(
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        1,
        min_overlaps,
        z_axis=z_axis,
        z_center=z_center)
    mAP_bev = get_mAP_v2(ret["precision"])
    ret = eval_class(
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        2,
        min_overlaps,
        z_axis=z_axis,
        z_center=z_center)
    mAP_3d = get_mAP_v2(ret["precision"])
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos

def do_eval_v3(gt_annos,
               dt_annos,
               current_classes,
               min_overlaps,
               compute_aos=False,
               difficultys=(0, 1, 2),
               z_axis=1,
               z_center=1.0,
               extra_info=None):
    # min_overlaps: [num_minoverlap, metric, num_class]
    types = ["bbox", "bev", "3d"]
    metrics = {}
    for i in range(3):
        dprint("Currently on {}".format(types[i]))
        ret = eval_class(
            gt_annos,
            dt_annos,
            current_classes,
            difficultys,
            metric=i,
            min_overlaps=min_overlaps,
            compute_aos=compute_aos,
            z_axis=z_axis,
            z_center=z_center,
            extra_info=extra_info)
        metrics[types[i]] = ret
    return metrics


def do_coco_style_eval(gt_annos,
                       dt_annos,
                       current_classes,
                       overlap_ranges,
                       compute_aos,
                       z_axis=1,
                       z_center=1.0):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval_v2(
        gt_annos,
        dt_annos,
        current_classes,
        min_overlaps,
        compute_aos,
        z_axis=z_axis,
        z_center=z_center)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()

'''
Args:
    gt_annos: list of annotation dicts. Reference kitti_label.py for format
    dt_annos: list of annotation dicts. Reference kitti_label.py for format
    extra_info: tuple (gt_extra_info, dt_extra_info, general_extra_info). 
        gt_extra_info and dt_extra_info must be lists of dicts, either empty or of same length as gt_annos and dt_annos.
        general_extra_info is a dict w/ thresholds, current classes, etc
    current_classes: list of strings denoting classes we're evaluating
        ex: ["Car", "Pedestrian", "Cyclist"]
    IoUs: either
        (3, len(current_classes)) numpy array. IoUs[i, c] denotes the required overlap for a detection, for
            metric type i and class current_classes[c].
            metric type 0 -> bbox
                        1 -> bev
                        2 -> 3d
        or (# overall evaluation levels, 3, len(current_classes)). Same as above but the first dimension denotes the number
        of overall rounds of evaluation we do. For reference, the first case is puffed up to (1, 3, len(current_classes)).
    
'''
def kitti_eval(
    gt_annos,
    dt_annos,
    extra_info,
    current_classes,
    IoUs
):
    try:
        assert len(gt_annos) == len(dt_annos)
        assert len(extra_info[0]) == 0 or len(extra_info[0]) == len(gt_annos)
        assert len(extra_info[1]) == 0 or len(extra_info[1]) == len(dt_annos)
        # assert IoUs.shape == (3, len(current_classes))
    except Exception as e:
        print("gt_annos: {}, dt_annos: {}, gt_extra_info: {}, dt_extra_info: {}, current_classes: {}, IoUs: {}".format(
            len(gt_annos),
            len(dt_annos),
            len(extra_info[0]),
            len(extra_info[1]),
            len(current_classes),
            IoUs.shape
        ))
        raise e

    print("Doing evaluation over: \nclasses {}, \nIoUs {}".format(current_classes, IoUs))
    
    if IoUs.shape == 2:
        IoUs = np.expand_dims(IoUs, axis=0) #! Now (1, 3, len(current_classes)) to fit original format of min_overlaps
    class_to_name = {
        i: c for i, c in enumerate(current_classes)
    } #! int -> string
    current_classes = list(range(len(current_classes))) #! Change to numbers

    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    '''
    metrics: {
        'bbox': ...
        'bev': ...
        '3d': ...
    }

    ... = {
        "recall": recall, # [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
        "precision": precision, #? Order of INCREASING RECALL, so precision DECREASES (as we would expect in a graph)
        "orientation": aos,
        "thresholds": all_thresholds,
        "min_overlaps": min_overlaps,
        "gt_box_typess": gt_box_typess, # np array shape [num_class, num_difficulty, num_minoverlap], each elem is list
        "dt_box_typess": dt_box_typess
    }
    '''
    metrics = do_eval_v3(
        gt_annos,
        dt_annos,
        current_classes,
        min_overlaps=IoUs,
        compute_aos=compute_aos,
        extra_info=extra_info
    )
    dprint("Done generating metrics.")

    result = ''
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(IoUs.shape[0]):
            mAPbbox = get_mAP_v2(metrics["bbox"]["precision"][j, :, i])
            mAPbbox = ", ".join(f"{v:.2f}" for v in mAPbbox) # ! This is what we care about
            mAPbev = get_mAP_v2(metrics["bev"]["precision"][j, :, i])
            mAPbev = ", ".join(f"{v:.2f}" for v in mAPbev)
            mAP3d = get_mAP_v2(metrics["3d"]["precision"][j, :, i])
            mAP3d = ", ".join(f"{v:.2f}" for v in mAP3d)
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:".format(*IoUs[i, :, j])))
            result += print_str(f"bbox AP:{mAPbbox}")
            result += print_str(f"bev  AP:{mAPbev}")
            result += print_str(f"3d   AP:{mAP3d}")
            if compute_aos:
                mAPaos = get_mAP_v2(metrics["bbox"]["orientation"][j, :, i])
                mAPaos = ", ".join(f"{v:.2f}" for v in mAPaos)
                result += print_str(f"aos  AP:{mAPaos}")

    return result, metrics





def get_official_eval_result(gt_annos,
                             dt_annos,
                             current_classes,
                             difficultys=[0, 1, 2],
                             z_axis=1,
                             z_center=1.0):
    """
        gt_annos and dt_annos must contains following keys:
        [bbox, location, dimensions, rotation_y, score]
    """
    if os.environ["KITTI_EVAL_CHANGES"] == "0":
        print("Using Kitti Eval {}".format(os.environ["KITTI_EVAL_CHANGES"]))
        overlap_mod = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7]])
    # ! All that matters here is that Car required over lap is 0.7, etc
    elif os.environ["KITTI_EVAL_CHANGES"] == "1" or os.environ["KITTI_EVAL_CHANGES"] == "2":
        print("Using Kitti Eval {}".format(os.environ["KITTI_EVAL_CHANGES"]))
        CAR_IOU = float(os.environ["KITTI_EVAL_CAR_IOU"])
        PED_IOU = float(os.environ["KITTI_EVAL_PED_IOU"])
        CYC_IOU = float(os.environ["KITTI_EVAL_CYC_IOU"])
        overlap_mod = np.array(
            [[CAR_IOU, PED_IOU, CYC_IOU, CAR_IOU, PED_IOU, CAR_IOU, CAR_IOU, CAR_IOU]] * 3
        )

    

    overlap_easy = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5]])
    min_overlaps = np.stack([overlap_mod, overlap_easy], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'car',
        6: 'tractor',
        7: 'trailer',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    
    metrics = do_eval_v3(  # ! Now go to here
        gt_annos,
        dt_annos,
        current_classes,
        min_overlaps,
        compute_aos,
        difficultys,
        z_axis=z_axis,
        z_center=z_center)
    mAPbbox_store = None
    res_precision = None
    res_recall = None
    gt_box_types = None
    dt_box_types = None #TODO: probably should add some stuff to make these work for 3d boxes in the future
    #TODO: Does not work for multiple classes, nor does precision or recall. 
    #? Note that the return format is different from precision or recall. for p/r, we don't index into difficulty
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        # precision is shape [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
        for i in range(min_overlaps.shape[0]):
            mAPbbox = get_mAP_v2(metrics["bbox"]["precision"][j, :, i])
            if mAPbbox_store is None:
                mAPbbox_store = [
                    get_mAP_v2(metrics["bbox"]["precision"][c, :, i])
                    for c in range(len(current_classes))
                ] #! Just stores the first overlap_mod metrics, for all classes

                res_precision = metrics["bbox"]["precision"][:, :, i] #! # classes x 3 x 41 (difficulty x points)
                res_recall = metrics["bbox"]["recall"][:, :, i]
                gt_box_types = metrics["bbox"]["gt_box_typess"][:, :, i] #! #classes x difficulty, with values being (list of numpy arrays)
                dt_box_types = metrics["bbox"]["dt_box_typess"][:, :, i]
            mAPbbox = ", ".join(f"{v:.2f}" for v in mAPbbox) # ! This is what we care about
            mAPbev = get_mAP_v2(metrics["bev"]["precision"][j, :, i])
            mAPbev = ", ".join(f"{v:.2f}" for v in mAPbev)
            mAP3d = get_mAP_v2(metrics["3d"]["precision"][j, :, i])
            mAP3d = ", ".join(f"{v:.2f}" for v in mAP3d)
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str(f"bbox AP:{mAPbbox}")
            result += print_str(f"bev  AP:{mAPbev}")
            result += print_str(f"3d   AP:{mAP3d}")
            if compute_aos:
                mAPaos = get_mAP_v2(metrics["bbox"]["orientation"][j, :, i])
                mAPaos = ", ".join(f"{v:.2f}" for v in mAPaos)
                result += print_str(f"aos  AP:{mAPaos}")


    return result, mAPbbox_store, (res_precision, res_recall), (gt_box_types, dt_box_types)


def get_coco_eval_result(gt_annos,
                         dt_annos,
                         current_classes,
                         z_axis=1,
                         z_center=1.0):
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'car',
        6: 'tractor',
        7: 'trailer',
    }
    class_to_range = {
        0: [0.5, 1.0, 0.05],
        1: [0.25, 0.75, 0.05],
        2: [0.25, 0.75, 0.05],
        3: [0.5, 1.0, 0.05],
        4: [0.25, 0.75, 0.05],
        5: [0.5, 1.0, 0.05],
        6: [0.5, 1.0, 0.05],
        7: [0.5, 1.0, 0.05],
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
        5: [0.5, 0.95, 10],
        6: [0.5, 0.95, 10],
        7: [0.5, 0.95, 10],
    }

    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos,
        dt_annos,
        current_classes,
        overlap_ranges,
        compute_aos,
        z_axis=z_axis,
        z_center=z_center)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, "
                             f"{mAPbbox[j, 1]:.2f}, "
                             f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, "
                                 f"{mAPaos[j, 1]:.2f}, "
                                 f"{mAPaos[j, 2]:.2f}"))
    return result