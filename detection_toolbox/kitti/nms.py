'''
REQUIRES DETECTRON2 TO BE INSTALLED
https://github.com/facebookresearch/detectron2/blob/de098423c675dad38c23110407926ccf2919474d/detectron2/layers/nms.py#L101
'''

'''
Takes in a list of KittiLabel classes, and returns a list of KittiLabel classes after doing NMS with iou_threshold in bev
over all the KittiLabel classes together.
'''
def bev_nms(kitti_labels, iou_threshold):
    from detectron2.layers import batched_nms_rotated
    import numpy as np
    import torch

    #? NOTE: This might be different from elsewhere. However, does not matter because this CATEGORY_TO_IDX will never
    #? have any influence outside this function.
    CATEGORY_TO_IDX = {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2,
        "Motorcycle": 3,
        "Undefined": 4
    }
    boxes = []
    scores = []
    idxs = []

    #! For each label, maps its index in boxes (and scores, idx) to 
    #! -> a tuple (index of its parent (view) in kitti_labels, its index inside its kitti_label)
    overall_idx_to_label_idx = dict()
    curr_overall_idx = 0
    for kitti_label_idx, kitti_label in enumerate(kitti_labels):
        for label_idx, label in enumerate(kitti_label):
            # should be (x_ctr, y_ctr, width, height, angle_degrees)
            boxes.append([
                label.t[0], label.t[2], label.l, label.w, label.ry * (180.0 / np.pi)
            ])
            scores.append(label.score)
            idxs.append(CATEGORY_TO_IDX[label.type])

            overall_idx_to_label_idx[curr_overall_idx] = (kitti_label_idx, label_idx)
            curr_overall_idx += 1

    if len(boxes) == 0: #! No detections
        return kitti_labels

    boxes = torch.FloatTensor(boxes).to("cuda")
    scores = torch.FloatTensor(scores).to("cuda")
    idxs = torch.LongTensor(idxs).to("cuda")

    #! Performs per-class nms
    resulting_box_inds = batched_nms_rotated(
        boxes,
        scores,
        idxs,
        iou_threshold
    )
    keep_inds = [[] for i in range(len(kitti_labels))]
    for overall_idx in resulting_box_inds.cpu().tolist():
        kitti_label_idx, label_idx = overall_idx_to_label_idx[overall_idx]
        keep_inds[kitti_label_idx].append(label_idx)

    for kitti_label_idx, kitti_label in enumerate(kitti_labels):
        kitti_label.labels = [kitti_label.labels[i] for i in keep_inds[kitti_label_idx]]

    del boxes, scores, idxs, resulting_box_inds
    
    return kitti_labels