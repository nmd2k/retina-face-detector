# author: www.kaggle.com/kshitijpatil09/pytorch-mean-absolute-precision-calculation
import torch
import numpy as np
from utils.box_utils import jaccard

def get_mappings(iou_mat):
    mappings = torch.zeros_like(iou_mat)
    gt_count, pr_count = iou_mat.shape
    
    #first mapping (max iou for first pred_box)
    if not iou_mat[:,0].eq(0.).all():
        # if not a zero column
        mappings[iou_mat[:,0].argsort()[-1],0] = 1

    for pr_idx in range(1,pr_count):
        # Sum of all the previous mapping columns will let 
        # us know which gt-boxes are already assigned
        not_assigned = torch.logical_not(mappings[:,:pr_idx].sum(1)).long()

        # Considering unassigned gt-boxes for further evaluation 
        targets = not_assigned * iou_mat[:,pr_idx]

        # If no gt-box satisfy the previous conditions
        # for the current pred-box, ignore it (False Positive)
        if targets.eq(0).all():
            continue

        # max-iou from current column after all the filtering
        # will be the pivot element for mapping
        pivot = targets.argsort()[-1]
        mappings[pivot,pr_idx] = 1
    return mappings

def calculate_map(gt_boxes,pr_boxes,scores,thresh=0.5,form='pascal_voc'):
    # sorting
    pr_boxes = pr_boxes[scores[:, 1].argsort().flip(-1)]
    iou_mat = jaccard(gt_boxes,pr_boxes)
    
    # thresholding
    iou_mat = iou_mat.where(iou_mat>thresh, torch.tensor(0.).cuda())
    
    mappings = get_mappings(iou_mat)
    
    # mAP calculation
    tp = mappings.sum()
    fp = mappings.sum(0).eq(0).sum()
    fn = mappings.sum(1).eq(0).sum()
    mAP = tp / (tp+fp+fn)
    
    return mAP.cpu().numpy()

def calculate_running_map(targets, prediction):
    map_5, map_5_95 = [] , []

    loc_data, conf_data, landm_data = prediction
    num = loc_data.size(0)
    
    for idx in range(num):
        truths = targets[idx][:, :4].data

        for thresh in np.arange(0.5, 0.95, 0.05):
            map = calculate_map(truths, loc_data[idx], conf_data[idx], thresh)
            map_5_95.append(map)

            if thresh == 0.5:
                map_5.append(map)

    map_5    = np.array(map_5).mean()
    map_5_95 = np.array(map_5_95).mean()
    
    return map_5, map_5_95