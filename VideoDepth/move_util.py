import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.ops.boxes import batched_nms
from torch.nn import functional as F
from segment_anything.utils.amg import batched_mask_to_box, calculate_stability_score
# from utils import iou, is_bg_mask, save_indexed, update_iousummary, filter_data, hard_thres
from scipy.optimize import linear_sum_assignment

def is_bg_mask(
    masks_fil, thres = 0.5
):  
    masks_edge = (masks_fil[:, 0].mean(-1) + masks_fil[:, -1].mean(-1) + masks_fil[:, :, 0].mean(-1) + masks_fil[:, :, -1].mean(-1))/4 
    return masks_edge > thres


def iou(masks, gt, thres=0.5, emp=True):
    """ IoU predictions """
    if isinstance(masks, torch.Tensor): # for tensor inputs
        masks = (masks>thres).float()
        gt = (gt>thres).float()
        intersect = (masks * gt).sum(dim=[-2, -1])
        union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
        empty = (union < 1e-6).float()
        iou = torch.clip(intersect/(union + 1e-12) + empty, 0., 1.)
        return iou
    else: # for numpy inputs
        masks = (masks>thres)
        gt = (gt>thres)
        intersect = (masks * gt).sum((-1,-2))
        union = masks.sum((-1,-2)) + gt.sum((-1,-2)) - intersect
        empty = (union < 1e-6) if emp else 0
        iou = np.clip(intersect/(union + 1e-12) + empty, 0., 1.)
        return iou



def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png """
    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')
    

def save_indexed(filename, img):
    """ Save image with given colour palette """
    color_palette = np.array([[0,0,0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [191, 0, 0], [64, 128, 0]]).astype(np.uint8)
    imwrite_indexed(filename, img, color_palette)


def is_box_near_image_edge(
    boxes, orig_box, atol: float = 20.0
):
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    return torch.any(near_image_edge, dim=1)


def filter_data(data_list, condition, is_idx = False):
    """Filter data according to condition provided"""
    """ is_idx = True represents that the conditions are given as the index in the tensor"""
    """ is_idx = False represents that the condtions are binary masks"""
    data_fil_list = []
    for i, data in enumerate(data_list):
        if is_idx:
            if condition is None:
                data_fil = torch.zeros_like(data[0:1], device=data.device)
            else:
                data_fil = data[torch.as_tensor(condition, device=data.device)]
        else:
            data_fil_tmp = [a for i, a in enumerate(data) if condition[i]]
            if len(data_fil_tmp) == 0:
                data_fil = torch.zeros_like(data[0:1], device=data.device)
            else:
                data_fil = torch.stack(data_fil_tmp, 0)
        data_fil_list.append(data_fil)
    return data_fil_list

def hard_thres(masks, ious, output_savemask = False):
    """ Hard thresholding (overlaying) the masks according to IoUs (Scores)"""
    masks_np = masks.detach().cpu().numpy()
    ious_np = ious.detach().cpu().numpy()
    saveidxs_np = np.arange(masks.shape[0])
    ious_rank = np.argsort(ious_np)

    output_mask = np.copy(masks_np[0]) * 0.
    for score_idx in ious_rank:
        output_mask = output_mask * (1 - masks_np[score_idx]) + masks_np[score_idx] * (saveidxs_np[score_idx] + 1)
    mask_out = np.clip(output_mask, 0, masks.shape[0])

    masks_out_torch = []
    for obj_idx in range(1, masks.shape[0] + 1):
        mask_torch = torch.from_numpy(mask_out == obj_idx).float().cuda()
        masks_out_torch.append(mask_torch)
    masks_out_torch = torch.stack(masks_out_torch, 0)
    if output_savemask: # Optionally output the mask for saving
        return masks_out_torch, mask_out
    else:
        return masks_out_torch
       
def update_iousummary(masks_hung, masks_nonhung, anno, num_obj, path, iou_summary, save_path = None):
    # Updating the performance
    for obj_idx in range(1, num_obj + 1):
        iou_summary[os.path.dirname(path[0])][obj_idx - 1].append(iou(masks_hung[obj_idx - 1], anno[0, obj_idx - 1]).item())
    if save_path is not None:
        save_path_hung = os.path.join(save_path, "hung")
        os.makedirs(os.path.dirname(os.path.join(save_path_hung, path[0])), exist_ok = True)
        save_path_nonhung = os.path.join(save_path, "nonhung")
        os.makedirs(os.path.dirname(os.path.join(save_path_nonhung, path[0])), exist_ok = True)
        # Saving Hungarian matched masks
        masks_hung_np = masks_hung.detach().cpu().numpy()
        saved_mask_hung = np.copy(masks_hung_np[0]) * 0.
        for save_idx in range(masks_hung_np.shape[0]):
            saved_mask_hung = saved_mask_hung * (1 - masks_hung_np[save_idx]) + masks_hung_np[save_idx] * (save_idx + 1)
        saved_mask_hung = np.clip(saved_mask_hung, 0, masks_hung_np.shape[0])
        save_indexed(os.path.join(save_path_hung, path[0]), saved_mask_hung.astype(np.uint8))
        # Saving Non-Hungarian masks
        masks_nonhung_np = masks_nonhung.detach().cpu().numpy()
        saved_mask_nonhung = np.copy(masks_nonhung_np[0]) * 0.
        for save_idx in range(masks_nonhung_np.shape[0]):
            saved_mask_nonhung = saved_mask_nonhung * (1 - masks_nonhung_np[save_idx]) + masks_nonhung_np[save_idx] * (save_idx + 1)
        saved_mask_nonhung = np.clip(saved_mask_nonhung, 0, masks_nonhung_np.shape[0])
        save_indexed(os.path.join(save_path_nonhung, path[0]), saved_mask_nonhung.astype(np.uint8))
    return iou_summary

def remove_overlapping_masks(masks, filter=False):
    result_mask = masks.copy()
    for i in range(result_mask.shape[0] - 1):
        result_mask[i+1:] = np.logical_and(result_mask[i+1:], np.logical_not(masks[i]))
    result_mask = np.clip(result_mask, 0 ,1)
    return result_mask

def warp_flow(curImg, flow):
    H, W = np.shape(curImg)
    flow = cv2.resize(flow, (W, H))
    h, w = flow.shape[:2]
    flow = flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    prevImg = cv2.remap(curImg, flow, None, cv2.INTER_LINEAR)
    return prevImg

def hungarian_iou(masks, gt, thres = 0.5, emp=False):
    masks = (masks>thres)
    gt = (gt>thres)
    ious = iou(gt[:, None], masks[None], emp=False)
    g, p = np.shape(ious)

    orig_idx, hung_idx = linear_sum_assignment(-ious)
    out = ious[orig_idx, hung_idx]
    return out, masks[hung_idx] * 1.

def seq_hungarian_iou(masks, gts, thres = 0.5):
    g, h, w = np.shape(gts[0])
    p = np.shape(masks[0])[0]
    ious = np.zeros([g, 20])
    num_seq = len(gts)
    for i in range(num_seq):
        ious = ious + iou(gts[i][:, None], np.concatenate([masks[i], np.zeros([20-p,h,w])], 0)[None])
    orig_idx, hung_idx = linear_sum_assignment(-ious)
    out = ious[orig_idx, hung_idx] / num_seq
    return out, 0

def run_flowpsam(args, flowsam, info):
    with torch.no_grad():
        # Inputs
        original_size = (info["size"][0][0].item(), info["size"][1][0].item())
        input_size = (int(original_size[0] * 1024 / max(*original_size)), int(original_size[1] * 1024 / max(*original_size)))
        flow_image = info["flow_image"].cuda()   # 1 4 3 1024 1024
        rgb_image = info["rgb_image"].cuda()  # 1 3 1024 1024
        grid_coords_set = info["grid"].cuda().squeeze(0) # 100 1 2
        # Inference with iterative point prompt inputs
        masks_set = []
        scores_set = []
        flowsam.rgb_feature = None
        flowsam.flow_feature = None
        for coords_idx in range(grid_coords_set.shape[0] // 10):
            grid_coords = grid_coords_set[coords_idx * 10 : coords_idx * 10 + 10]
            point_labels = torch.ones(grid_coords.size()[:2], dtype=torch.int, device=grid_coords.device)
            point_prompts = (grid_coords, point_labels)
            masks_logit, fiou, mos = flowsam(rgb_image, flow_image, point_prompts, use_cache = True)  
            fiou = fiou[:, args.sam_channel]
            mos = mos[:, 0]
            score = fiou + mos
            masks_logit = masks_logit[..., : input_size[0], : input_size[1]]
            masks_logit = F.interpolate(masks_logit, original_size, mode="bilinear", align_corners=False)
            masks = (masks_logit > args.mod_thres).float()
            masks = masks[:, args.sam_channel]
            masks_set.append(masks)
            scores_set.append(score)
    masks_set = torch.cat(masks_set, 0)
    scores_set = torch.cat(scores_set, 0)
    boxes_set = batched_mask_to_box(masks_set.long()).float()
    return masks_set, scores_set, boxes_set


def run_flowisam(args, flowsam, info):
    with torch.no_grad():
        # Inputs
        original_size = (info["size"][0][0].item(), info["size"][1][0].item())
        input_size = (int(original_size[0] * 1024 / max(*original_size)), int(original_size[1] * 1024 / max(*original_size)))
        flow_image = info["flow_image"].cuda()   # 1 4 3 1024 1024
        grid_coords_set = info["grid"].cuda().squeeze(0) # 100 1 2
        # Inference with iterative point prompt inputs
        masks_set = []
        scores_set = []
        flowsam.flow_feature = None
        for coords_idx in range(grid_coords_set.shape[0] // 10):
            grid_coords = grid_coords_set[coords_idx * 10 : coords_idx * 10 + 10]
            point_labels = torch.ones(grid_coords.size()[:2], dtype=torch.int, device=grid_coords.device)
            point_prompts = (grid_coords, point_labels)
            masks_logit, fiou = flowsam(flow_image, point_prompts, use_cache = True)  
            fiou = fiou[:, args.sam_channel]
            score = fiou 
            masks_logit = masks_logit[..., : input_size[0], : input_size[1]]
            masks_logit = F.interpolate(masks_logit, original_size, mode="bilinear", align_corners=False)
            masks = (masks_logit > args.mod_thres).float()
            masks = masks[:, args.sam_channel]
            masks_set.append(masks)
            scores_set.append(score)
    masks_set = torch.cat(masks_set, 0)
    scores_set = torch.cat(scores_set, 0)
    boxes_set = batched_mask_to_box(masks_set.long()).float()
    return masks_set, scores_set, boxes_set

def move_eval(args, val_loader, flowsam, model_type, save_path):
   
    flowsam.eval()
    iou_summary = {}
    for idx, info in enumerate(val_loader):
        if idx % 100 == 0:
            print("---Inference step: {}".format(idx))

        # Set up performance logger
        if os.path.dirname(info["path"][0]) not in iou_summary.keys() and ("num_obj" in info.keys()):
            iou_summary[os.path.dirname(info["path"][0])] = {}
            for obj_idx in range(info["num_obj"].item()):
                iou_summary[os.path.dirname(info["path"][0])][obj_idx] = []
        
        # Running model
        if model_type == "flowi":
            masks_set, scores_set, boxes_set = run_flowisam(args, flowsam, info)
        elif model_type == "flowp":
            masks_set, scores_set, boxes_set = run_flowpsam(args, flowsam, info)

        """
        Post-processing
        """
        if "anno" in info.keys():  
            anno = info["anno"].cuda()  # 1 C H W
        else: # No GT
            anno = torch.zeros(1, 1) # empty array with anno.shape[1]=1
        
                
        # NMS
        keep_idx = batched_nms(boxes_set, scores_set, torch.zeros_like(boxes_set[:, 0]), iou_threshold=0.9)
        masks_fil, scores_fil, boxes_fil = filter_data([masks_set, scores_set, boxes_set], keep_idx, is_idx = True)
        
        # Removing bg masks
        keep_maskidx = ~is_bg_mask(masks_fil)
        masks_fil, scores_fil, boxes_fil = filter_data([masks_fil, scores_fil, boxes_fil], keep_maskidx)
 
        # Ordering masks according to the scores
        sel_idxs = torch.argsort(scores_fil, descending = True)
        scores = (scores_fil[sel_idxs])[0:max(args.max_obj, anno.shape[1])]
        masks_nonhung = (masks_fil[sel_idxs])[0:max(args.max_obj, anno.shape[1])]
        # Overlaying masks
        masks_nonhung, saved_mask_nonhung = hard_thres(masks_nonhung, scores, output_savemask = True)
        # Padding masks to match with num_obj
        if masks_nonhung.shape[0] < max(args.max_obj, anno.shape[1]):
            masks_nonhung_pad = torch.repeat_interleave(torch.zeros_like(masks_nonhung[0:1], device = masks_nonhung.device), max(args.max_obj, anno.shape[1]) - masks_nonhung.shape[0], 0)
            masks_nonhung = torch.cat([masks_nonhung, masks_nonhung_pad], 0)
            scores_pad = torch.zeros(max(args.max_obj, anno.shape[1]) - masks_nonhung.shape[0]).cuda()
            scores = torch.cat([scores, scores_pad], 0)
        
        if "anno" in info.keys():    
            # Hungarian matching and result summary
            result_iou = iou(anno[0, :, None], masks_nonhung[None])
            orig_idx, hung_idx = linear_sum_assignment(-result_iou.cpu().detach().numpy())
            masks_hung = masks_nonhung[hung_idx]  # Hungarian matched masks
            iou_summary = update_iousummary(masks_hung, masks_nonhung, anno, info["num_obj"].item(), info["path"], iou_summary, save_path = save_path)
        else:  # No GT
            if save_path:
                save_path_nonhung = os.path.join(save_path, "nonhung")
                os.makedirs(os.path.dirname(os.path.join(save_path_nonhung, info["path"][0])), exist_ok = True)
                save_indexed(os.path.join(save_path_nonhung, info["path"][0]), saved_mask_nonhung.astype(np.uint8))
        
    if len(iou_summary.keys()) != 0:  
        # IoU result output
        iou_list = []     
        for cat in iou_summary.keys():
            for obj in iou_summary[cat].keys():
                iou_list.append(np.mean(np.array(iou_summary[cat][obj])))
        print("---Mean IoU is: {} ".format(np.mean(np.array(iou_list))))
        print("")
        return np.mean(np.array(iou_list))