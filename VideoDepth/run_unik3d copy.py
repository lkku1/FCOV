# OUR
from unik_util import ImageDataset, generatemask, read_video_frames, write_exr_file, read_exr_file

import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import argparse
from E2P import equi2pers
from P2E import pers2equi
from torchvision.transforms import functional as TF
import kornia
from unik3d.models import UniK3D
from unik3d.utils.camera import (Pinhole, OPENCV, Fisheye624, MEI, Spherical)
import json
from numpy.polynomial import Polynomial
import open3d as o3d
from video_depth_anything.video_depth import VideoDepthAnything
import OpenEXR
import Imath
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.sparse import lil_matrix, csr_matrix

from mmseg.apis import init_model, inference_model, show_result_pyplot
from torch.utils.data import Dataset
from flow.core.raft import RAFT
from torch.utils.data import DataLoader
from flow.core.utils import flow_viz
from flow.core.utils.utils import InputPadder

Fov = (1.0, 0.5)
depth_size = (518, 518)
Nrows = 1
device = None

def get_patch_num():
    if Nrows == 3:
        return 10
    if Nrows == 2:
        return 8
    if Nrows == 1:
        return 6
    
def load_image(imfile, resolution=None):
    img = cv2.imread(imfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resolution:
        img = cv2.resize(img, dsize=resolution, interpolation=cv2.INTER_LINEAR)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.to(device)

class ImgPair(Dataset):
    def __init__(self, data_dir, gap, reverse):
        self.data_dir = data_dir
        self.images = None
        self.images_ = None
        self.gap = gap
        self.reverse = reverse
        self.images = sorted(os.listdir(data_dir))
        self.images_ = sorted(os.listdir(data_dir))[:-gap]

    def __len__(self):
        return len(self.images_)

    def __getitem__(self, index):
        images = self.images
        images_ = self.images_
        gap = self.gap
        if self.reverse:
            image1 = load_image(os.path.join(self.data_dir, images[index + gap]))
            image2 = load_image(os.path.join(self.data_dir, images_[index]))
            svfile = images[index + gap]
        else:
            image1 = load_image(os.path.join(self.data_dir, images_[index]))
            image2 = load_image(os.path.join(self.data_dir, images[index + gap]))
            svfile = images_[index]
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1[None], image2[None])
        return image1[0], image2[0], svfile
    
def run_flows(option):
    model = torch.nn.DataParallel(RAFT(option))
    model.load_state_dict(torch.load(option.model, weights_only=False))
    model = model.module

    model.to(device)
    model.eval()

    folder = os.path.basename(option.data_dir)
    floout = os.path.join(option.flow_dir, folder)

    for gap_idx in option.gap:
        for reverse_idx in option.reverse:
            imgpair_dataset = ImgPair(data_dir=option.data_dir, gap = gap_idx, reverse = reverse_idx)
            imgpair_loader = DataLoader(imgpair_dataset, batch_size=1, shuffle=False)
            with torch.no_grad():
            
                for _, data in enumerate(imgpair_loader):
                    image1, image2, svfiles = data
                    image1 = equi2pers(image1, Fov, Nrows, (480, 480))
                    image2 = equi2pers(image2, Fov, Nrows, (480, 480))
                    
                    _, flow_up = model(image1, image2, iters=20, test_mode=True)
                
                    for k, svfile in enumerate(svfiles):
                        flopath = os.path.join(floout, os.path.basename(svfile))
                        flo = torch.cat(torch.split(flow_up, 1, dim=0), dim=-1).permute(0, 2, 3, 1)[0].cpu().numpy()
                
                        flo = flow_viz.flow_to_image(flo)
                        cv2.imwrite(flopath, flo[:, :, [2, 1, 0]])


def run_background(dataset, option):

    mog_large = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=True)  # 大尺度
    mog_small = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=True)  # 小尺度
    bg_accumulators = np.zeros((option.height, option.width, 3))
    bg_mask = np.zeros((option.height, option.width)).astype(np.bool_)

    if (not os.path.exists(os.path.join(option.sem_dir, "background.png"))) or (not os.path.exists(os.path.join(option.sem_dir, dataset.files_name[-1]))):
        sem_model = init_model(option.config_file, option.checkpoint, device=device)

    if not os.path.exists(option.middle_rgb_dir):
        os.makedirs(option_.middle_rgb_dir)

    if not os.path.exists(option.sem_dir):
        os.makedirs(option.sem_dir)

    if not os.path.exists(os.path.join(option.data_dir, "background.png")):
        for image_ind, images in enumerate(dataset):

            if not os.path.exists(os.path.join(option.middle_rgb_dir, images.name + ".png")):
                rgb_patchs = equi2pers(torch.from_numpy(images.rgb_image.transpose(2,0,1))[None].float(), Fov, Nrows, depth_size).numpy()
                rgb_sets = []
                for index in range(0, get_patch_num(), 1):
                    rgb_patch = cv2.cvtColor(rgb_patchs[index].transpose(1,2,0).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    rgb_sets.append(rgb_patch)
                cat_rgb_patch = np.concatenate(rgb_sets, axis=1)
                cv2.imwrite(os.path.join(option.middle_rgb_dir, images.name + ".png" ), cat_rgb_patch)

            if not os.path.exists(os.path.join(option.sem_dir, images.name + ".png")):
                sem_set = []
                cat_rgb_patch = cv2.imread(os.path.join(option.middle_rgb_dir, images.name + ".png" ))
                rgb_patchs = np.split(cat_rgb_patch, get_patch_num(), axis=1)
                for index in range(0, get_patch_num(), 1):
                    result = inference_model(sem_model, rgb_patchs[index])
                    sem_set.append(result.pred_sem_seg.data[0].cpu().numpy())
                cat_sem_patch = np.concatenate(sem_set, axis=1)
                cv2.imwrite(os.path.join(option.sem_dir, images.name + ".png"),  cat_sem_patch)

            if image_ind % 5 == 0:
                if image_ind == 0:
                    extract_image = cv2.imread(os.path.join(option.data_dir, dataset.files_name[-3]))
                    extract_image = cv2.cvtColor(extract_image, cv2.COLOR_BGR2RGB)
                    fg_large = mog_large.apply(extract_image)
                    fg_small = mog_small.apply(cv2.GaussianBlur(extract_image, (5,5), 0))

                fg_large = mog_large.apply(images.rgb_image)
                fg_small = mog_small.apply(cv2.GaussianBlur(images.rgb_image, (5,5), 0))

                fg_mask = cv2.bitwise_or(fg_large, fg_small)

                cat_sem_patch = cv2.imread(os.path.join(option.sem_dir, images.name + ".png" ), -1)
                cat_sem_mask = np.where(cat_sem_patch == 12, 255, 0)
                sem_patch_masks = np.split(cat_sem_mask, get_patch_num(), axis=1)
        
                sem_patch_set = torch.tensor(np.stack(sem_patch_masks, axis=-1)[None, None]).float()
                alpha = torch.tensor(generatemask(depth_size).reshape(1, 1, depth_size[0],depth_size[1], 1))
                alpha = torch.repeat_interleave(alpha, get_patch_num(), dim=-1)
                sem_patch_set = torch.cat([sem_patch_set, alpha], dim=1).to(device)

                sem_image = pers2equi(sem_patch_set, Fov, Nrows, depth_size, (option.height, option.width), "Patch_P2E2048_1")[0, 0].cpu().numpy()
                
                fg_mask = np.where(sem_image > 0, 255, fg_mask)
                
                fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
                fg_mask = ndimage.binary_fill_holes(fg_mask).astype(np.bool_)
                
                increase_mask = ~bg_mask &  ~fg_mask
                intersection_mask = bg_mask & ~fg_mask
                
                bg_accumulators[intersection_mask] = 0.99 * bg_accumulators[intersection_mask] + 0.01 * images.rgb_image[intersection_mask]
                bg_accumulators[increase_mask] = images.rgb_image[increase_mask]
                bg_mask[increase_mask] = True
    
        bg_accumulators = cv2.convertScaleAbs(bg_accumulators)
        bg_accumulators = cv2.cvtColor(bg_accumulators.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(option.data_dir, "background.png"), bg_accumulators)
        cv2.imwrite(os.path.join(option.sem_dir, "background_hole.png"), (~bg_mask) * 255)

    if not os.path.exists(os.path.join(option.sem_dir, "background.png")):
        background = cv2.imread(os.path.join(option.data_dir, "background.png"))
        background_patchs = equi2pers(torch.from_numpy(background.transpose(2,0,1))[None].float(), Fov, Nrows, depth_size).numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        backsem_sets = []
        for index in range(0, get_patch_num(), 1):
            
            result = inference_model(sem_model, background_patchs[index])
            backsem_sets.append(result.pred_sem_seg.data[0].cpu().numpy())
         
        cat_sem_patch = np.concatenate(backsem_sets, axis=1)
        cv2.imwrite(os.path.join(option.sem_dir, "background.png"),  cat_sem_patch)

   
def run_move_mask(dataset, option, background_depth, depths):
    
    if not os.path.exists(option.move_dir):
        os.makedirs(option.move_dir)

    if not os.path.exists(os.path.join(option.move_dir, dataset.files_name[-1])):
        background_substractor = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)
        background = cv2.imread(os.path.join(option.data_dir, 'background.png'))
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB) 
        background_substractor.apply(background)

        depths = np.stack(depths, axis=0)
        disparity = (1/depths - 1/depths.max()) / (1/depths.min() - 1/depths.max()) * 255

        depth_substractors = []
        for index in range(get_patch_num()):
        
            depth_substractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=8,  detectShadows=False)
            depth_substractor.apply(disparity[index].min(0).astype(np.uint8))
    
            depth_substractors.append(depth_substractor)

        alpha = torch.tensor(generatemask(depth_size).reshape(1, 1, depth_size[0],depth_size[1], 1))
        alpha = torch.repeat_interleave(alpha, get_patch_num(), dim=-1)

        background_hole = cv2.imread(os.path.join(option.sem_dir, "background_hole.png"), -1)[None]
        for image_ind, images in enumerate(dataset):
            image_ind = 28
            if image_ind == 100:
                break
            # if not os.path.exists(os.path.join(move_dir, str(seq_name) + ".png")):

            rgb_mask = background_substractor.apply(images.rgb_image)
            _, rgb_mask = cv2.threshold(rgb_mask, 250, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5,5), np.uint8)
            rgb_mask = cv2.morphologyEx(rgb_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            rgb_mask = cv2.morphologyEx(rgb_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
           
            rgb_mask = ndimage.binary_fill_holes(rgb_mask)[None]

            cat_rgbandhole = np.concatenate((rgb_mask, background_hole), axis=0)

            rgbandhole_mask_patchs = equi2pers(torch.from_numpy(cat_rgbandhole)[None].float(), Fov, Nrows, depth_size).numpy().astype(np.uint8)
           
            # rgb_mask_patchs = rgbandhole_mask_patchs[:, :1]
            # hole_mask_patch = rgbandhole_mask_patchs[:, 1:]

            # move_mask_patchs = []
            # for index in range(get_patch_num()):
            #     depth_mask_patch = depth_substractors[index].apply(depths[index, image_ind].astype(np.uint8), learningRate=0)
            #     sem_patch = sem_patchs[index]
            #     if index == 1:
            #         cv2.imwrite("2.png", depths[index, image_ind].astype(np.uint8))
            #         a = 0
                
                # if depth_mask_patch.sum() == 0:
                #     labels, N = ndimage.label(rgb_mask_patchs[index, 0])

                #     remain_label = np.zeros_like(labels)
                #     for l_index in range(1, N + 1, 1):
                #         label_index = np.where(labels == l_index, 1, 0)
                #         if label_index.sum() > 60:
                #             remain_label = remain_label + label_index
                #     move_mask = (remain_label.astype(np.bool_) | hole_mask_patch[index, 0].astype(np.bool_)) * 255
                # else:
                #     labels, N = ndimage.label(rgb_mask_patchs[index, 0])

                #     remain_label = np.zeros_like(labels)
                #     for l_index in range(1, N + 1, 1):
                #         label_index = np.where(labels == l_index, 1, 0)
                #         if label_index.sum() > 60:
                #             remain_label = remain_label + label_index
        
                #     sems_index = np.unique(sem_patch[depth_mask_patch.astype(np.bool_)])
                #     sems_index = sems_index[sems_index>0]
                #     remain_sem = np.zeros_like(depth_mask_patch)
                #     for sem_index in sems_index:
                #         sem_map = np.where(sem_patch == sem_index, 255, 0)
                       
                #         labels, N = ndimage.label(sem_map)
                       
                #         for l_index in range(1, N + 1, 1):
                #             sem_map = labels == l_index
                #             ratio = (sem_map * depth_mask_patch.astype(np.bool_)).sum() / sem_map.sum()
                #             if ratio > 0.5:
                #                 remain_sem = remain_sem + sem_map

                #         remain_sem = remain_sem + depth_mask_patch.astype(np.bool_)

                #     move_mask = (remain_label.astype(np.bool_) | hole_mask_patch[index, 0].astype(np.bool_)) * 255
                #     move_mask[remain_sem.astype(np.bool_)] = 127

                # move_mask_patchs.append(move_mask)
            # cat_move_patch = np.concatenate(move_mask_patchs, axis=1)
            # cv2.imwrite(os.path.join(option.move_dir, images.name + ".png"), cat_move_patch)

def run_depth_fusion(depths, background_depth, dataset, option):

    backper_depths = equi2pers(torch.from_numpy(background_depth.copy())[None, None].float(), Fov, Nrows, depth_size).numpy()

    alpha = generatemask(depth_size).reshape(1, 1, depth_size[0],depth_size[1], 1)
    alpha = np.repeat(alpha, get_patch_num(), axis=-1)

    # fit_matrix = np.zeros((get_patch_num(), depths[0].shape[0], 2))
    # for image_ind, images in enumerate(dataset):
    #     if image_ind == 100:
    #         break
       
    #     cat_move_patch = cv2.imread(os.path.join(option.move_dir, images.name + ".png" ), -1)
    #     move_patchs = np.split(cat_move_patch, get_patch_num(), axis=1)

    #     cat_sem_patch = cv2.imread(os.path.join(option.sem_dir, images.name + ".png" ), -1)
    #     sem_patchs = np.split(cat_sem_patch, get_patch_num(), axis=1)

    #     for patch_idx in range(0, get_patch_num(), 1):
    #         sem_patch = sem_patchs[patch_idx]
    #         move_patch = move_patchs[patch_idx]
    #         backper_depth = backper_depths[patch_idx, 0]
    #         depth = depths[patch_idx][image_ind]

    #         normal_clone = cv2.seamlessClone(depth, backper_depth, np.where(move_patch>0, 255, 0), (int(depth_size[0]/2), int(depth_size[1]/2)), cv2.NORMAL_CLONE)

    #         mov_dilate_mask = cv2.dilate(move_patch.astype(np.uint8), np.ones((21, 21), dtype=np.uint8), iterations=1)
    #         mov_around_mask = (mov_dilate_mask - move_patch).astype(np.bool_)

    #         not_skytree_mask = 1.0 - (np.where(sem_patch == 2.0, 1.0, 0.0) + np.where(sem_patch == 4.0, 1.0, 0.0))
    #         not_skytree_mask = cv2.erode(not_skytree_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(np.bool_)
            
    #         inter_mask = mov_around_mask & not_skytree_mask & (backper_depth < backper_depths.max() * 0.5)

    #         if inter_mask.sum() > 100:
    #             # A = np.vstack([depth[inter_mask], np.ones(len(depth[inter_mask]))]).T
    #             # coefficients, _ = nnls(A, backper_depth[inter_mask])
    #             # intercept, slope = Polynomial.fit(depth[inter_mask], backper_depth[inter_mask], deg=1).convert().coef
    #             backper_mean, backper_std = backper_depth[inter_mask].mean(), backper_depth[inter_mask].std()
    #             depth_mean, depth_std = depth[inter_mask].mean(), depth[inter_mask].std()

    #             slope = backper_std / (depth_std + 1e-6)
    #             intercept = backper_mean - slope * depth_mean

    #             fit_matrix[patch_idx, image_ind, 0] = slope
    #             fit_matrix[patch_idx, image_ind, 1] = intercept
    #         else:
    #             fit_matrix[patch_idx, image_ind] = fit_matrix[patch_idx, :image_ind].sum(axis=0) / (image_ind + 1e-6)

    #         # if patch_idx ==2:
    #         #     # depth = slope * depth + intercept
    #         #     a  = (1/depth - 1/depth.max())/(1/depth.min() - 1/depth.max()) * 255
    #         #     cv2.imwrite("2.png", a)
    #         #     b = (1/backper_depth - 1/backper_depth.max())/(1/backper_depth.min() - 1/backper_depth.max()) * 255
    #         #     cv2.imwrite("3.png", b)
    #         #     a = 1
    
    # fit_coefs = []
    # for idx in range(0, get_patch_num(), 1):
    #     fit_coefs.append(np.mean(fit_matrix[idx], axis=0))

    # k_size = int(2 * np.ceil(2 * int(depth_size[0]/32)) + 1)
    # sigma = int(depth_size[0]/32)
    for image_ind, images in enumerate(dataset):
 
        cat_move_patch = cv2.imread(os.path.join(option.move_dir, images.name + ".png" ), -1)
        move_patchs = np.split(cat_move_patch, get_patch_num(), axis=1)
        
        refine_depths = []
        for patch_idx in range(0, get_patch_num(), 1):
     
            move_patch = move_patchs[patch_idx].astype(np.uint8)

            backper_depth = backper_depths[patch_idx, 0].copy()
          
            depth = depths[patch_idx][image_ind]
            # depth = fit_coefs[patch_idx][0] * depth + fit_coefs[patch_idx][1]



            cv2.imwrite("1.png", blended)

            refine_depths.append(blended)

            # if patch_idx == 3:
            #     cv2.imwrite("1.png", (1/depth - 1/depth.max())/(1/depth.min() - 1/depth.max()) * 255)
            #     cv2.imwrite("2.png", (1/refine_depth - 1/refine_depth.max())/(1/refine_depth.min() - 1/refine_depth.max()) * 255)
            #     cv2.imwrite("4.png", (1/backper_depth - 1/backper_depth.max())/(1/backper_depth.min() - 1/backper_depth.max()) * 255)
            #     cv2.imwrite("3.png", images.rgb_image)
            #     a = 1
        refine = np.stack(refine_depths, axis=-1)[None, None]
        refine = np.concatenate((refine, alpha), axis=1)
        omni_depth = pers2equi(torch.tensor(refine), Fov, Nrows, depth_size, (background_depth.shape[0], option), "Patch_P2E2048_1").numpy()[0, 0]

        omni_depth = np.where(sem_image==2, background_depth.max(), omni_depth)
        omni_depth = np.clip(omni_depth, 0.0, background_depth.max())
        a = (1/omni_depth - 1/omni_depth.max()) / (1/omni_depth.min() - 1/omni_depth.max()) * 255
        cv2.imwrite("1.png",a)
        b = 1


def run_perspective_depth(dataset, option):

    name = os.path.basename(option.data_dir)

    if not os.path.exists(option_.middle_dep_dir):
        os.makedirs(option_.middle_dep_dir)

    depth_set = []
    if not os.path.exists(os.path.join(option.middle_dep_dir, name + str(get_patch_num()-1) + ".npz")):

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        video_depth_anything = VideoDepthAnything(**model_configs["vitl"])
        video_depth_anything.load_state_dict(torch.load(os.path.join(option.load_vda, f'metric_video_depth_anything_{"vitl"}.pth'), map_location='cpu'), strict=True)
        video_depth_anything = video_depth_anything.to(device).eval()

        frame_patch = []
        for index in range(0, get_patch_num(), 1):
            frame_patch.append([])

        # for frame_index in range(len(dataset.files_name)):
        for frame_index in range(100):
            cat_frame_patch = cv2.imread(os.path.join(option.middle_rgb_dir, dataset.files_name[frame_index]))
            cat_frame_patch = cv2.cvtColor(cat_frame_patch, cv2.COLOR_BGR2RGB)
            rgb_patchs = np.split(cat_frame_patch, get_patch_num(), axis=1)
            for index in range(0, get_patch_num(), 1):
                frame_patch[index].append(rgb_patchs[index])

        for index in range(0, get_patch_num(), 1):
            frames, target_fps = np.stack(frame_patch[index], axis=0), 30

            depth, _ = video_depth_anything.infer_video_depth(frames, target_fps, input_size=depth_size[0])
            np.savez_compressed(os.path.join(option.middle_dep_dir, name + str(index) + ".npz"), depth)
            depth_set.append(depth)

    else:
        for idx in range(0, get_patch_num(), 1):
            data = np.load(os.path.join(option.middle_dep_dir, name + str(idx) + ".npz"))
            depth_set.append(data["arr_0"])
            data.close()

    return depth_set
    

def depth_estimation(unik, omni_rgb_ori, sky_sem, omni_camera, per_camera, alpha):
    
    omni_rgb = torch.from_numpy(omni_rgb_ori).permute(2, 0, 1).float()
    omni_sem = torch.from_numpy(sky_sem)[None, None].float().to(device)

    add_length = int((omni_rgb.shape[-1] / 140)) * 14
    omni_rgb_add = torch.cat([omni_rgb, omni_rgb[..., :add_length]], dim=-1)

    omni_predictions = unik.infer(omni_rgb_add, omni_camera.clone())
    omni_depth = omni_predictions["distance"]
    
    _, weight_width = torch.meshgrid(torch.linspace(0, 1, omni_rgb_add.shape[1]), torch.linspace(0, 1, add_length), indexing="ij")
    weight_width = weight_width[None,None].to(device)
    omni_depth[..., :add_length] = omni_depth[..., -add_length:] * (1 - weight_width) + omni_depth[..., :add_length] * weight_width
    omni_depth = omni_depth[..., :-add_length]
    # a = ((1/omni_depth - 1/omni_depth.max())/(1/omni_depth.min() - 1/omni_depth.max()))[0,0].cpu().numpy()*255
    # cv2.imwrite("1.png", a)
    
    per_rgb = equi2pers(torch.from_numpy(omni_rgb_ori)[None].permute(0, 3, 1, 2).float(), Fov, Nrows, depth_size)
    omni_info = torch.cat([omni_depth, omni_sem],dim=1)
    omni_per_info = equi2pers(omni_info, Fov, Nrows, depth_size)
    omni_per_depth = omni_per_info[:, :1]
    omni_per_sem= omni_per_info[:, 1:]

    omni_per_sem = torch.where(omni_per_sem > 0.0, True, False).to(torch.bool)
    kernel = torch.ones(11, 11, device=device)
    
    per_depths = []
    for index in range(per_rgb.shape[0]):

        source_depth = omni_per_depth[index:index+1]
        source_sem = omni_per_sem[index:index+1]
        if index < per_rgb.shape[0] - 1:

            per_predictions = unik.infer(per_rgb[index:index+1], per_camera.clone())
            target_depth = per_predictions["distance"]

            per_sem_mask = kornia.morphology.dilation(source_sem, kernel)
            confidence_mask = (1.0 - per_sem_mask).to(torch.bool)
        
            valid_d1 = source_depth[confidence_mask].reshape(-1).cpu().numpy()
            valid_d2 = target_depth[confidence_mask].reshape(-1).cpu().numpy()

            if confidence_mask.sum() <= 100:
                target_depth = source_depth.clone()
            else:
                coef = Polynomial.fit(valid_d2, valid_d1, deg=1).convert().coef
                slope = torch.tensor(coef[1]).to(device) 
                intercept = torch.tensor(coef[0]).to(device)
                target_depth = slope * target_depth + intercept
        else:
            target_depth = source_depth.clone()
        
        target_depth = torch.where(torch.logical_and(target_depth < source_depth, target_depth > 1.0), target_depth, source_depth)
        target_depth[source_sem] = omni_depth.max()

        per_depths.append(target_depth)

    per_depth = torch.cat(per_depths)[None].permute(0, 2, 3, 4, 1)
    per_dep_refine = torch.cat([per_depth, alpha], dim=1)
    per_depth = pers2equi(per_dep_refine, Fov, Nrows, depth_size, (omni_rgb.shape[1], omni_rgb.shape[2]), "Patch_P2E2048_1")

    torch.cuda.empty_cache()
    return per_depth[0,0].cpu().numpy()

def equirectangular_to_pointcloud(rgb, depth):
    """将等距柱状投影的RGB-D转换为3D点云"""
    h, w = depth.shape
    theta = np.linspace(-np.pi, np.pi, w)  # 经度 [-π, π]
    phi = np.linspace(-np.pi/2, np.pi/2, h)  # 纬度 [-π/2, π/2]
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # 球面坐标→3D笛卡尔坐标
    x = depth * np.cos(phi_grid) * np.sin(theta_grid)
    y = depth * np.sin(phi_grid)
    z = depth * np.cos(phi_grid) * np.cos(theta_grid)
    
    # 展平并过滤无效点
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    valid = depth.flatten() > 0
    return points[valid], colors[valid]

def visualize_with_open3d(rgb, depth):
    points, colors = equirectangular_to_pointcloud(rgb, depth)
    
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Mesh Viewer")
    vis.add_geometry(pcd)

    # # 设置初始相机参数
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])  # 前方向（Z轴负方向）
    view_control.set_lookat([0, 0, 0])  # 注视点
    view_control.set_up([0, -1, 0])      # 上方向
    view_control.set_zoom(0.03)
    # 启动交互窗口（WASD控制）

    vis.run()
    vis.destroy_window()

def vis():
    image = o3d.io.read_image("D:/linux/github2/3dpb/00000.jpg")
    depth = o3d.io.read_image("D:/linux/github2/3dpb/2.png")
    visualize_with_open3d(image, depth)

def run_background_depth(option):
    
    # if os.path.exists(os.path.join(option.out_dir, "background.exr")):
    #     background_depth = read_exr_file(os.path.join(option.out_dir, "background.exr"))

    #     return background_depth
    # else:
    if True:

        if not os.path.exists(option_.out_dir):
            os.makedirs(option_.out_dir)

        unik = UniK3D.from_pretrained(option.load_from).to(device)
        unik.resolution_level = 9
        unik.interpolation_mode = "bilinear"
        unik.to(device).eval()
        
        omni_camera_path = os.path.join(option.load_config, "equirectangular.json")
        with open(omni_camera_path, "r") as f:
            camera_dict = json.load(f)
        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        omni_camera = eval(name)(params=params)

        per_camera_path = os.path.join(option.load_config, "pinhole.json")
        with open(per_camera_path, "r") as f:
            camera_dict = json.load(f)
        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        per_camera = eval(name)(params=params)

        alpha = torch.tensor(generatemask(depth_size).reshape(1, 1, depth_size[0],depth_size[1], 1))
        alpha = torch.repeat_interleave(alpha, get_patch_num(), dim=-1)

        background = cv2.imread(os.path.join(option.data_dir, "background.png"))
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        cat_background_sem = cv2.imread(os.path.join(option.sem_dir, "background.png"), -1)
        cat_sem_mask = np.where(cat_background_sem == 2, 255, 0)
        sem_patch_masks = np.split(cat_sem_mask, get_patch_num(), axis=1)

        sem_patch_set = torch.tensor(np.stack(sem_patch_masks, axis=-1)[None, None]).float()
        sem_patch_set = torch.cat([sem_patch_set, alpha], dim=1).to(device)
        background_sky = pers2equi(sem_patch_set, Fov, Nrows, depth_size, (option.height, option.width), "Patch_P2E2048_1")[0, 0].cpu().numpy()
        background_sky = np.where(background_sky > 0 , 1, 0)

        background_depth = depth_estimation(unik, background, background_sky, omni_camera, per_camera, alpha.to(device))
 
        write_exr_file(os.path.join(option.out_dir, "background.exr"), background_depth)
        
        return background_depth

if __name__ == "__main__":
    
    # Adding necessary input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="D:/linux/github2/test_image/image/cafeteria1", type=str)
    parser.add_argument('--move_dir', default="D:/linux/github2/test_image/preprocess/cafeteria1_move_mask", type=str)
    parser.add_argument('--sem_dir', default="D:/linux/github2/test_image/preprocess/cafeteria1_sem_mask", type=str)
    parser.add_argument('--flow_dir', default="D:/linux/github2/test_image/preprocess/cafeteria1_flow", type=str)

    parser.add_argument('--middle_rgb_dir', default="D:/linux/github2/test_image/preprocess/cafeteria1_middle_rgb", type=str)
    parser.add_argument('--middle_dep_dir', default="D:/linux/github2/test_image/preprocess/cafeteria1_middle_dep", type=str)
    parser.add_argument('--out_dir', default="D:/linux/github2/test_image/preprocess/cafeteria1_depth", type=str)
    parser.add_argument('--width', default=2048, type=int)
    parser.add_argument('--height', default=1024, type=int)

    parser.add_argument('--load_from', type=str, default='D:/linux/github2/3dpb/VideoDepth/checkpoints/unik3d')
    parser.add_argument('--load_vda', type=str, default='D:/linux/github2/3dpb/VideoDepth/checkpoints')
    parser.add_argument('--encoder', default="vits")
    parser.add_argument('--load_config', type=str, default='D:/linux/github2/3dpb/VideoDepth/assets/')
    parser.add_argument('--gpu', default=2, type=int)
    parser.add_argument('--gap', default=[-2, -1, 1, 2])
    parser.add_argument('--reverse', default=[0, 1])
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', default=False)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    
    parser.add_argument('--model', default="D:/linux/github2/3dpb/VideoDepth/checkpoints/raft-things.pth")
    parser.add_argument('--config_file', default="D:/linux/github2/3dpb/VideoSem/mmsegmentation/configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py")
    parser.add_argument('--checkpoint', default="D:/linux/github2/3dpb/VideoSem/chickpoint/mask2former_swin-s_8xb2-160k_ade20k-512x512_20221204_143905-e715144e.pth")
    # Check for required input
    option_, _ = parser.parse_known_args()
    print(option_)

    # select device
    device = torch.device("cuda")

    # Create dataset from input images
    dataset_ = ImageDataset(option_.data_dir, option_.out_dir, option_.width, option_.height)
    run_flows(option_)
    # run_background(dataset_, option_)
    # background_depth = run_background_depth(option_)

    # depth = run_perspective_depth(dataset_, option_)

    # run_move_mask(dataset_, option_, background_depth, depth)
    # run_depth_fusion(depth, background_depth, dataset_, option_)


