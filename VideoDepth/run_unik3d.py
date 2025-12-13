from utils.util import generatemask, write_exr_file, read_exr_file

import os
import torch
import cv2
import numpy as np
from E2P import equi2pers
from P2E import pers2equi
import argparse
from unik3d.models import UniK3D
from unik3d.utils.camera import (Pinhole, OPENCV, Fisheye624, MEI, Spherical)

import open3d as o3d
from video_depth_anything.video_depth import VideoDepthAnything
# from scipy.ndimage import distance_transform_edt
# from pietorch import blend, blend_numpy
import time
from scipy import ndimage
# import math

depth_size = None
device = None
patch_num = 4
alpha = None

def run_background(option, step=5):

    if not os.path.exists(os.path.join(option.data_dir, option.name, "background.png")):
        per_depths = []
        for index in range(0, patch_num, 1):
            data = np.load(os.path.join(option.save_dir, option.name + "_per_depth", str(index)+".npz"))
            per_depths.append(data["arr_0"])
            data.close()

        per_depths = np.stack(per_depths, axis=-1)[::step]
        depth_means = np.mean(per_depths, axis=0)

        file_names = sorted(os.listdir(os.path.join(option.data_dir, option.name)))
        if "background.png" in file_names:
            file_names.remove("background.png")
    
        rgbs = []
        for file_index in range(0, len(file_names), step):
            image = cv2.imread(os.path.join(option.data_dir, option.name, file_names[file_index]))
            rgbs.append(image)
        rgb = np.stack(rgbs, axis=0)
        rgb_mean = np.mean(rgb, axis=0)

        mask = np.where(per_depths > depth_means[None], True, False).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        for idx in range(0, mask.shape[0], 1):
            mask[idx] = cv2.erode(mask[idx], kernel)
        mask = torch.cat([torch.tensor(mask[:, None]), torch.repeat_interleave(alpha, mask.shape[0], dim=0)], dim=1).float()
        mask = pers2equi(mask, 1, depth_size, (option.height, option.width), "111").numpy()[:, 0, :, :, None].astype(np.bool_)
        
        rgb = (rgb * mask).sum(axis=0) / (mask.sum(axis=0) + 1e-6)
        rgb = np.where(mask.sum(axis=0) == 0, rgb_mean, rgb)

        cv2.imwrite(os.path.join(option.data_dir, option.name,  "background.png"), rgb.astype(np.uint8))
       
def run_move_mask(option, file_num = 10):
    
    move_path = os.path.join(option.save_dir, option.name + "_move")
    if not os.path.exists(move_path):
        os.makedirs(move_path)

    if not os.path.exists(os.path.join(move_path, "move.npz")):
        per_depths = []
        intrinsic_matrix = np.array([
                    [depth_size * 0.8, 0, depth_size/2],
                    [0, depth_size * 0.8, depth_size/2],
                    [0, 0, 1]
            ])
            
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        
        u, v = np.meshgrid(np.arange(0, depth_size, 1), np.arange(0, depth_size, 1))
        for index in range(0, patch_num, 1):
            data = np.load(os.path.join(option.save_dir, option.name + "_per_depth",  "per_depth_" + str(index) + ".npz"))
            z = data["arr_0"]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            ds = np.sqrt(x**2 + y**2 + z**2)
            per_depths.append(ds)
            data.close()
        depth = np.stack(per_depths, axis=1)
        
        depth_max = depth.max(0)
        depth_back = ((depth_max - depth.min()) / (depth.max() - depth.min()) * 255)
        disparity_back = ((1/depth_max - 1 / depth.max()) / (1/depth.min() - 1/depth.max()) * 255)
        depth_infor = torch.cat([torch.tensor(np.concatenate((depth_back[None, None], disparity_back[None, None]), axis=1)).permute(0, 1, 3, 4, 2), alpha], dim=1)
        depth_infor = pers2equi(depth_infor, 1, depth_size, (option.height, option.width), "222").numpy()
        depth_back = depth_infor[0, 0]
        disparity_back = depth_infor[0, 1]

        iteration = depth.shape[0] // file_num + 1
        # write_exr_file(os.path.join(option.save_dir, option.name + "_depth", "1231.exr"), depth[0])
        file_names = sorted(os.listdir(os.path.join(option.data_dir, option.name)))
        file_names = []
        for index in range(len(os.listdir(os.path.join(option.data_dir, option.name))) - 1):
            file_names.append(str(index) + ".jpg")
        if "background.png" in file_names:
            file_names.remove("background.png")

        depth_substractor = cv2.createBackgroundSubtractorMOG2()
        depth_substractor.apply(depth_back.astype(np.uint8))
        disparity_substractor = cv2.createBackgroundSubtractorMOG2()
        disparity_substractor.apply(disparity_back.astype(np.uint8))
        start_time = time.time()
        move_masks = []
        for interation_index in range(0, iteration, 1):
            start_index = interation_index*file_num
            end_index = min((interation_index+1) * file_num, len(file_names))

            depth_norm = (depth[start_index:end_index] - depth.min()) / (depth.max() - depth.min()) * 255
            disparity = ((1/depth[start_index:end_index] - 1 / depth.max()) / (1/depth.min() - 1/depth.max()) * 255)
            depth_infor = torch.cat([torch.tensor(np.concatenate((depth_norm[:, None], disparity[:, None]), axis=1)).permute(0, 1, 3, 4, 2), torch.repeat_interleave(alpha, depth_norm.shape[0], dim=0)], dim=1)
            depth_infor = pers2equi(depth_infor, 1, depth_size, (option.height, option.width), "222").numpy()
            depth_file = depth_infor[:, 0]
            disparity_file = depth_infor[:, 1]

            for file_index in range(0, end_index - start_index , 1):
                
                depth_mask = depth_substractor.apply(depth_file[file_index].astype(np.uint8), learningRate=0)
                disparity_mask = disparity_substractor.apply(disparity_file[file_index].astype(np.uint8), learningRate=0)

                sem_index = interation_index * file_num + file_index
                sem = np.load(os.path.join(option.save_dir, option.name+ "_sem", os.path.splitext(file_names[sem_index])[0] + ".npy"))
                sky_mask = np.where(sem[..., 1] == 179, False, True)

                if disparity_mask.sum() > 0:
                    move_mask = (disparity_mask * sky_mask).astype(np.bool_)
                else:
                    move_mask = (depth_mask * sky_mask).astype(np.bool_)

                if move_mask.sum() > 0 :
                    instance_mask = sem[..., 0][move_mask]
                    unique_instance = sorted(np.unique(instance_mask))
                    remain_instance = np.zeros_like(move_mask)
                    for i_index in unique_instance:
                        instance_index = np.where(sem[..., 0] == i_index, True, False)
                        insert_index = move_mask * instance_index
                        if sem[..., 1][instance_index].any() == 1:
                            th = 0.1
                        else:
                            th = 0.5
                        if insert_index.sum() / instance_index.sum() > th:
                            remain_instance = remain_instance + instance_index * 255
                else:
                    remain_instance = np.where(sem[..., 1]==1, 255, 0)
                move_masks.append(remain_instance)
        move_masks = np.stack(move_masks, axis=0).astype(np.bool_)
        np.savez_compressed(os.path.join(move_path, "move.npz"), move_masks)
        end_time = time.time()
        dura_time = end_time - start_time
        a= 0

def run_depth_fusion(option, file_num= 10):

    depth_path = os.path.join(option.save_dir, option.name + "_depth")
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    file_names = sorted(os.listdir(os.path.join(option.data_dir, option.name)))
    file_names = []
    for index in range(len(os.listdir(os.path.join(option.data_dir, option.name))) - 1):
        file_names.append(str(index) + ".jpg")

    if "background.png" in file_names:
        file_names.remove("background.png")

    intrinsic_matrix = np.array([
            [depth_size * 0.8, 0, depth_size/2],
            [0, depth_size * 0.8, depth_size/2],
            [0, 0, 1]
    ])
    
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    u, v = np.meshgrid(np.arange(0, depth_size, 1), np.arange(0, depth_size, 1))

    per_depths = []
    for idx in range(0, patch_num, 1):
        data = np.load(os.path.join(option.save_dir, option.name + "_per_depth", "per_depth_" + str(idx) + ".npz"))
        z = data["arr_0"]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        ds = np.sqrt(x**2 + y**2 + z**2)
        per_depths.append(ds)
        data.close()
    per_depths = np.stack(per_depths, axis=0)

    move_data = np.load(os.path.join(option.save_dir, option.name + "_move", "move.npz"))
    move_masks = move_data["arr_0"]
    move_data.close()

    iteration = len(file_names) // file_num + 1

    background_depth = read_exr_file(os.path.join(option.save_dir, option.name + "_init_depth",  "background.exr"))
    u2, v2 = np.meshgrid(np.arange(option.width), np.arange(option.height))
    lat2 = (v2 / (option.height)) * np.pi - np.pi / 2
    background_depth_re = background_depth.copy()
    start_time = time.time()
    for interation_index in range(0, iteration, 1):
        start_index = max(interation_index*file_num - 1, 0)
        end_index = min((interation_index+1) * file_num + 1, len(file_names))
        # if not os.path.exists(os.path.join(option.save_dir, option.name + "_depth", os.path.splitext(file_names[end_index-1])[0] + ".exr")):
        if True:
            depth_information = torch.cat([torch.tensor(per_depths)[None, :, start_index:end_index].permute(2, 0, 3, 4, 1), torch.repeat_interleave(alpha, end_index-start_index, dim=0)], dim=1)
            per_depth = pers2equi(depth_information, 1, (depth_size, depth_size), (option.height, option.width), layer_name="222").numpy()[:, 0]
            per_mask = np.where(per_depth==0, False, True)

            move_masks_iteration = move_masks[start_index:end_index]

            omni_depths = []
            dynamic_masks = []
            omni_sems = []
            for file_idx in range(0, end_index - start_index, 1):
                sem_index = start_index + file_idx
                omni_sem = np.load(os.path.join(option.save_dir, option.name + "_sem", os.path.splitext(file_names[sem_index])[0] + ".npy"))
                sky_per = np.where(np.logical_or(omni_sem[..., 1] == 179, omni_sem[..., 1]==1), True, False)
                movesky_mask = ~(move_masks_iteration[file_idx] | sky_per)
                movesky_mask_erode = cv2.erode(movesky_mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(np.bool_)
                
                omni_depth = read_exr_file(os.path.join(option.save_dir, option.name + "_init_depth", os.path.splitext(file_names[sem_index])[0] + ".exr"))
            
                std_omni = (omni_depth)[movesky_mask_erode].std()
                min_back = background_depth[movesky_mask].min()
                std_back = (background_depth)[movesky_mask_erode].std()
                slope = std_back / (std_omni + 1e-6)
                omni_depth = (omni_depth * slope + (min_back - (omni_depth * slope).min())).astype(np.float32)

                omni_sems.append(omni_sem)
                omni_depths.append(omni_depth)
                dynamic_masks.append(movesky_mask_erode)
                
            dynamic_masks = np.stack(dynamic_masks, axis=0)
            omni_depth = np.stack(omni_depths, axis=0)

            dynamic_masks = np.where(per_mask==True, dynamic_masks, False)
            per_depth = per_depth * omni_depth[dynamic_masks].std() / per_depth[dynamic_masks].std() 
            per_depth = per_depth - per_depth[dynamic_masks].min() + omni_depth[dynamic_masks].min()
            per_depth = np.where(per_mask == False, omni_depth, per_depth)
            per_depth = np.where(per_depth > omni_depth.max(), omni_depth.max(), per_depth)

            if interation_index == 0:
                write_depth = omni_depth[0].copy()
                per_mask_0 = np.where(omni_sems[0][..., 1] == 1, True, False)
                unique_instance = sorted(np.unique(omni_sems[0][..., 0][per_mask_0]))
                for i_index in unique_instance:
                    if i_index != 0:
                        instance = np.where(omni_sems[0][..., 0]==i_index, True, False)
                        intersection = per_mask_0 * instance

                        if intersection.sum() / instance.sum() > 0.5:
                            plane = np.median((write_depth * np.cos(lat2))[instance])
                            # distance = (omni_depth[index+1] * np.cos(lat2)) - plane
                            # weight = distance / distance[instance].max()
                            great_mask = (write_depth * np.cos(lat2) > plane) & instance
                            # omni_depth[index+1][great_mask] = (plane / np.cos(lat2) * (1-weight) + omni_depth[index+1] * (weight))[great_mask]
                            write_depth = np.where(great_mask, plane / np.cos(lat2), write_depth)
                write_depth = np.where(omni_sems[0][..., 1] == 179, omni_depth.max(), write_depth)
                write_exr_file(os.path.join(option.save_dir, option.name + "_depth", "00000.exr"), write_depth.astype(np.float32))
                
            for index in range(0, end_index - start_index - 1, 1):
                before_mask = (move_masks_iteration[index]>0.0)
                back_mask = (move_masks_iteration[index+1]>0.0)
                intersection = before_mask & back_mask
                if (intersection * per_mask[index]).sum() > 0:
                    
                    label, num_label = ndimage.label(intersection)
                    for mask_idx in range(1, num_label+1, 1):

                        mask_idx_mask = (label == mask_idx) * per_mask[index]
                        if mask_idx_mask.sum() > 0:
                            befor_intersection_instance = np.unique(omni_sems[index][..., 0][mask_idx_mask])
                            back_intersection_instance =  np.unique(omni_sems[index + 1][..., 0][mask_idx_mask])
                            if len(befor_intersection_instance) == 1 and len(back_intersection_instance) == 1:
                                omni_diff = (omni_depth[index+1] * np.cos(lat2))[mask_idx_mask].mean() - (omni_depth[index] * np.cos(lat2))[mask_idx_mask].mean()
                                per_diff = (per_depth[index+1] * np.cos(lat2))[mask_idx_mask].mean() - (per_depth[index] * np.cos(lat2))[mask_idx_mask].mean()
                            
                                diff = per_diff - omni_diff

                                back_ = (omni_sems[index + 1][..., 0] == back_intersection_instance)

                                omni_depth[index+1][back_] = omni_depth[index+1][back_] + (diff / np.cos(lat2))[back_]

                            else:
                                for instance_index in back_intersection_instance:
                                    arr = omni_sems[index][..., 0][omni_sems[index + 1][..., 0] == instance_index]
                                    values, counts = np.unique(arr, return_counts=True)
                                    max_index = np.argmax(counts)
                                    
                                    befor_ = np.where(omni_sems[index][..., 0] == values[max_index], True, False)
                                    back_ = np.where(omni_sems[index + 1][..., 0] ==instance_index, True, False)

                                    intersection_mask = (befor_ & back_)
                                    intersection_mask = np.where(per_mask[index] == True, intersection_mask, False)

                                    omni_diff = (omni_depth[index] * np.cos(lat2))[mask_idx_mask].min() - (omni_depth[index + 1] * np.cos(lat2))[mask_idx_mask].min()
                                    per_diff = (per_depth[index] * np.cos(lat2))[mask_idx_mask].mean() - (per_depth[index + 1] * np.cos(lat2))[mask_idx_mask].mean()
                                
                                    diff = omni_diff - per_diff
                                
                                    omni_depth[index+1][back_] = omni_depth[index+1][back_] + (diff / np.cos(lat2))[back_]
                
                write_depth = omni_depth[index+1].copy()
                per_mask = np.where(omni_sems[index+1][..., 1] == 1, True, False)
                unique_instance = sorted(np.unique(omni_sems[index+1][..., 0][per_mask]))
                for i_index in unique_instance:
                    if i_index != 0:
                        instance = np.where(omni_sems[index+1][..., 0]==i_index, True, False)
                        intersection = per_mask * instance

                        if intersection.sum() / instance.sum() > 0.5:
                            plane = np.median((write_depth * np.cos(lat2))[instance])
                            # distance = (omni_depth[index+1] * np.cos(lat2)) - plane
                            # weight = distance / distance[instance].max()
                            great_mask = (write_depth * np.cos(lat2) > plane) & instance
                            # omni_depth[index+1][great_mask] = (plane / np.cos(lat2) * (1-weight) + omni_depth[index+1] * (weight))[great_mask]
                            write_depth = np.where(great_mask, plane / np.cos(lat2), write_depth)
                write_depth = np.where(omni_sems[index + 1][..., 1] == 179, omni_depth.max(), write_depth)
                background_depth_re = np.where(np.logical_and(background_depth_re < write_depth, back_mask==False), write_depth, background_depth_re)
                write_exr_file(os.path.join(option.save_dir, option.name + "_depth", os.path.splitext(file_names[start_index + index + 1])[0] + ".exr"), write_depth.astype(np.float32))

    write_exr_file(os.path.join(option.save_dir, option.name + "_depth", "background.exr"), background_depth_re.astype(np.float32))
    end_time = time.time()
    dura_time = end_time - start_time
    a =0
def run_per_depth(option):

    per_depth_path = os.path.join(option_.save_dir, option.name + "_per_depth")
    if not os.path.exists(per_depth_path):
        os.makedirs(per_depth_path)

    file_names = sorted(os.listdir(os.path.join(option.data_dir, option.name)))
    file_names = []
    for index in range(len(os.listdir(os.path.join(option.data_dir, option.name))) - 1):
        file_names.append(os.path.join(option.data_dir, option.name, str(index) + ".jpg"))
    if "background.png" in file_names:
        file_names.remove("background.png")

    if not os.path.exists(os.path.join(per_depth_path, "per_depth_3.npz")):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        video_depth_anything = VideoDepthAnything(**model_configs["vitl"], metric=True)
        video_depth_anything.load_state_dict(torch.load(os.path.join(option.check_dir, f'metric_video_depth_anything_{"vitl"}.pth'), map_location='cpu',weights_only=True), strict=True)
        video_depth_anything = video_depth_anything.to(device).eval()

        rgbs = []
        for file_name in file_names:
            img = cv2.imread(os.path.join(option.data_dir, option.name, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgbs.append(img)
        rgbs = np.stack(rgbs, axis=0)
        image_patch = equi2pers(torch.tensor(rgbs).float().permute(0, 3, 1, 2), 1, depth_size, mode="bilinear").permute(0, 2, 3, 1, 4)
        # image_patch = equi2pers(torch.tensor(rgbs).float().permute(0, 3, 1, 2), 1, depth_size, mode="bilinear")
        # b = pers2equi(image_patch, 1, depth_size, (option.height, option.width), "111").numpy()[0, 0]
        start_time = time.time()
        for index in range(0, image_patch.shape[-1], 1):
            pred_depth, _ = video_depth_anything.infer_video_depth(image_patch[..., index].numpy().astype(np.uint8), 30, input_size=depth_size)
            np.savez_compressed(os.path.join(per_depth_path, "per_depth_{}.npz".format(index)), pred_depth)
        end_time = time.time()
        dura_time = end_time - start_time

    
def depth_estimation(unik, image, weight, add_length):

    omni_image = torch.from_numpy(image).permute(2, 0, 1)[None].float()
    
    image_add = torch.cat([omni_image, omni_image[..., :add_length]], dim=-1)
    omni_predictions = unik.infer(image_add)
    omni_depth = omni_predictions["distance"].cpu()
    
    omni_depth[..., :add_length] = omni_depth[..., -add_length:] * (1 - weight) + omni_depth[..., :add_length] * weight
    omni_depth = omni_depth[..., :-add_length]
        
    return omni_depth[0,0].numpy()


def depth_to_pointcloud( rgb_img, depth_map,intrinsic_matrix=None, downsample_factor=1):
    """
    将深度图转换为点云，可选添加RGB颜色信息
    
    参数:
    depth_map: 深度图(HxW)，单位建议为米(m)
    rgb_img: 对应彩色图像(HxWx3)
    intrinsic_matrix: 相机内参矩阵[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    downsample_factor: 点云下采样因子(1表示全分辨率)
    
    返回:
    points: (Nx3)点云坐标
    colors: (Nx3)点云颜色(可选)
    """
    # 获取深度图尺寸
    H, W = depth_map.shape[:2]
    
    # 如果没有提供内参矩阵，使用默认值
    if intrinsic_matrix is None:
        # 假设中心点和标准焦距
        intrinsic_matrix = np.array([
            [W * 0.8, 0, W/2],
            [0, H * 0.8, H/2],
            [0, 0, 1]
        ])
    
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(0, W, downsample_factor),
                       np.arange(0, H, downsample_factor))
    
    # 获取对应的深度值
    z = depth_map[v, u]
    
    # 应用下采样掩码（忽略无效深度）
    valid_mask = (z > 0) & (z < np.inf) & (np.isfinite(z))
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]
    
    # 从像素坐标转换到3D点云坐标
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = np.column_stack((x, y, z))
    
    # 处理颜色信息
    colors = None
    if rgb_img is not None:
        # 确保图像类型为uint8且为三通道
        if rgb_img.dtype != np.uint8:
            rgb_img = (rgb_img * 255).astype(np.uint8)
        if len(rgb_img.shape) == 2:  # 如果是灰度图，转为三通道
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
            
        # 获取有效点的颜色
        colors = rgb_img[v, u] / 255.0  # 归一化到[0,1]
    
    return points, colors
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
    # points2, colors2 = equirectangular_to_pointcloud(rgb2, depth2)
    # points = np.concatenate((points, points2), axis=0)
    # colors = np.concatenate((colors, colors2), axis=0)
    # points, colors = depth_to_pointcloud(rgb, depth)
    
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
    image = cv2.imread("D:/linux/github2/test_image/image/cafeteria/00000.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    depth = read_exr_file("D:/linux/github2/test_image/preprocess/cafeteria_init_depth/00000.exr")

    visualize_with_open3d(image, depth)

def run_depth(option):
    
    depth_path = os.path.join(option.save_dir, option.name + "_init_depth")
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    # if not os.path.exists(os.path.join(depth_path, "background.exr")):
    if True:
        unik = UniK3D.from_pretrained(os.path.join(option.check_dir, "unik3d"))
        unik.resolution_level = 9
        unik.interpolation_mode = "bilinear"
        unik.to(device).eval()
    
        file_names = sorted(os.listdir(os.path.join(option.data_dir, option.name)))

        start_time = time.time()
        for file_name in file_names:
            # if not os.path.exists(os.path.join(option.save_dir, option.name + "_init_depth", os.path.splitext(file_name)[0] + ".exr")):
            if True:
        
                image = cv2.imread(os.path.join(option.data_dir, option.name, file_name))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(image).permute(2, 0, 1)

                H, W = image.shape[1:]
                hfov_half = np.pi
                vfov_half = np.pi * H / W

                params = [W, H, hfov_half, vfov_half]
                camera = Spherical(params = torch.tensor([1.0] * 4 + params))
                outputs = unik.infer(rgb=image, camera=camera, normalize=True)
                write_exr_file(os.path.join(option.save_dir, option.name + "_init_depth", os.path.splitext(file_name)[0] + ".exr"), outputs["distance"][0, 0].cpu().numpy())
                
                
        end_time = time.time()
        dura_time = end_time - start_time
 


if __name__ == "__main__":
   
    # Adding necessary input arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="D:/linux/github2/move_scene/scene", type=str)
    parser.add_argument('--save_dir', default="D:/linux/github2/move_scene/scene_condition", type=str)
    parser.add_argument('--name', default="scene2", type=str)

    parser.add_argument('--width', default=432, type=int)
    parser.add_argument('--height', default=240, type=int)
    parser.add_argument('--load_config', type=str, default='D:/linux/github2/3dpb/VideoDepth/assets/')
    parser.add_argument('--check_dir', type=str, default='D:/linux/github2/3dpb/VideoDepth/checkpoints')
    parser.add_argument('--encoder', default="vits")

    # Check for required input
    option_, _ = parser.parse_known_args()
    depth_size = 518
    alpha = torch.tensor(generatemask([depth_size, depth_size]).reshape(1, 1, depth_size,depth_size, 1))
    alpha = torch.repeat_interleave(alpha, patch_num, dim=-1)
    # select device
    device = torch.device("cuda")

    # run_per_depth(option_)
    # run_move_mask(option_)
    # run_background(option_)
    # run_depth(option_)
    run_depth_fusion(option_)




