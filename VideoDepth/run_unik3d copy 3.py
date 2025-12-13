# OUR
from unik_util import ImageDataset, generatemask, write_exr_file, read_exr_file

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
import open3d as o3d
from video_depth_anything.video_depth import VideoDepthAnything

from pietorch import blend
from scipy.ndimage import label


Fov = (140, 140)
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
        
def run_per_rgb(option):

    per_rgb_dir = os.path.join(option.save_dir, option.name + "_per_rgb")
    if not os.path.exists(per_rgb_dir):
        os.mkdir(per_rgb_dir)

    file_names = sorted(os.listdir(os.path.join(option.data_dir, option.name)))
    if "background.png" in file_names:
        file_names.remove("background.png")

    if not os.path.exists(os.path.join(per_rgb_dir, file_names[-1])):
        for file_name in file_names:
            if not os.path.exists(os.path.join(per_rgb_dir, file_name)):

                rgb_image = load_image(os.path.join(option.data_dir, option.name, file_name))
                rgb_patchs = equi2pers(rgb_image[None].float(), Fov, Nrows, depth_size, mode="bilinear").cpu().numpy()
                rgb_sets = []
                for index in range(0, get_patch_num(), 1):
                    rgb_patch = cv2.cvtColor(rgb_patchs[index].transpose(1,2,0).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    rgb_sets.append(rgb_patch)
                
                cat_rgb_patch = np.concatenate(rgb_sets, axis=1)
                cv2.imwrite(os.path.join(per_rgb_dir, file_name), cat_rgb_patch)

def run_background(option, step=5):

    if not os.path.exists(os.path.join(option.data_dir, option.name, "background.png")):

        depth_set = []
        for idx in range(0, get_patch_num(), 1):
            data = np.load(os.path.join(option.save_dir, option.name + "_per_depth", str(idx) + ".npz"))
            depth_set.append(data["arr_0"])
            data.close()
        depth = np.concatenate(depth_set, axis=-1)[::step]
        depth_means = np.mean(depth, axis=0)

        file_names = sorted(os.listdir(os.path.join(option.save_dir, option.name + "_per_rgb")))
        if "background.png" in file_names:
            file_names.remove("background.png")
    
        rgb_set = []
        for file_index in range(0, len(file_names), step):
            image = cv2.imread(os.path.join(option.save_dir, option.name + "_per_rgb", file_names[file_index]))
            rgb_set.append(image)
        rgb = np.stack(rgb_set, axis=0)
        rgb_mean = np.mean(rgb, axis=0)

        mask = np.where(depth > depth_means[None], True, False).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        for idx in range(0, mask.shape[0], 1):
            mask[idx] = cv2.erode(mask[idx], kernel)
        mask = mask[..., None].astype(np.bool_)
        mask_sum = mask.sum(axis=0)
        
        rgb = (rgb * mask).sum(axis=0) / (mask_sum + 1e-6)
        rgb = np.where(mask_sum == 0, rgb_mean, rgb)

        cv2.imwrite(os.path.join(option.save_dir, option.name + "_per_rgb",  "background.png"), rgb.astype(np.uint8))

        rgb = np.split(rgb, get_patch_num(), axis=1)
        rgb  = torch.tensor(np.stack(rgb, axis=0)[None].transpose(0, 4, 2, 3, 1))

        alpha = torch.tensor(generatemask(depth_size).reshape(1, 1, depth_size[0],depth_size[1], 1))
        alpha = torch.repeat_interleave(alpha, get_patch_num(), dim=-1)

        cat_rgb = torch.cat([rgb, alpha], dim=1)
        omni_background = pers2equi(cat_rgb, Fov, Nrows, depth_size, (option.height, option.width), "Patch_P2E_{}".format(str(option.width))).numpy()[0].transpose(1,2,0)
       
        cv2.imwrite(os.path.join(option.data_dir, option.name,  "background.png"), omni_background.astype(np.uint8))


def run_move_mask(option):
    
    move_path = os.path.join(option.save_dir, option.name + "_move")
    if not os.path.exists(move_path):
        os.makedirs(move_path)

    depth_set = []
    for idx in range(0, get_patch_num(), 1):
        data = np.load(os.path.join(option.save_dir, option.name + "_per_depth", str(idx) + ".npz"))
        depth_set.append(data["arr_0"])
        data.close()
    depth = np.stack(depth_set, axis=0)

    file_names = sorted(os.listdir(os.path.join(option.save_dir, option.name + "_per_rgb")))

    if not os.path.exists(os.path.join(move_path, "move.npz")):
        background = cv2.imread(os.path.join(option.data_dir, option.name, "background.png"))
        # background_sem = np.load(os.path.join(option.save_dir, option.name + "_sem", "background.npy"))
        # background_sky_sem = np.where(background_sem[..., 1] == 178, 1, 0)
        # if background_sky_sem.sum() > 0:
        #     background_sky_sem = cv2.dilate(background_sky_sem.astype(np.uint8), np.ones((5, 5)), iterations=1)
        #     background_sem_patch = equi2pers(torch.from_numpy(background_sky_sem[None, None]).float(), Fov, Nrows, depth_size, mode="bilinear").numpy()
        #     background_sem_patch = background_sem_patch.repeat(depth.shape[1], axis=1).astype(np.bool_)
        #     depth[background_sem_patch] = depth[~background_sem_patch].min()
        depth = np.clip(depth, 0, 80)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        
        background_substractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,  detectShadows=False)
        background_substractor.apply(background)

        depth_substractors = []
        for index in range(0, get_patch_num(), 1):
            depth_substractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,  detectShadows=False)
            depth_substractor.apply(depth_norm[index].max(0).astype(np.uint8))
            depth_substractors.append(depth_substractor)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        alpha = torch.tensor(generatemask(depth_size).reshape(1, 1, depth_size[0],depth_size[1], 1))
        alpha = torch.repeat_interleave(alpha, get_patch_num(), dim=-1)

        move_masks = []
        for file_index in range(0, len(file_names), 1):
            
            image = cv2.imread(os.path.join(option.data_dir, option.name, file_names[file_index]))
            image_mask = background_substractor.apply(image, learningRate=0)
            image_mask =cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE, kernel)

            move_mask_patchs = []
            for index in range(get_patch_num()):
                depth_mask_patch = depth_substractors[index].apply(depth_norm[index, file_index].astype(np.uint8), learningRate=0)
                move_mask_patchs.append(depth_mask_patch)

            move_mask_patch = np.stack(move_mask_patchs, axis=-1)[None, None]
            move_mask_patch = np.concatenate((move_mask_patch, alpha), axis=1)
            depth_mask = pers2equi(torch.tensor(move_mask_patch), Fov, Nrows, depth_size, (option_.height, option_.width), "Patch_P2E_{}".format(str(option.width))).numpy()[0, 0]
            depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
            
            move_mask = image_mask.astype(np.bool_) & depth_mask.astype(np.bool_)
            sem = np.load(os.path.join(option.save_dir, option.name+ "_sem", file_names[file_index][:-4] + ".npy"))
            move_mask = np.where(sem[..., 1] == 178, False, move_mask)

            instance_mask = sem[..., 0][move_mask]
            unique_instance = sorted(np.unique(instance_mask))
            remain_instance = np.zeros_like(depth_mask)
            for i_index in unique_instance:
                instance_index = np.where(sem[..., 0] == i_index, 1, 0)
                insert_index = move_mask[instance_index.astype(np.bool_)]
                if insert_index.sum() / instance_index.sum() > 0.1:
                    remain_instance = remain_instance + instance_index * 255
            # remain_instance = (remain_instance.astype(np.bool_) | move_mask) * 255
            move_masks.append(remain_instance)
        move_masks = np.stack(move_masks, axis=0).astype(np.bool_)
        np.savez_compressed(os.path.join(move_path, "move.npz"), move_masks)

def run_depth_fusion(option):

    depth_path = os.path.join(option.save_dir, option.name + "_depth")
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    file_names = sorted(os.listdir(os.path.join(option.save_dir, option.name + "_per_rgb")))
    if not os.path.exists(os.path.join(option.save_dir, option.name + "_depth", os.path.splitext(file_names[-1])[0] + ".exr")):
        depth_set = []
        for idx in range(0, get_patch_num(), 1):
            data = np.load(os.path.join(option.save_dir, option.name + "_per_depth", str(idx) + ".npz"))
            depth_set.append(data["arr_0"])
            data.close()
        depth = torch.tensor(np.stack(depth_set, axis=0))

        background_depth = read_exr_file(os.path.join(option.save_dir, option.name + "_depth",  "background.exr"))

        backper_depths = equi2pers(torch.from_numpy(background_depth.copy())[None, None].float(), Fov, Nrows, depth_size)
        alpha = torch.tensor(generatemask(depth_size).reshape(1, 1, depth_size[0],depth_size[1], 1))
        alpha = torch.repeat_interleave(alpha, get_patch_num(), dim=-1)

        move_data = np.load(os.path.join(option.save_dir, option.name + "_move", "move.npz"))
        move_masks = torch.from_numpy(move_data["arr_0"])
        move_data.close()
        
        move_patchs = equi2pers(move_masks[:, None].float(), Fov, Nrows, depth_size).bool().reshape(-1, get_patch_num(), 1, depth_size[0], depth_size[1])
        
        
        for file_idx in range(0, move_patchs.shape[0], 1):

            blend_depths = []
            for idx in range(0, get_patch_num(), 1):
                fit_mask = move_patchs[file_idx:file_idx+1, idx, 0]
                if fit_mask.any():
                    
                    label_array, label_num = label(fit_mask[0].numpy().astype(np.uint8), structure=np.ones((3, 3)))
                    label_array = torch.from_numpy(label_array)

                    back_dep = backper_depths[idx].clone()
                    for label_idx in range(1, label_num + 1, 1):
                        label_idx_map = torch.where(label_array==label_idx, True, False)

                        label_idx_dep = backper_depths[idx, 0][label_idx_map]
                        min_depth = label_idx_dep.min()
                        back_dep = torch.where(label_idx_map== True, min_depth, back_dep)

                    blend_depth = blend(back_dep, depth[idx, file_idx:file_idx+1, 1:, 1:], fit_mask[0, 1:, 1:].float(), torch.tensor([1, 1]), True, channels_dim=0)
                    
                else:
                    blend_depth = backper_depths[idx]

                blend_depths.append(blend_depth)
            
            blend_depths = torch.stack(blend_depths, dim=-1)[None]
            blend_depths = torch.cat([blend_depths, alpha], dim=1)

            omni_depth = pers2equi(blend_depths, Fov, Nrows, depth_size, (option.height, option.width), "Patch_P2E_{}".format(str(option.width))).numpy()[0, 0]
            write_exr_file(os.path.join(option.save_dir, option.name + "_depth", os.path.splitext(file_names[file_idx])[0] + ".exr"), omni_depth)

def run_per_depth(option):

    per_depth_path = os.path.join(option_.save_dir, option.name + "_per_depth")

    if not os.path.exists(per_depth_path):
        os.makedirs(per_depth_path)

    file_names = sorted(os.listdir(os.path.join(option.data_dir, option.name)))
    if "background.png" in file_names:
        file_names.remove("background.png")

    if not os.path.exists(os.path.join(per_depth_path, str(get_patch_num() - 1) + ".npz")):

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        video_depth_anything = VideoDepthAnything(**model_configs["vitl"])
        video_depth_anything.load_state_dict(torch.load(os.path.join(option.check_dir, f'metric_video_depth_anything_{"vitl"}.pth'), map_location='cpu', weights_only=True), strict=True)
        video_depth_anything = video_depth_anything.to(device).eval()

        frame_patch = []
        for index in range(0, get_patch_num(), 1):
            frame_patch.append([])

        for file_name in file_names:
            cat_frame_patch = cv2.imread(os.path.join(option.save_dir, option.name + "_per_rgb", file_name))
            cat_frame_patch = cv2.cvtColor(cat_frame_patch, cv2.COLOR_BGR2RGB)
            rgb_patchs = np.split(cat_frame_patch, get_patch_num(), axis=1)
            for index in range(0, get_patch_num(), 1):
                frame_patch[index].append(rgb_patchs[index])

        for index in range(0, get_patch_num(), 1):
            frames, target_fps = np.stack(frame_patch[index], axis=0), 30
            depth, _ = video_depth_anything.infer_video_depth(frames, target_fps, input_size=depth_size[0])
            np.savez_compressed(os.path.join(per_depth_path, "{}.npz".format(str(index))), depth)
           
    
def depth_estimation(unik, omni_rgb_ori, sky_sem, omni_camera):
    
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
    
    # per_rgb = equi2pers(torch.from_numpy(omni_rgb_ori)[None].permute(0, 3, 1, 2).float(), Fov, Nrows, depth_size)
    # omni_info = torch.cat([omni_depth, omni_sem],dim=1)
    # omni_per_info = equi2pers(omni_info, Fov, Nrows, depth_size)
    # omni_per_depth = omni_per_info[:, :1]
    # omni_per_sem= omni_per_info[:, 1:]

    # omni_per_sem = torch.where(omni_per_sem > 0.0, True, False)
    # kernel = torch.ones(11, 11, device=device)
    
    # per_depths = []
    # for index in range(per_rgb.shape[0]):

    #     source_depth = omni_per_depth[index:index+1]
    #     source_sem = omni_per_sem[index:index+1]
    #     if index < per_rgb.shape[0] - 1:

    #         per_predictions = unik.infer(per_rgb[index:index+1], per_camera.clone())
    #         target_depth = per_predictions["distance"]

    #         per_sem_mask = kornia.morphology.dilation(source_sem, kernel)
    #         confidence_mask = (1.0 - per_sem_mask).to(torch.bool)
        
    #         valid_d1 = source_depth[confidence_mask].reshape(-1).cpu().numpy()
    #         valid_d2 = target_depth[confidence_mask].reshape(-1).cpu().numpy()

    #         if confidence_mask.sum() <= 100:
    #             target_depth = source_depth.clone()
    #         else:
    #             valid_d1_mean, valid_d1_std = valid_d1.mean(), valid_d1.std()
    #             valid_d2_mean, valid_d2_std = valid_d2.mean(), valid_d2.std()

    #             slope = valid_d1_std / (valid_d2_std + 1e-6)
    #             intercept = valid_d1_mean - slope * valid_d2_mean

    #             target_depth = slope * target_depth + intercept
    #     else:
    #         target_depth = source_depth.clone()
        
    #     target_depth = torch.where(torch.logical_and(target_depth < source_depth, target_depth > 1.0), target_depth, source_depth)
    omni_depth[omni_sem.bool()] = omni_depth.max()

    # per_depths.append(target_depth)

    # per_depth = torch.cat(per_depths)[None].permute(0, 2, 3, 4, 1)
    # per_dep_refine = torch.cat([per_depth, alpha], dim=1)
    # per_depth = pers2equi(per_dep_refine, Fov, Nrows, depth_size, (omni_rgb.shape[1], omni_rgb.shape[2]), "Patch_P2E2048_1")

    torch.cuda.empty_cache()
    return omni_depth[0,0].cpu().numpy()

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

def run_depth(option):
    
    depth_path = os.path.join(option.save_dir, option.name + "_depth")
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    if not os.path.exists(os.path.join(depth_path, "background.exr")):
        unik = UniK3D.from_pretrained(os.path.join(option.check_dir, "unik3d")).to(device)
        unik.resolution_level = 9
        unik.interpolation_mode = "bilinear"
        unik.to(device).eval()
        
        omni_camera_path = os.path.join(option.load_config, "equirectangular.json")
        with open(omni_camera_path, "r") as f:
            camera_dict = json.load(f)
        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        omni_camera = eval(name)(params=params)

        # per_camera_path = os.path.join(option.load_config, "pinhole.json")
        # with open(per_camera_path, "r") as f:
        #     camera_dict = json.load(f)
        # params = torch.tensor(camera_dict["params"])
        # name = camera_dict["name"]
        # per_camera = eval(name)(params=params)

        file_names = sorted(os.listdir(os.path.join(option.data_dir, option.name)))
        for file_name in file_names:
            image = cv2.imread(os.path.join(option.data_dir, option.name, file_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            sem = np.load(os.path.join(option.save_dir, option.name + "_sem", os.path.splitext(file_name)[0] + ".npy"))[..., 1]
            sky = np.where(sem == 178, 1, 0)
    
            depth = depth_estimation(unik, image, sky, omni_camera)
            write_exr_file(os.path.join(option.save_dir, option.name + "_depth", os.path.splitext(file_name)[0] + ".exr"), depth)


if __name__ == "__main__":
    
    # Adding necessary input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="D:/linux/github2/360_train/480p", type=str)
    parser.add_argument('--save_dir', default="D:/linux/github2/360_train/480p_condition", type=str)
    parser.add_argument('--name', default="019cc67f-512f-4b8a-96ef-81f806c86ce1_0", type=str)

    parser.add_argument('--width', default=432, type=int)
    parser.add_argument('--height', default=240, type=int)
    parser.add_argument('--load_config', type=str, default='D:/linux/github2/3dpb/VideoDepth/assets/')
    parser.add_argument('--check_dir', type=str, default='D:/linux/github2/3dpb/VideoDepth/checkpoints')
    parser.add_argument('--encoder', default="vits")
    parser.add_argument('--function', type=str, choices=["stage1", "stage2"])
    
    # Check for required input
    option_, _ = parser.parse_known_args()

    # a = read_exr_file("D:/linux/github2/360_train/480p_condition/019cc67f-512f-4b8a-96ef-81f806c86ce1_0_depth/background.exr")
    # b = read_exr_file("D:/linux/github2/360_train/480p_condition/019cc67f-512f-4b8a-96ef-81f806c86ce1_0_depth/00000.exr")
    # a = np.load("D:/linux/github2/360_train/480p_condition/019cc67f-512f-4b8a-96ef-81f806c86ce1_0_per_depth/5.npz")["arr_0"]
    # c = a / a.max() * 255
    # d = b / b.max() * 255
    option_.function = "stage2"
    # select device
    device = torch.device("cuda")
    if option_.function == "stage1":
        print("start stage1")
        run_per_rgb(option_)
        run_per_depth(option_)
    elif option_.function == "stage2":
        print("stage2")
        run_background(option_)
        run_depth(option_)
        run_move_mask(option_)
        run_depth_fusion(option_)


