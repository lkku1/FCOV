# OUR
from utils import ImageDataset, generatemask

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

Fov = (0.8, 0.4)
depth_size = (518, 518)
Nrows = 1
device = None

def write_depth(path, depth):
    depth = depth * 2000
    # depth = np.clip(depth, 0, 60000)
    cv2.imwrite(path + ".png", depth.astype(np.uint16))
    return


# warnings.simplefilter('ignore', np.RankWarning)
def depth_estimation(omni_rgb_ori, sem_image, omni_camera, per_camera, alpha):
    
    omni_rgb = torch.from_numpy(omni_rgb_ori).permute(2, 0, 1).float()
    omni_sem = torch.from_numpy(sem_image)[None, None].float().to(alpha.device)

    add_length = int((omni_rgb.shape[-1] / 140)) * 14
    omni_rgb_add = torch.cat([omni_rgb, omni_rgb[..., :add_length]], dim=-1)

    omni_predictions = unik.infer(omni_rgb_add, omni_camera.clone())
    omni_depth = omni_predictions["distance"]
    
    _, weight_width = torch.meshgrid(torch.linspace(0, 1, omni_rgb_add.shape[1]), torch.linspace(0, 1, add_length), indexing="ij")
    weight_width = weight_width[None,None].to(device)
    omni_depth[..., :add_length] = omni_depth[..., -add_length:] * (1 - weight_width) + omni_depth[..., :add_length] * weight_width
    omni_depth = omni_depth[..., :-add_length]
    
    per_rgb = equi2pers(torch.from_numpy(omni_rgb_ori)[None].permute(0, 3, 1, 2).float(), Fov, Nrows, depth_size)
    omni_info = torch.cat([omni_depth, omni_sem],dim=1)
    omni_per_info = equi2pers(omni_info, Fov, Nrows, depth_size)
    omni_per_depth = omni_per_info[:, :1]
    omni_per_sem= omni_per_info[:, 1:]

    omni_per_sem = torch.where(omni_per_sem==2.0, True, False).to(torch.bool)
    kernel = torch.ones(11, 11, device=alpha.device)
    
    per_depths = []
    for index in range(per_rgb.shape[0]):

        source_depth = omni_per_depth[index:index+1]
        source_sem = omni_per_sem[index:index+1]
        if index < per_rgb.shape[0] - 1:

            # if (source_sem == 2.0).all() == True:
            #     target_depth = torch.ones_like(source_depth) * omni_depth.max()
            # else:
            per_predictions = unik.infer(per_rgb[index:index+1], per_camera.clone())
            target_depth = per_predictions["distance"]

            per_sem_mask = kornia.morphology.dilation(source_sem, kernel)
            confidence_mask = (1.0 - per_sem_mask).to(torch.bool)
        
            valid_d1 = source_depth[confidence_mask].reshape(-1).cpu().numpy()
            valid_d2 = target_depth[confidence_mask].reshape(-1).cpu().numpy()

            if confidence_mask.sum() <= 100:
                target_depth = source_depth.clone()
            else:
                # slope, intercept = np.polyfit(valid_d2, valid_d1, deg=1)
                p = Polynomial.fit(valid_d2, valid_d1, deg=1).convert().coef
              
                a = torch.tensor(p[1]).to(device) 
                b = torch.tensor(p[0]).to(device)
                # k = torch.sum(valid_d1 * valid_d2) / torch.sum(valid_d2 ** 2)
                target_depth = a * target_depth + b


        else:
            target_depth = source_depth.clone()
        
        target_depth = torch.where(torch.logical_and(target_depth < source_depth, target_depth > 1.0), target_depth, source_depth)
        target_depth[source_sem] = omni_depth.max()
       
        per_depths.append(target_depth)

    per_depth = torch.cat(per_depths)[None].permute(0, 2, 3, 4, 1)
    per_dep_refine = torch.cat([per_depth, alpha], dim=1)
    per_depth = pers2equi(per_dep_refine, Fov, Nrows, depth_size, (omni_rgb.shape[1], omni_rgb.shape[2]), "Patch_P2E2048_1")
    per_depth = per_depth / per_depth.max() * 30
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

# def move_camera(vis, delta_z):
#     ctr = vis.get_view_control()
#     params = ctr.convert_to_pinhole_camera_parameters()
    
#     # 修改相机位置（沿Z轴移动）
#     extrinsic = params.extrinsic
#     extrinsic[2, 3] += delta_z  # 第3行第4列是Z坐标
    
#     # 应用新参数
#     params.extrinsic = extrinsic
#     ctr.convert_from_pinhole_camera_parameters(params)
#     vis.update_renderer()
# def key_callback(vis, key_code):
#     if key_code == ord('W'):  # 按W向前
#         move_camera(vis, delta_z=-0.1)
#     elif key_code == ord('S'):  # 按S向后
#         move_camera(vis, delta_z=0.1)
#     return False
def move_camera(vis, delta):
    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()
    new_extrinsic = np.copy(cam.extrinsic)
    new_extrinsic[:3, 3] += delta
    cam.extrinsic = new_extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam)
    return False

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
    '''
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    rgb, 
    depth,
    depth_scale=30000.0,  # 深度值的缩放因子（例如Kinect的深度单位为毫米）
    depth_trunc=1.0,     # 深度截断值（超出此值的点将被忽略）
    convert_rgb_to_intensity=False  # 保留RGB颜色信息
)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=518,          # 图像宽度
    height=518,         # 图像高度
    fx=500.0,           # x轴焦距
    fy=500.0,           # y轴焦距
    cx=259.0,           # 光学中心x坐标
    cy=259.0            # 光学中心y坐标
)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, 
    intrinsic
)
    o3d.visualization.draw_geometries([point_cloud])
    '''

def vis():
    image = o3d.io.read_image("D:/linux/github2/3dpb/00000.jpg")
    depth = o3d.io.read_image("D:/linux/github2/3dpb/2.png")
    visualize_with_open3d(image, depth)

def run(dataset, option):
   
    global unik
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
    alpha = torch.repeat_interleave(alpha, 6, dim=-1).to(device)

    for image_ind, images in enumerate(dataset):
        # if not os.path.exists(os.path.join(option.output_dir, str(images.name) + ".png")):
        if True:
            omni_dep = depth_estimation(images.rgb_image, images.sem_image, omni_camera, per_camera, alpha)
            write_depth(os.path.join(option.output_dir, str(images.name)), omni_dep)
            
    print("depth finished")

if __name__ == "__main__":
    
    # Adding necessary input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="D:/linux/github2/test_image/image/cafeteria", type=str)
    parser.add_argument('--output_dir', default="D:/linux/github2/test_image/preprocess/cafeteria_depth", type=str)
    parser.add_argument('--width', default=2048, type=int)
    parser.add_argument('--height', default=1024, type=int)
    parser.add_argument('--background', default=False, type=bool)
    parser.add_argument('--encoder', default="vits")
    parser.add_argument('--load_from', type=str, default='D:/linux/github2/3dpb/VideoDepth/checkpoints/unik3d')
    parser.add_argument('--load_config', type=str, default='D:/linux/github2/3dpb/VideoDepth/assets/')
    parser.add_argument('--gpu', default=2, type=int)
    # Check for required input
    option_, _ = parser.parse_known_args()
    print(option_)

    # select device
    device = torch.device("cuda:" + str(option_.gpu))

    # Create dataset from input images
    dataset_ = ImageDataset(option_.data_dir, option_.output_dir, option_.width, option_.height, option_.background)

    # Run pipeline
    run(dataset_, option_)

