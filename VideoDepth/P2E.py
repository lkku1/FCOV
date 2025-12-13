import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import time
from os import makedirs
from os.path import join, exists
from E2P import equi2pers
import time
import os


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def pers2equi(pers_img, nrows, patch_size, erp_size, layer_name, save_weight=True):
    device = pers_img.device
    height, width = pair(patch_size)
    erp_h, erp_w = pair(erp_size)
    n_patch = pers_img.shape[-1]
    grid_dir = './grid'
    if not exists(grid_dir):
        makedirs(grid_dir)
    grid_file = join(grid_dir, layer_name + '.pth')

    if not exists(grid_file):
        FOV = torch.tensor([0.6, 1.2], dtype=torch.float32)

        PI = math.pi
        PI_2 = math.pi * 0.5
        PI2 = math.pi * 2

        if nrows == 3:
            num_rows = 3
            num_cols = [3, 4, 3]
            phi_centers = [-59.6, 0, 59.6]
        if nrows == 2:
            num_rows = 2
            num_cols = [4, 4]
            phi_centers = [-35.5, 35.5]
        if nrows == 1:
            num_rows = 1
            num_cols = [4]
            phi_centers = [0.0]
        phi_interval = 180 // num_rows
        all_combos = []

        for i, n_cols in enumerate(num_cols):
            for j in np.arange(n_cols):
                theta_interval = 360 / n_cols
                # theta_center = j * theta_interval + theta_interval / 2
                theta_center = j * theta_interval
                center = [theta_center, phi_centers[i]]
                all_combos.append(center)

        all_combos = np.vstack(all_combos)
        n_patch = all_combos.shape[0]

        center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
        center_point[:, 0] = (center_point[:, 0]) / 360  # 0 to 1
        center_point[:, 1] = (center_point[:, 1] + 90) / 180  # 0 to 1

        cp = center_point * 2 - 1
        cp[:, 0] = cp[:, 0] * PI
        cp[:, 1] = cp[:, 1] * PI_2
        cp = cp.unsqueeze(1)
        """
        lat_grid, lon_grid = torch.meshgrid(torch.linspace(-PI_2, PI_2, erp_h), torch.linspace(-PI, PI, erp_w))
        lon_grid = lon_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1)
        lat_grid = lat_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1)

        cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(lon_grid - cp[..., 0])
        new_x = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0])) / cos_c
        new_y = (torch.cos(cp[..., 1])*torch.sin(lat_grid) - torch.sin(cp[...,1])*torch.cos(lat_grid)*torch.cos(lon_grid-cp[...,0])) / cos_c
        new_x = new_x / FOV[0] / PI   # -1 to 1
        new_y = new_y / FOV[1] / PI_2

        new_x = (new_x + 1) * 0.5 * height
        new_y = (new_y + 1) * 0.5 * width
        new_x = new_x.reshape(num_patch, erp_h, erp_w)
        new_y = new_y.reshape(num_patch, erp_h, erp_w)

        new_x = new_x + shifts.reshape(-1, 1, 1)

        new_x *= erp_mask
        new_y *= erp_mask
        new_x = new_x.sum(0)
        new_y = new_y.sum(0)

        new_x = (new_x / pers_img.shape[-1] - 0.5) * 2
        new_y = (new_y / pers_img.shape[-2] - 0.5) * 2

        new_grid = torch.stack([new_x, new_y], -1)
        torch.save(new_grid, grid_file)
    else:
        new_grid = torch.load(grid_file)
    new_grid = new_grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(pers_img.device)
    img_erp = F.grid_sample(pers_img, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
        """

        lat_grid, lon_grid = torch.meshgrid(torch.linspace(-PI_2, PI_2, erp_h), torch.linspace(-PI, PI, erp_w), indexing="ij")
        lon_grid = lon_grid.float().reshape(1, -1)  # .repeat(num_rows*num_cols, 1)
        lat_grid = lat_grid.float().reshape(1, -1)  # .repeat(num_rows*num_cols, 1)
        cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(
            lon_grid - cp[..., 0])
        new_x = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0])) / cos_c
        new_y = (torch.cos(cp[..., 1]) * torch.sin(lat_grid) - torch.sin(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(
            lon_grid - cp[..., 0])) / cos_c
        new_x = new_x / FOV[0] / PI  # -1 to 1
        new_y = new_y / FOV[1] / PI_2
        cos_c_mask = cos_c.reshape(n_patch, erp_h, erp_w)
        cos_c_mask = torch.where(cos_c_mask > 0, 1, 0)

        w_list = torch.zeros((n_patch, erp_h, erp_w, 4), dtype=torch.float32)

        new_x_patch = (new_x + 1) * 0.5 * height
        new_y_patch = (new_y + 1) * 0.5 * width
        new_x_patch = new_x_patch.reshape(n_patch, erp_h, erp_w)
        new_y_patch = new_y_patch.reshape(n_patch, erp_h, erp_w)
        mask = torch.where((new_x_patch < width) & (new_x_patch > 0) & (new_y_patch < height) & (new_y_patch > 0), 1, 0)
        mask *= cos_c_mask

        x0 = torch.floor(new_x_patch).type(torch.int64)
        x1 = x0 + 1
        y0 = torch.floor(new_y_patch).type(torch.int64)
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, width - 1)
        x1 = torch.clamp(x1, 0, width - 1)
        y0 = torch.clamp(y0, 0, height - 1)
        y1 = torch.clamp(y1, 0, height - 1)

        wa = (x1.type(torch.float32) - new_x_patch) * (y1.type(torch.float32) - new_y_patch)
        wb = (x1.type(torch.float32) - new_x_patch) * (new_y_patch - y0.type(torch.float32))
        wc = (new_x_patch - x0.type(torch.float32)) * (y1.type(torch.float32) - new_y_patch)
        wd = (new_x_patch - x0.type(torch.float32)) * (new_y_patch - y0.type(torch.float32))

        wa = wa * mask.expand_as(wa)
        wb = wb * mask.expand_as(wb)
        wc = wc * mask.expand_as(wc)
        wd = wd * mask.expand_as(wd)

        w_list[..., 0] = wa
        w_list[..., 1] = wb
        w_list[..., 2] = wc
        w_list[..., 3] = wd

        save_file = {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'w_list': w_list, 'mask': mask}
        torch.save(save_file, grid_file)
    else:
        # the online merge really takes time
        # pre-calculate the grid for once and use it during training
        load_file = torch.load(grid_file, weights_only=False)
        # print('load_file')
        x0 = load_file['x0']
        y0 = load_file['y0']
        x1 = load_file['x1']
        y1 = load_file['y1']
        w_list = load_file['w_list']
        mask = load_file['mask']

    w_list = w_list.to(device)
    mask = mask.to(device)
    z = torch.arange(n_patch)
    z = z.reshape(n_patch, 1, 1)
    # start = time.time()
    Ia = pers_img[:, :, y0, x0, z]
    Ib = pers_img[:, :, y1, x0, z]
    Ic = pers_img[:, :, y0, x1, z]
    Id = pers_img[:, :, y1, x1, z]
    # print(time.time() - start)
    output_a = Ia * mask.expand_as(Ia)
    output_b = Ib * mask.expand_as(Ib)
    output_c = Ic * mask.expand_as(Ic)
    output_d = Id * mask.expand_as(Id)

    output_a = output_a.permute(0, 1, 3, 4, 2)
    output_b = output_b.permute(0, 1, 3, 4, 2)
    output_c = output_c.permute(0, 1, 3, 4, 2)
    output_d = output_d.permute(0, 1, 3, 4, 2)
    # print(time.time() - start)
    w_list = w_list.permute(1, 2, 0, 3)
    w_list = w_list.flatten(3)
    w_list *= torch.gt(w_list, 1e-5).type(torch.float32)
    w_list = F.normalize(w_list, p=1, dim=-1).reshape(erp_h, erp_w, n_patch, 4)
    w_list = w_list.unsqueeze(0).unsqueeze(0)
    output = output_a * w_list[..., 0] + output_b * w_list[..., 1] + \
             output_c * w_list[..., 2] + output_d * w_list[..., 3]

    depth = output[:, :-1]
    alpha = output[:, -1:]
    alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)

    output = depth * (alpha / alpha_sum)
    output[output.isnan()] = 0
    output = output.sum(-1)

    return output

def weight_allocate(position_list, length, width=True):
    num = 0
    point_index = 0
    weight = torch.zeros(len(position_list))
    if width == True:
        col = 0
        index = 1
        shift = length
    else:
        col = 1
        index = 0
        shift = length
    for i in range(len(position_list)):
        if (position_list[i][col] == position_list[point_index][col]):
            num = num + 1
        else:
            for j in range(num):
                if num == shift:
                    weight[point_index + j] = 0.5
                else:
                    weight[point_index + j] = (position_list[point_index + j][index] - position_list[point_index][index] + 1) / num

            point_index = point_index + num
            num = 1
    return weight

# def pers2equi(erp_img, fov, nrows, patch_size, erp_size, layer_name, save_weight=True):
#     if nrows == 3:
#         num_cols = [3, 4, 3]
#         phi_centers = [-59.6, 0, 59.6]
#     if nrows == 2:
#         num_cols = [4, 4]
#         phi_centers = [-30.5, 30.5]
#     if nrows == 1:
#         num_cols = [1, 4, 1]
#         phi_centers = [-90.0, 0.0, 90.0]

#     all_combos = []
#     for i, n_cols in enumerate(num_cols):
#         for j in np.arange(n_cols):
#             theta_interval = 360 / n_cols
#             theta_center = j * theta_interval + theta_interval / 2

#             center = [theta_center, phi_centers[i]]
#             all_combos.append(center)
    
#     all_combos = np.vstack(all_combos)
#     num_patch = all_combos.shape[0]
#     center_point =torch.from_numpy(all_combos).float().to(erp_img.device)

#     lon = torch.linspace(-np.pi, np.pi, patch_size, device=erp_img.device)
#     lat = torch.linspace(-np.pi/2, np.pi/2, patch_size, device=erp_img.device)
#     lon_grid, lat_grid = torch.meshgrid(lon, lat)
    
#     # 球面坐标转笛卡尔坐标
#     x = torch.cos(lat_grid) * torch.sin(lon_grid)
#     y = torch.sin(lat_grid)
#     z = torch.cos(lat_grid) * torch.cos(lon_grid)
#     rays = torch.stack([x, y, z], dim=-1)
    
#      # 应用相机旋转
#     rot_y = torch.zeros((center_point.shape[0], 3, 3), device=erp_img.device)
#     rot_y[:, 0, 0] = torch.cos(center_point[:, 1])
#     rot_y[:, 0, 2] = torch.sin(center_point[:, 1])
#     rot_y[:, 1, 1] = torch.ones_like(rot_y[:, 1, 1])
#     rot_y[:, 2, 0] = -torch.sin(center_point[:, 1])
#     rot_y[:, 2, 2] = torch.cos(center_point[:, 1])

#     rot_x = torch.zeros((center_point.shape[0], 3, 3), device=erp_img.device)
#     rot_x[:, 0, 0] = torch.ones_like(rot_x[:, 0, 0])
#     rot_x[:, 1, 1] = torch.cos(center_point[:, 0])
#     rot_x[:, 1, 2] = -torch.sin(center_point[:, 0])
#     rot_x[:, 2, 1] = torch.sin(center_point[:, 0])
#     rot_x[:, 2, 2] = torch.cos(center_point[:, 0])

#     rays = torch.einsum('bij,bhwj->bhwi', rot_x @ rot_y, rays)
    
#     # 透视投影
#     f = persp_img.shape[1] / (2 * np.tan(np.radians(fov / 2)))
#     u_persp = (rays[..., 0] / rays[..., 2] * f) + persp_img.shape[1] / 2
#     v_persp = (-rays[..., 1] / rays[..., 2] * f) + persp_img.shape[0] / 2
    
#     # 仅保留有效区域（需处理遮挡问题）
#     mask = (rays[..., 2] < 0) & (u_persp >= 0) & (u_persp < persp_img.shape[1]) & (v_persp >= 0) & (v_persp < persp_img.shape[0])
#     equi_img = torch.zeros(equi_shape + (persp_img.shape[-1],), device=persp_img.device)
#     grid = torch.stack([u_persp / (persp_img.shape[1] - 1) * 2 - 1, 
#                         v_persp / (persp_img.shape[0] - 1) * 2 - 1], dim=-1).unsqueeze(0)
#     sampled = torch.nn.functional.grid_sample(
#         persp_img.permute(2, 0, 1).unsqueeze(0).float(),
#         grid, align_corners=True, mode='bilinear'
#     ).squeeze(0).permute(1, 2, 0)
#     equi_img[mask] = sampled[mask]
#     return equi_img

if __name__ == "__main__":
    img = cv2.imread('moon.jpg', cv2.IMREAD_COLOR)
    img_new = img.astype(np.float32)
    img_new = np.transpose(img_new, [2, 0, 1])
    img_new = torch.from_numpy(img_new)
    img_new = img_new.unsqueeze(0)
    fov = (80, 80)

    pers, _, _, _ = equi2pers(img_new, nrows=3, fov=fov, patch_size=(64, 64))
    # pers = F.unfold(pers, kernel_size=64, stride=64)
    # pers = pers.reshape(1, 3, 64, 64, -1)

    erp = pers2equi(pers, nrows=3, fov=fov, patch_size=(64, 64), erp_size=(256, 512), layer_name="ss")

    # n_patch = pers.shape[-1]
    img_erp_int = erp[0, ...].permute(1, 2, 0).numpy()
    img_erp_int = img_erp_int  # * 255
    img_erp_int = img_erp_int.astype(np.uint8)
    cv2.imwrite('interp_erp.png', img_erp_int)
