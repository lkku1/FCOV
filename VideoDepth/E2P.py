import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2


# generate patches in a closed-form
# the transformation and equation is referred from http://blog.nitishmutha.com/equirectangular/360degree/2017/06/12/How-to-project-Equirectangular-image-to-rectilinear-view.html
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def uv2xyz(uv):
    xyz = np.zeros((*uv.shape[:-1], 3), dtype=np.float32)
    xyz[..., 0] = np.multiply(np.cos(uv[..., 1]), np.sin(uv[..., 0]))
    xyz[..., 1] = np.multiply(np.cos(uv[..., 1]), np.cos(uv[..., 0]))
    xyz[..., 2] = np.sin(uv[..., 1])
    return xyz

#bilinear
def equi2pers(erp_img, nrows, patch_size, mode="nearest"):
   
    bs, _, erp_h, erp_w = erp_img.shape
    height, width = pair(patch_size)
    # fov_h, fov_w = pair(fov)
    FOV = torch.tensor([0.6, 1.2], dtype=torch.float32)

    PI = math.pi
    PI_2 = math.pi * 0.5
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
    screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)   # 0 to 1 (h ,w)

    if nrows == 3:
        num_cols = [3, 4, 3]
        phi_centers = [-59.6, 0, 59.6]
    if nrows == 2:
        num_cols = [4, 4]
        phi_centers = [-35.5, 35.5]
    if nrows == 1:
        num_cols = [4]
        phi_centers = [0.0]

    # phi_interval = 180 // num_rows
    all_combos = []
    # erp_mask = []
    for i, n_cols in enumerate(num_cols):
        for j in np.arange(n_cols):
            theta_interval = 360 / n_cols
            # theta_center = j * theta_interval + theta_interval / 2
            theta_center = j * theta_interval
            center = [theta_center, phi_centers[i]]
            all_combos.append(center)
            
    all_combos = np.vstack(all_combos)
    num_patch = all_combos.shape[0]

    center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
    center_point[:, 0] = (center_point[:, 0]) / 360  # 0 to 1
    center_point[:, 1] = (center_point[:, 1] + 90) / 180  # 0 to 1

    cp = center_point * 2 - 1  # -1 to 1
    cp[:, 0] = cp[:, 0] * PI   # -pi to pi
    cp[:, 1] = cp[:, 1] * PI_2  # -pi_2 to pi_2
    cp = cp.unsqueeze(1)
    convertedCoord = screen_points * 2 - 1  # -1 to 1 (h, w)
    convertedCoord[:, 0] = convertedCoord[:, 0] * PI  # -PI to PI w
    convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2   # -PI_2 to PI_2 h
    convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)  # fov (h, w)
    convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

    x = convertedCoord[:, :, 0]  #-fov to fov h
    y = convertedCoord[:, :, 1]  #-fov to fov w

    rou = torch.sqrt(x ** 2 + y ** 2)
    c = torch.atan(rou)
    sin_c = torch.sin(c)
    cos_c = torch.cos(c)
    lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
    lon = cp[:, :, 0] + torch.atan2(x * sin_c,
                                    rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)

    lat_new = lat / PI_2  #
    lon_new = lon / PI
    lon_new[lon_new > 1] -= 2
    lon_new[lon_new < -1] += 2

    lon_new = lon_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch * width)
    lat_new = lat_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch * width)
    grid = torch.stack([lon_new, lat_new], -1)
    grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)

    pers = F.grid_sample(erp_img, grid, mode=mode, padding_mode='border', align_corners=False)
    pers = F.unfold(pers, kernel_size=(height, width), stride=(height, width))
    pers = pers.reshape(bs, -1, height, width, num_patch)

    return pers

if __name__ == '__main__':
    img = cv2.imread('inputs/10.jpg', cv2.IMREAD_COLOR)
    img_new = img.astype(np.float32)
    img_new = torch.from_numpy(img_new).permute(2, 0, 1)
    img_new = img_new.unsqueeze(0)
    pers, _, _, _ = equi2pers(img_new, nrows=2, fov=(52, 52), patch_size=(64, 64), index=0)
    pers = pers[0, :, :, :, 0].numpy()
    pers = pers.transpose(1, 2, 0).astype(np.uint8)
    cv2.imwrite('pers.png', pers)
    print(pers.shape)
