import os
import cv2
import torch
import random
import einops
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import build_all_layer_point_grids

from ..dataset_utils import readRGB, processMultiSeg, preprocess



class Example_eval_dataset(Dataset):
    def __init__(
        self, 
        data_dirs, 
        ref_sam, 
        seqs = None, 
        flow_gaps = [1, -1, 2, -2],
        num_gridside = 10,
    ):
        self.flow_dir = data_dirs[0]
        self.rgb_dir = data_dirs[1]
        self.ref_sam = ref_sam
        self.flow_gaps = flow_gaps
        self.preprocess = preprocess
        self.transform = ResizeLongestSide(1024)
        self.point_grids = build_all_layer_point_grids(n_per_side = num_gridside, n_layers = 0, scale_per_layer = 1)[0]

        self.seqs = seqs

        self.rgb_paths = []
        self.flow_paths = []

        for filename in sorted(os.listdir(self.rgb_dir)):
            flow_path = os.path.join(self.flow_dir, filename)
            rgb_path = os.path.join(self.rgb_dir, filename)
            self.flow_paths.append(flow_path)
            self.rgb_paths.append(rgb_path)

        print("Dataset consists of {} images".format(len(self.flow_paths)))

    def __len__(self):
        return len(self.flow_paths)

    def val_loading(self, idx):
        info = {}
        
        # Get reference resolution 
        ref = readRGB(self.rgb_paths[idx])
        original_size = ref.shape[0:2]

        # Get number of objects
        # cat_name = os.path.basename(os.path.dirname(self.rgb_paths[idx]))

        # Read optical flow
        flow_images = []
        for flow_gap in self.flow_gaps:
            flow_path = self.flow_paths[idx].replace("_middle_flow", "_middle_flow_{}".format(flow_gap))
            if os.path.exists(flow_path):
                flow_image = readRGB(flow_path, gt_resolution = original_size)
            else:
                flow_path = self.flow_paths[idx].replace("_middle_flow", "_middle_flow_{}".format(flow_gap))
                flow_image = readRGB(flow_path, gt_resolution = original_size)
            flow_image = self.transform.apply_image(flow_image)
            flow_image_torch = torch.as_tensor(flow_image)
            flow_image_torch = flow_image_torch.permute(2, 0, 1).contiguous()
            flow_image = self.preprocess(flow_image_torch)  # 3 1024 1024
            flow_images.append(flow_image)
        flow_images = torch.stack(flow_images, 0)

        # Read RGB 
        rgb_image = readRGB(self.rgb_paths[idx], gt_resolution = original_size)
        rgb_image = self.transform.apply_image(rgb_image)
        rgb_image_torch = torch.as_tensor(rgb_image)
        rgb_image_torch = rgb_image_torch.permute(2, 0, 1).contiguous()
        rgb_image = self.preprocess(rgb_image_torch)  #3 1024 1024

        # Setup grid points
        grid_points = np.array(self.point_grids) * np.array(original_size)[None, ::-1]        
        grid_coords = self.transform.apply_coords(grid_points, original_size)  # 100, 2
        
        info["rgb_image"] = rgb_image.float()    # RGB image
        info["flow_image"] = flow_images.float() # Flow images with different frame gaps      
        info["grid"] = torch.from_numpy(grid_coords).unsqueeze(1).float()   # Uniform grid points, size 100, 1, 2
        info["size"] = original_size             # Original image size
        info["path"] = os.path.join(*self.rgb_paths[idx].split("/")[-2:]).replace('.jpg', '.png')  # Filename reference
        return info
       
    def __getitem__(self, idx):
        return self.val_loading(idx)
      
 