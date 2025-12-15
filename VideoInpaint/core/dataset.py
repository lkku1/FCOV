import os
import cv2
import io
import glob
import scipy
import json
import zipfile
import random
import collections
import torch
import math
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from skimage.color import rgb2gray, gray2rgb
from core.utils import create_random_shape_with_random_motion
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip, GroupRandomHorizontalROll
from core.file_client import FileClient, imfrombytes
import OpenEXR
import Imath
def read_exr_file(file_path, channel_names=None):
    """
    读取OpenEXR文件并返回各通道的numpy数组
    :return: dict {channel_name: np.array}
    """
    # 打开文件
    exr_file = OpenEXR.InputFile(file_path)
    
    # 获取图像属性
    header = exr_file.header()
    channels = header['channels'].keys()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    z_data = exr_file.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
    z_array = np.frombuffer(z_data, dtype=np.float32).reshape(height, width)
    
    exr_file.close()
    
    return z_array

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train', debug=False):
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.video_root = args['data_root']
        self.condition_root = args['condition_root']
        self.size = self.w, self.h = (args['w'], args['h'])

        self.video_names = sorted(os.listdir(self.video_root))
        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)

            self.video_dict[v] = v_len
            self.frame_dict[v] = frame_list

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])
        
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]

        all_masks = create_random_shape_with_random_motion(self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)
        ref_index = get_ref_index(self.video_dict[video_name], self.sample_length)
        # read video frames
        frames = []
        masks = []
        depths = []
        for idx in ref_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            # dep_path = os.path.join(self.condition_root, video_name + "_depth", frame_list[idx].replace("jpg", "png"))
            # img_bytes = self.file_client.get(dep_path, 'img')
            # dep = imfrombytes(img_bytes, flag="unchanged", float32=False)
            dep_path = os.path.join(self.condition_root, video_name + "_depth", frame_list[idx].replace("jpg", "exr"))
            dep = read_exr_file(dep_path)
            dep = cv2.resize(dep, self.size, interpolation=cv2.INTER_LINEAR)
            dep = Image.fromarray(dep.astype(np.float64))

            frames.append(img)
            depths.append(dep)
            masks.append(all_masks[idx])

        if self.split == 'train':
            frames, depths = GroupRandomHorizontalFlip()(frames, depths)

        # To tensors
        frame_tensors = self._to_tensors(frames).div(255)*2.0 - 1.0
        depth_tensors = self._to_tensors(depths)
        depth_tensors = (depth_tensors - depth_tensors.min()) / (depth_tensors.max() - depth_tensors.min())

        frame_tensors, depth_tensors = GroupRandomHorizontalROll()(frame_tensors, depth_tensors)
        mask_tensors = self._to_tensors(masks).div(255)
        return frame_tensors, depth_tensors, mask_tensors
    



def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index
