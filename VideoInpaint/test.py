# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import math
import importlib
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models
from torchvision import transforms

# My libs
from core.utils import Stack, ToTorchFormatTensor
from model.i3d import InceptionI3d
from scipy import linalg
from model.MiDaS.midas.dpt_depth import DPTDepthModel
from model.MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from model.DGDVI import DepthCompletion, ContentReconstruction, ContentEnhancement
import OpenEXR
import Imath
import time

def write_exr_file(file_path, file):

    header = OpenEXR.Header(file.shape[1], file.shape[0])
    header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    exr_file = OpenEXR.OutputFile(file_path, header)
    exr_file.writePixels({"Z": file.tobytes()})
    exr_file.close()


parser = argparse.ArgumentParser(description="DGDVI")
parser.add_argument("--video_path", type=str, default='D:/linux/github2/test_image/image/cafeteria1')
parser.add_argument("--depth_path", type=str, default='D:/linux/github2/test_image/preprocess/cafeteria1_depth')
parser.add_argument("--mask_path", type=str, default='D:/linux/github2/test_image/preprocess/cafeteria1_inpaint_mask')
parser.add_argument("--stage1_model_path", type=str, default="D:/linux/github2/DGDVI/checkpoints/stage1.pth")
parser.add_argument("--stage2_model_path", type=str, default="D:/linux/github2/3dpb/VideoInpaint/checkpoints/stage2.pth")
parser.add_argument("--stage3_model_path", type=str, default="D:/linux/github2/3dpb/VideoInpaint/checkpoints/stage3.pth")
parser.add_argument("--stage1_lora_model_path", type=str, default="D:/linux/github2/3dpb/release_model/DGDVI_davis/gen_dc_00007.pth")
parser.add_argument("--stage2_lora_model_path", type=str, default="D:/linux/github2/3dpb/release_model/DGDVI_davis/gen_cr_00007.pth")
parser.add_argument("--stage3_lora_model_path", type=str, default="D:/linux/github2/3dpb/release_model/DGDVI_davis/gen_ce_00007.pth")
parser.add_argument("--midas_path", type=str, default='VideoInpaint/checkpoints/dpt_large-midas-2f21e586.pt')
parser.add_argument("--width", type=int, default=432)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--outw", type=int, default=432)
parser.add_argument("--outh", type=int, default=240)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=6)
parser.add_argument("--savefps", type=int, default=24)
parser.add_argument("--use_mp4", action='store_true')
parser.add_argument("--dump_results", action='store_true')
args = parser.parse_args()


w, h = args.width, args.height
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

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

# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > num_ref:
                #if len(ref_index) >= 5-len(neighbor_ids):
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks 
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.where(m==255, 1.0, m)
        m = np.where(m==128, 2.0, m)
        masks.append(Image.fromarray(m))

    # image = cv2.imread(mpath + "/" + "background.png", -1)
    # image = Image.fromarray(image)
    # masks.append(image.resize((w,h), resample=0))

    return masks

def read_depth(dpath):
    depths = []
    dnames = os.listdir(dpath)
    dnames.sort()
    for d in dnames: 
        d =read_exr_file(os.path.join(dpath, d))
        depths.append(Image.fromarray(cv2.resize(d, (432, 240), interpolation=cv2.INTER_LINEAR).astype(np.float64)))
    # d = read_exr_file(os.path.join(dpath, "background.exr"))
    # d = cv2.resize(d, (432, 240), interpolation=cv2.INTER_LINEAR)
    # d = Image.fromarray(d.astype(np.float64))
    # depths.append(d.resize((w,h), resample=2))
    # depths.append(d)
    return depths


#  read frames from video 
def read_frame_from_videos(args):
    vname = args.video_path
    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w,h)))
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname+'/'+name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w,h)))
        
    return frames       

def main_worker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set up stage-1 (depth compeltion) pretrained model
    stage1_model = DepthCompletion().to(device)
    data = torch.load(args.stage1_model_path, map_location=device, weights_only=True)
    stage1_model.load_state_dict(data['netG'], strict=False)
    data = torch.load(args.stage1_lora_model_path, map_location=device, weights_only=True)
    stage1_model.load_state_dict(data['netG'], strict=False)
    print('loading from: {}'.format(args.stage1_model_path))
    stage1_model.eval()
    # set up stage-2 (content reconstruction) pretrained model
    stage2_model = ContentReconstruction().to(device)
    data = torch.load(args.stage2_model_path, map_location=device, weights_only=True)
    stage2_model.load_state_dict(data['netG'], strict=False)
    data = torch.load(args.stage2_lora_model_path, map_location=device, weights_only=True)
    stage2_model.load_state_dict(data['netG'], strict=False)
    print('loading from: {}'.format(args.stage2_model_path))
    stage2_model.eval()
    # set up stage-3 (content enhancement) pretrained model
    stage3_model = ContentEnhancement().to(device)
    data = torch.load(args.stage3_model_path, map_location=device, weights_only=True)
    stage3_model.load_state_dict(data['netG'], strict=False)
    data = torch.load(args.stage3_lora_model_path, map_location=device, weights_only=True)
    stage3_model.load_state_dict(data['netG'], strict=False)
    print('loading from: {}'.format(args.stage3_model_path))
    stage3_model.eval()

    frames_PIL = read_frame_from_videos(args)
    video_length = len(frames_PIL)
    imgs = _to_tensors(frames_PIL).unsqueeze(0).div(255)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames_PIL]

    depths_PIL = read_depth(args.depth_path)
    # deps = _to_tensors(depths_PIL).unsqueeze(0)
    deps = torch.from_numpy(np.stack(depths_PIL, axis=0))[None, :, None]
    deps_min = deps.min()
    deps_max = deps.max()
    deps = (1/deps - 1/deps_max) / (1/deps_min - 1/deps_max)
    depths = [np.array(f).astype(np.float32) for f in depths_PIL]

    masks = read_mask(args.mask_path)    
    # masks = masks * len(frames)
    binary_masks = [np.expand_dims((np.array(m) == 128).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)

    imgs, deps, masks = imgs.to(device), deps.to(device), masks.to(device)
    comp_frames = [None]*video_length
    comp_depths = [None]*video_length

    if not os.path.exists('results'):
        os.mkdir('results')
    name = args.video_path.split("/")[-1]
    if not os.path.exists(os.path.join('results', f'{name}')):
        os.mkdir(os.path.join('results', f'{name}'))
    name = args.video_path.split("/")[-1]

    lst = os.listdir(args.video_path)
    lst.sort()
    neighbor_stride = 3
    start_time = time.time()
    for f in range(0, video_length, neighbor_stride):
        # neighbor_ids = [f for i in range(0, 5)]
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        # ref_ids = get_ref_index(f, neighbor_ids, video_length)
        # len_temp = len(neighbor_ids) + len(ref_ids)

        selected_imgs = imgs[:1, neighbor_ids, :, :, :]
        selected_deps = deps[:1, neighbor_ids, :, :, : ]
        selected_masks = masks[:1, neighbor_ids, :, :, :]
        syn_mask = torch.where(selected_masks==128, 1.0, 0.0)
        con_mask = torch.where(selected_masks==255, 1.0, 0.0)
        with torch.no_grad():

            b, t, c, h, w = selected_imgs.size()
            # depths = read_exr_file("D:/linux/github2/test_image/preprocess/pier_depth/background.exr")
            # depths = cv2.resize(depths, (432, 240), interpolation=cv2.INTER_LINEAR)
            # depths = (1/depths - 1/depths.max())/(1/depths.min() - 1/depths.max())
            # depths = torch.repeat_interleave(torch.from_numpy(depths)[None, None, None], t, dim=1).cuda()
            
            pred_depths = stage1_model(selected_deps*(con_mask).float(), 1.- con_mask)
            input_imgs = selected_imgs*con_mask.float()
            pred_img, feat = stage2_model(input_imgs, pred_depths.float())
            bt, c_, h_, w_ = feat.shape
            feat = feat.view(b, t, c_, h_, w_)
            feat = feat[:, :len(neighbor_ids), ...]
            pred_img = pred_img.view(b, t, c, h, w)
            pred_img = pred_img[:, :len(neighbor_ids), ...]
            selected_imgs = selected_imgs[:, :len(neighbor_ids), ...].contiguous()
            selected_masks = selected_masks[:, :len(neighbor_ids), ...].contiguous()
            pred_img = stage3_model(selected_imgs*con_mask + pred_img*(1 - con_mask), feat)

            pred_img = (pred_img + 1.) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            # selected_imgs = selected_imgs.view(b*t, 1, h, w).cpu().permute(0, 2, 3, 1).numpy()*255
            # selected_imgs = selected_imgs.clip(0,255).astype(np.uint8)
            pred_depths = pred_depths.view(b, t, 1, h, w)
            # pred_depths = np.clip(pred_depths, 0, 1)
            # pred_depths = pred_depths.clip(0,255).astype(np.uint8)
            # a = pred_depths[0] * (deps_max.numpy() - deps_min.numpy()) + deps_min.numpy()
            # cv2.imwrite("1.png", a.astype(np.uint16))
            # for i in range(len(neighbor_ids)):
            #     idx = neighbor_ids[i]
            #     img = np.array(pred_img[i]).astype(
            #         np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
            #     if comp_frames[idx] is None:
            #         comp_frames[idx] = img
            #     else:
            #         comp_frames[idx] = comp_frames[idx].astype(
            #             np.float32)*0.5 + img.astype(np.float32)*0.5
            #     cv2.imwrite("2.png", cv2.cvtColor(np.array(comp_frames[0]).astype(np.uint8), cv2.COLOR_BGR2RGB))
            #     depth = np.array(pred_depths[i]).astype(np.uint8)*binary_masks[idx] + depths[i] * (1-binary_masks[idx])
            #     if comp_depths[idx] is None:
            #         comp_depths[idx] = depth
            #     else:
            #         comp_depths[idx] = comp_depths[idx].astype(np.float32)*0.5 + depth.astype(np.float32)*0.5

            #     break
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)*syn_mask[0, 0, 0, ...][..., None].cpu().numpy() + frames[idx] * (1 - syn_mask[0, 0, 0, ...][..., None].cpu().numpy())
                
                depth = (pred_depths * (syn_mask + con_mask) + selected_deps * (1 - syn_mask - con_mask))[0,0,0].cpu().numpy()
                depth = 1 / (depth * (1/deps_min.numpy() - 1/deps_max.numpy()) + 1/deps_max.numpy())
                # dep = 1 / (np.array(pred_depths[i]) * (1/deps_min.numpy() - 1/deps_max.numpy()) + 1/deps_max.numpy())
                # dep = dep*syn_mask[0, 0, 0, ...][..., None].cpu().numpy() 
                # dep = dep[:, :, 0] + depths[idx] * (1 - syn_mask[0, 0, 0, ...].cpu().numpy())
                
                # cv2.imwrite("3.png", frames[idx] * (con_mask[0, 0, 0, ...][..., None].cpu().numpy()))
                # cv2.imwrite("D:/linux/github2/result/scene_condition/pier_our/"+ name[f], cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2RGB))
                # cv2.imwrite("D:/linux/github2/result/scene_condition/cafeteria1_our/background.png", cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2RGB))
                # write_exr_file(os.path.join("D:/linux/github2/result/scene_condition/cafeteria1_our/background.exr"), depth)

    end_time = time.time()
    dura_time = end_time - start_time
    a = 0              
    # writer = cv2.VideoWriter(f"results/{name}/{name}_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    # for f in range(video_length):
    #     comp = np.array(comp_frames[f]).astype(
    #         np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
    #     if w != args.outw:
    #         comp = cv2.resize(comp, (args.outw, args.outh), interpolation=cv2.INTER_LINEAR)
    #     writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # writer.release()
    # print('Finish in {}'.format(f"{name}_result.mp4"))
    # writer = cv2.VideoWriter(f"results/{name}/{name}_result_mask.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    # for f in range(video_length):
    #     comp = np.array(comp_frames[f]).astype(
    #         np.double)*binary_masks[f].astype(
    #         np.double) + frames[f].astype(
    #         np.double) * (1.-binary_masks[f].astype(
    #         np.double)) + binary_masks[f].astype(
    #         np.double) * 25
    #     comp = np.clip(comp,0,255)
    #     writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # writer.release()
    # print('Finish in {}'.format(f"{name}_result_mask.mp4"))

if __name__ == '__main__':
    main_worker()
