import os
import sys
import glob
import cv2
import numpy as np
# from sklearn.cluster import KMeans
from scipy import ndimage
# from rembg import remove
from skimage.measure import label
from scipy.signal import convolve2d
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt
import torch
import time

from VideoDepth.utils.util import write_exr_file, read_exr_file

DEPTH_BASE = 'VideoDepth'
DEPTH_OUTPUTS = 'depth'
SEM_BASE = 'VideoSem/mmsegmentation/tools'
SEM_OUTPUTS = 'sem'

anaconda_path = "E:\\anaconda3\\Scripts\\activate.bat"
conda_name = "dibr"

def get_samples_video(video_folder, format):
    lines = sorted([os.path.splitext(os.path.basename(xx)) for xx in glob.glob(os.path.join(video_folder, '*' + format))])
    samples = []

    # all video names
    for seq_dir in lines:
        samples.append({})
        sdict = samples[-1]
        sdict["video_file"] = seq_dir[0]
        sdict["video_format"] = seq_dir[1]

    return samples

def build_folder(folder_path, folder_name):
    if not os.path.exists(os.path.join(folder_path, folder_name)):
        os.makedirs(os.path.join(folder_path, folder_name))

def run_videotoimage(video_path, video_name, video_format, image_path, width, height):

    # capture video
    cap = cv2.VideoCapture(os.path.join(video_path, video_name + video_format))
   
    # video to image
    index = 0
    split_index = 0
    frame_index = 0
    isOpened = cap.isOpened
    build_folder(image_path, video_name)
    while (isOpened):
        ret, frame = cap.read()
        
        if ret == True:
            # if index % 100 == 0:
            #     index = 0
            #     build_folder(image_path, video_name+"_"+str(split_index))
            #     split_index = split_index + 1

            frame = cv2.resize(frame, (width, height))
            # cv2.imwrite(os.path.join(image_path, video_name+"_"+str(split_index-1), str(index).zfill(5) + ".png"), frame)
            cv2.imwrite(os.path.join(image_path, video_name, str(index).zfill(5) + ".png"), frame)
            index = index + 1
        elif ret == False:
            break


def compute_mask(depth, move_mask, move_dilate_mask):
    depth_ori = depth.copy()
    depth = (1 / depth - 1 / depth.max()) / (1 / depth.min() - 1 / depth.max()) * 255
    # depth = cv2.medianBlur(depth, 5)
    H, W = depth.shape
    depth_edge = cv2.Canny(depth.astype(np.uint8), 40, 50) * move_dilate_mask

    opacity_map = cv2.GaussianBlur(cv2.dilate(depth_edge, np.ones((9, 9)), iterations=1), (5, 5), 1)
    if opacity_map.max() > 0:
        opacity_map = opacity_map / opacity_map.max() * 255

    from skimage.morphology import skeletonize, thin
    skeleton = skeletonize(depth_edge > 0)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3,3), np.uint8))
    branch_points = (neighbor_count >= 4) & skeleton # 分支点掩码
    depth_edge[branch_points] = 0

    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(depth_edge.astype(np.uint8), connectivity=8)
    depth_edge_sort = np.zeros((H, W))
    edge_index = 1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 20:
            depth_edge_sort[labels == label] = edge_index
            edge_index += 1

    depth_edge = []
    for index in range(1, edge_index):

        depth_edge_index = np.where(depth_edge_sort== index, 1, 0).astype(np.uint8)
        depth_edge_dindex3 = cv2.dilate(depth_edge_index, np.ones((3, 3)), iterations=1)
    
        depth_nf_edge = (depth_edge_dindex3 - depth_edge_index)
        depth_edge_value = depth_ori * depth_edge_index

        depth_edge_sum = convolve2d(depth_edge_value, np.ones((5, 5)), mode="same") * depth_nf_edge
        depth_edge_index = convolve2d(depth_edge_index, np.ones((5, 5)), mode="same") * depth_nf_edge
        near_side = np.where(depth_ori <= depth_edge_sum / (depth_edge_index + 1e-6), 128, 0 )
        far_side = np.where(np.logical_and(depth_nf_edge > 0, depth_ori > depth_edge_sum/(depth_edge_index + 1e-6)), 255, 0 )

        edge_img = near_side + far_side
        if (((far_side > 0) * move_mask).sum() - ((near_side > 0) * move_mask).sum()) > - ((near_side > 0) * move_mask).sum() / 2:
            depth_edge.append(edge_img)

    return depth_edge, opacity_map


def flood_fill(mask, depth):

    dilates_num = []
    near_dilate_range = np.zeros_like(depth)
    far_dilate_range = np.zeros_like(depth)

    for index in range(len(mask)):

        nf_map = mask[index]

        near = np.where(nf_map == 128, 1, 0)
        far = np.where(nf_map == 255, 1, 0)

        H, W = nf_map.shape
        near_min = depth[near.astype(np.bool_)].mean()
        far_max = depth[far.astype(np.bool_)].mean()
        
        dilate_num = int(np.ceil(np.arctan(2 * (far_max - near_min) / (far_max * near_min)) / min((2 * np.pi / W), np.pi / H))) // 2
        dilate_num = max(dilate_num, 15)
        dilates_num.append(dilate_num)

        far_dilate_range = far_dilate_range + cv2.dilate(far.astype(np.uint8), np.ones((2*dilate_num+1, 2*dilate_num+1), np.uint8), iterations=1)
        near_dilate_range = near_dilate_range + cv2.dilate(near.astype(np.uint8), np.ones((dilate_num, dilate_num), np.uint8), iterations=1)
    
    # dilate_range = np.where(far_dilate_range + near_dilate_range> 0, 1, 0)
    dilate_num = max(dilates_num)
    mask = np.stack(mask).sum(0)

    near = np.where(mask == 128, 1, 0)
    far = np.where(mask == 255, 1, 0)

    for dilate_idx in range(2*dilate_num+1):
        near = cv2.dilate(near.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        near = np.where(near - far > 0, 1, 0)

        far = cv2.dilate(far.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        far = np.where(far - near > 0, 1, 0)

    near = cv2.dilate(near.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    far[near.astype(np.bool_)] = 0
    
    near_far_map = near * 128 * (near_dilate_range > 0) + far * 255 * (far_dilate_range > 0)
    # near_far_map = near_far_map * dilate_range

    return near_far_map

def run_masks(save_dir, video_name, width, height, step=1, N=8):
    start_time = time.time()
    valid_mask = np.zeros((height, width))
    valid_mask[height//N:-height//N, :] = 1

    depth = read_exr_file(os.path.join(save_dir, video_name + "_depth", 'background.exr'))
    depth_edge, opacity_map = compute_mask(depth, valid_mask, valid_mask)
    inpaint_mask_path = os.path.join(save_dir, video_name + "_inpaint_mask")
    if not os.path.exists(inpaint_mask_path):
        os.makedirs(inpaint_mask_path)
    opacity_path = os.path.join(save_dir, video_name + "_opacity")
    if not os.path.exists(opacity_path):
        os.makedirs(opacity_path)


    if len(depth_edge) == 0:
        result = np.zeros((height, width, 1))  
        cv2.imwrite(os.path.join(inpaint_mask_path, 'background.png'), result.astype(np.uint8))
        cv2.imwrite(os.path.join(opacity_path, 'background.png'), opacity_map)
    else:
        result = flood_fill(depth_edge, depth)

        cv2.imwrite(os.path.join(inpaint_mask_path, 'background.png'), result.astype(np.uint8))
        cv2.imwrite(os.path.join(opacity_path, 'background.png'), opacity_map)

    move_data = np.load(os.path.join(save_dir, video_name + "_move", "move.npz"))
    move_masks = move_data["arr_0"]
    move_data.close()

    file_names = sorted(os.listdir(os.path.join(save_dir, video_name + "_depth")))
    
    for index in range(0, move_masks.shape[0], step):
        # if os.path.exists(os.path.join(write_path, str(index) + '.png')):
        #     continue
        move_mask = move_masks[index]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        move_dilate_mask = cv2.dilate(move_mask.astype(np.uint8), kernel, iterations=1)
    
        depth = read_exr_file(os.path.join(save_dir, video_name + "_depth", os.path.splitext(file_names[index])[0] + '.exr'))

        depth_edge, opacity_map = compute_mask(depth, move_mask, move_dilate_mask)
        if len(depth_edge) == 0:
            result = np.zeros((height, width, 1))
            cv2.imwrite(os.path.join(inpaint_mask_path, os.path.splitext(file_names[index])[0] + '.png'), result.astype(np.uint8))
        else:
            result = flood_fill(depth_edge, depth)
            cv2.imwrite(os.path.join(inpaint_mask_path, os.path.splitext(file_names[index])[0] + '.png'), result.astype(np.uint8))
        cv2.imwrite(os.path.join(opacity_path, os.path.splitext(file_names[index])[0] + '.png'), opacity_map)
    end_time = time.time()
    dura_time = end_time - start_time
    a = 0

def run_dep(video_name, image_path, save_path, width, height, gpu):

    path = os.path.join(image_path + "_condition", video_name + "_sem")
    # if os.path.exists(path):
    #     file_list = os.listdir(path)
    #     sky_masks = []
    #     per_masks = []
    #     for index in range(0, len(file_list), 1):
    #         sem = np.load(os.path.join(path, os.path.splitext(file_list[index])[0] + ".npy"))
    #         sky_mask = np.where(sem[..., 1] == 179, True, False)
    #         per_mask = np.where(sem[..., 1] == 1, True, False)
    #         sky_masks.append(sky_mask)
    #         per_masks.append(per_mask)
    #     sky_masks = np.sum(np.sum(np.stack(sky_masks, axis=0), axis=0), axis=-1) / 10000
    #     per_masks = np.sum(np.sum(np.stack(per_masks, axis=0), axis=0), axis=-1) / 10000
    # else:
    #     sky_masks = np.zeros((240))
    #     per_masks = np.zeros((240))
    if sys.platform.startswith('win'):
        os.system(f'cd OMG-Seg/demo && call {anaconda_path} {conda_name} && set CUDA_VISIBLE_DEVICES={gpu} && python image_demo.py --data_dir {image_path}  --save_dir {save_path} --name {video_name}')
    elif sys.platform.startswith("linux"):
        os.system(f'cd OMG-Seg/demo &&  CUDA_VISIBLE_DEVICES={gpu} python image_demo.py --data_dir {image_path}  --save_dir {save_path} --name {video_name}')
    
    if sys.platform.startswith('win'):
        os.system(f'cd {DEPTH_BASE} && call {anaconda_path} {conda_name} &&  set CUDA_VISIBLE_DEVICES={gpu} && python run_unik3d.py --data_dir {image_path}  --save_dir {save_path} --name {video_name} --width {width} --height {height}')
    elif sys.platform.startswith("linux"):
        os.system(f'cd {DEPTH_BASE} &&  CUDA_VISIBLE_DEVICES={gpu} python run_unik3d.py --data_dir {image_path}  --save_dir {save_path} --name {video_name} --width {width} --height {height}')
    
    run_masks(save_path, video_name, width, height)
    # return sky_masks, per_masks

    # video_path = os.path.join("D:/linux/github2/test_image/preprocess/cafeteria1_opacity", "background.png")

    # # image = read_exr_file(video_path)
    # # image = np.clip(255 / image, 0, 255).astype(np.uint8)
    # image = cv2.imread(video_path, 0)
    # # image = cv2.resize(image, (2048, 1024), cv2.INTER_CUBIC)
    # # image = np.repeat(image[:,:, None], 3, axis=-1)
    # # cv2.imwrite("D:/linux/github2/3PVO/src/vids/cafeteria_depth.png", image)
    # # image = cv2.resize(image, (2048, 1024))
    # images = []
    # images.append(255 - image) 
    # images = images * 30
    # # dep = np.stack(images, axis=0)
    # # np.savez_compressed("D:/linux/github2/result/video/cafeteria1_opa", dep)
    # # b = 0
    # size = (2048, 1024)
    
    # # # 创建视频写入对象
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式
    # video = cv2.VideoWriter("D:/linux/github2/3PVO/src/vids/cafeteria_a.mp4", fourcc, 30, size)
    # for idx in range(30):
        
    #     img = images[idx]
    #     img = np.repeat(img[:,:, None], 3, axis=-1)
        
    #     # 调整图片尺寸以匹配第一张图片
    #     if img.shape[0] != height or img.shape[1] != width:
    #         img = cv2.resize(img, size)
        
    #     video.write(img)
    
    # video.release()



def clean_folder(folder, img_exts=['.png', '.jpg', '.npy']):

    for img_ext in img_exts:
        paths_to_check = os.path.join(folder, f'*{img_ext}')
        if len(glob.glob(paths_to_check)) == 0:
            continue
        print(paths_to_check)
        os.system(f'rm {paths_to_check}')

if __name__ == '__main__':
    print(1)


   