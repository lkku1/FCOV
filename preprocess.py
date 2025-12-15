import numpy as np
import argparse
import glob
import os
from tqdm import tqdm

from util_other import get_samples_video, run_videotoimage, run_dep

os.environ['MKL_THREADING_LAYER'] = 'GNU'

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='D:/linux/github2/valid')
parser.add_argument('--format', type=str, default='mp4')
parser.add_argument('--width', type=int, default=432)
parser.add_argument('--height', type=int, default=240)
parser.add_argument('--image_path', type=str, default='D:/linux/github2/vaild_image/480p')
parser.add_argument('--save_path', type=str, default='D:/linux/github2/vaild_image/480p_condition')
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()


# video name (include name)
video_list = get_samples_video(args.path, args.format)


# convert all videos to images
for idx in tqdm(range(len(video_list))):
    depth = None
    video = video_list[idx]
    print("Start")
    print("video to images")
    print("video folder: " + video['video_file'])
    # video split to images and build corresponding folder(color, depth, flow etc)
    run_videotoimage(args.path, video['video_file'], video['video_format'], args.image_path, args.width, args.height)

# Calculate the depth, optical flow and semantics of images in each folder
image_list_set = sorted([file for file in glob.glob(os.path.join(args.image_path, '*'))])

for idx in range(len(image_list_set)):
    
    # compute depth and flow in per frame, background RGB and depth,
    base_name = os.path.basename(image_list_set[idx])
    run_dep(base_name, args.image_path, args.save_path, args.width, args.height, 0)
    


