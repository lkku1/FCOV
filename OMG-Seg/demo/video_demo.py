import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import copy
import os.path

import mmcv
import mmengine
import torch
from mmcv import LoadImageFromFile, Resize, TransformBroadcaster
import argparse

from mmdet.registry import MODELS
from mmengine import Config
from mmengine.dataset import Compose, default_collate

import sys

import numpy as np
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)



from seg.datasets.pipelines.formatting import PackVidSegInputs
from seg.evaluation.hooks.visual_hook import VidSegLocalVisualizer
from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix

VID_SIZE = (1280, 736)
test_pipeline = [
    dict(
        type=TransformBroadcaster,
        transforms=[
            dict(type=LoadImageFromFile, backend_args=None),
            dict(type=Resize, scale=VID_SIZE, keep_ratio=True),
        ]),
    dict(
        type=PackVidSegInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'frame_id', 'video_length', 'ori_video_length'),
        default_meta_keys=()
    )
]


pipeline = Compose(test_pipeline)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='D:/linux/github2/360_train/480p', type=str)
    parser.add_argument('--save_dir', default="D:/linux/github2/360_train/480p_condition", type=str)
    parser.add_argument('--name', default="019cc67f-512f-4b8a-96ef-81f806c86ce1_0", type=str)
    parser.add_argument('--load_config', default="D:/linux/github2/3dpb/OMG-Seg/demo/configs/m2_convl_vid.py", type=str)
    parser.add_argument('--load_weight', default="D:/linux/github2/3dpb/OMG-Seg/checkpoint/omg_seg_convl.pth", type=str)
    option_, _ = parser.parse_known_args()

    if not os.path.exists(os.path.join(option_.save_dir, option_.name + "_sem")):
        os.mkdir(os.path.join(option_.save_dir, option_.name + "_sem"))

    model_cfg = Config.fromfile(option_.load_config)

    model = MODELS.build(model_cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    model = model.eval()
    model_states = load_checkpoint_with_prefix(option_.load_weight)
    incompatible_keys = model.load_state_dict(model_states, strict=False)

    video_path = os.path.join(option_.data_dir, option_.name)

    imgs = sorted(list(mmengine.list_dir_or_file(video_path)))

    batch_size = 50
    batch_num = len(imgs) // batch_size + 1
    for batch_idx in range(0, batch_num, 1):
        img_list = []
        img_id_list = []

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(imgs))

        if os.path.exists(os.path.join(option_.save_dir, option_.name + "_sem", imgs[end_idx-1][:-4] + ".npy")):
            continue

        for idx, img in enumerate(imgs[start_idx:end_idx]):
            img_list.append(os.path.join(video_path, img))
            img_id_list.append(idx)
        inputs = pipeline(dict(
            img_path=img_list,
            img_id=img_id_list,
        ))
        for key in inputs:
            inputs[key] = inputs[key].to(device=device)

        inputs = default_collate([inputs])
        with torch.no_grad():
            results = model.val_step(inputs)

        print("Starting to visualize results...")

        classes = copy.deepcopy(model_cfg.get('CLASSES', None))
        assert classes is not None, "You need to provide classes for visualization."
        for idx, cls in enumerate(classes):
            classes[idx] = cls.split(',')[0]

        # Visualization
        visualizer = VidSegLocalVisualizer()
        visualizer.dataset_meta = dict(
            classes=classes
        )
        result = results[0]

        for data_sample in result:
            visualizer.save_seg(
                data_sample=data_sample,
                out_file=os.path.join(option_.save_dir, option_.name + "_sem", os.path.basename(data_sample.img_path)[:-4] + ".npy"),
            )

    print("Done!")