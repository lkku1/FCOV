import copy
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import mmcv
import torch
import mmengine

from mmcv import LoadImageFromFile, Resize
from mmdet.datasets.transforms import PackDetInputs

from mmdet.registry import MODELS
from mmengine import Config
from mmengine.dataset import Compose, default_collate
import argparse
import sys
import os
import time
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from seg.evaluation.hooks.visual_hook import SAMLocalVisualizer
from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix

IMG_SIZE = 1024
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),
    dict(type=Resize, scale=(IMG_SIZE, IMG_SIZE), keep_ratio=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]
pipeline = Compose(test_pipeline)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='D:/linux/github2/move_scene/scene', type=str)
    parser.add_argument('--save_dir', default="D:/linux/github2/move_scene/scene_condition", type=str)
    parser.add_argument('--name', default="scene2", type=str)
    parser.add_argument('--load_config', default="D:/linux/github2/3dpb/OMG-Seg/demo/configs/m2_convl.py", type=str)
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
    start_time = time.time()
    for img in imgs:
        # if not os.path.exists(os.path.join(option_.save_dir, option_.name + "_sem", img[:-4] + ".npy")):
        if True:
            inputs = pipeline(dict(
                img_path=os.path.join(video_path, img)
            ))

            for key in inputs:
                inputs[key] = inputs[key].to(device=device)

            inputs = default_collate([inputs])
            with torch.no_grad():
                results = model.val_step(inputs)

            classes = copy.deepcopy(model_cfg.get('CLASSES', None))
            assert classes is not None, "You need to provide classes for visualization."
            for idx, cls in enumerate(classes):
                classes[idx] = cls.split(',')[0]

            # Visualization
            visualizer = SAMLocalVisualizer()
            visualizer.dataset_meta = dict(
                classes=classes
            )
            result = results[0]

            visualizer.save_seg(
                data_sample=result,
                out_file=os.path.join(option_.save_dir, option_.name + "_sem", os.path.basename(result.img_path)[:-4] + ".npy"),
            )
    end_time = time.time()
    dura_time = end_time - start_time
    print("Done!")