# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import numpy as np
import cv2
import OpenEXR as exr
import Imath
import array

def compute_scale_and_shift(prediction, target, mask, scale_only=False):
    if scale_only:
        return compute_scale(prediction, target, mask), 0
    else:
        return compute_scale_and_shift_full(prediction, target, mask)


def compute_scale(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target)

    x_0 = b_0 / (a_00 + 1e-6)

    return x_0

def compute_scale_and_shift_full(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01

    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1


def get_interpolate_frames(frame_list_pre, frame_list_post):
    assert len(frame_list_pre) == len(frame_list_post)
    min_w = 0.0
    max_w = 1.0
    step = (max_w - min_w) / (len(frame_list_pre)-1)
    post_w_list = [min_w] + [i * step for i in range(1,len(frame_list_pre)-1)] + [max_w]
    interpolated_frames = []
    for i in range(len(frame_list_pre)):
        interpolated_frames.append(frame_list_pre[i] * (1-post_w_list[i]) + frame_list_post[i] * post_w_list[i])
    return interpolated_frames

def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(0.15*size[0]):size[0] - int(0.15*size[0]), int(0.15*size[1]): size[1] - int(0.15*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask

def read_exr_file(filename):
    """
    从OpenEXR文件中读取深度通道（Z）。
    
    参数:
    - filename (str): EXR 文件路径。
    
    返回:
    - numpy.ndarray: 深度图（float32），如果找不到Z通道则返回None。
    """
    try:
        # 1. 打开 EXR 文件
        exrfile = exr.InputFile(filename)
        header = exrfile.header()
        
        # 2. 获取数据窗口和图像尺寸
        dw = header['dataWindow']
        isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        
        # 3. 确定深度通道名称
        # 常见深度通道名是 'Z' 或 'Z.V'
        depth_channel_name = None
        for name in header['channels']:
            if name == 'Z' or name.startswith('Z.'):
                depth_channel_name = name
                break
        
        if depth_channel_name is None:
            print(f"警告: 文件 {filename} 中未找到 Z 通道。")
            return None

        # 4. 读取通道数据
        # 使用 FLOAT 类型（32位浮点数）读取
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        depth_data = exrfile.channel(depth_channel_name, pt)
        
        # 5. 将字节数据转换为 numpy 数组
        # OpenEXR 返回的是原始字节数据
        depth_array = np.frombuffer(depth_data, dtype=np.float32)
        
        # 6. 重塑为图像尺寸
        # 注意: OpenEXR 的通道数据通常是扁平的 (isize[0] * isize[1])
        depth_map = depth_array.reshape(isize)
        
        return depth_map

    except Exception as e:
        print(f"读取 EXR 文件时发生错误: {e}")
        return None
    
def write_exr_file(filename, depth_map):
    """
    将 numpy 数组写入为包含深度通道（Z）的 OpenEXR 文件。
    
    参数:
    - filename (str): 输出 EXR 文件路径。
    - depth_map (numpy.ndarray): 深度图数据（应为 float32）。
    """
    try:
        if depth_map.dtype != np.float32:
            depth_map = depth_map.astype(np.float32)

        height, width = depth_map.shape
        
        # 1. 定义数据窗口和显示窗口
        # 使用 Imath.Box2i 来定义图像边界
        dw = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1))
        
        # 2. 定义通道类型
        # 深度通道通常使用 FLOAT (32-bit float)
        channel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # 3. 创建头部 (Header)
        header = exr.Header(width, height)
        # 设定通道：只包含 'Z' 深度通道
        header['channels'] = {'Z': Imath.Channel(channel_type)}
        
        # 4. 展平数据为字节字符串
        # OpenEXR 要求通道数据为扁平的字节字符串
        depth_data_flat = depth_map.tobytes()
        
        # 5. 写入文件
        # 'Z' 必须对应 depth_data_flat
        outfile = exr.OutputFile(filename, header)
        outfile.writePixels({'Z': depth_data_flat})
        print(f"成功写入 EXR 深度图到 {filename}")
        
    except Exception as e:
        print(f"写入 EXR 文件时发生错误: {e}")