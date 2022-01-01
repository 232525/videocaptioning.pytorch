import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import cv2
import math
from PIL import Image

C, H, W = 3, 224, 224

class TmpModel(nn.Module):
    def __init__(self, model_type='resnet152'):
        super(TmpModel, self).__init__()
        self.model_type = model_type
        if self.model_type == 'inception_v3':
            self.model = torchvision.models.inception_v3(True)
            self.model.fc = nn.Identity()
            self.input_size = (299, 299)
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif self.model_type == 'resnet152':
            self.model = torchvision.models.resnet152(True)
            self.model.fc = nn.Identity()
            self.input_size = (224, 224)
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]    
        else:
            self.model = nn.Identity()
            print('Not Support')
            raise
            
    def forward(self, x):
        return self.model(x)


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))
    # video_list = glob.glob(os.path.join(params['video_path'], '*.mp4'))
    video_list = glob.glob(os.path.join(params['video_path'], '*.avi'))
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        # dst = params['model'] + '_' + video_id
        dst = dir_fc + '_' + video_id
        extract_frames(video, dst)
        
        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        # 均匀采样出40帧
        samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))
        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list), C, H, W))
        # 读取40帧图像数据，[40, 3, 224, 224]
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img
        with torch.no_grad():
            fc_feats = model(images.cuda()).squeeze()
        
        # 保存视频帧特征，一个视频被表示为大小为[40, 2048]的特征
        np.savez_compressed(
            os.path.join(dir_fc, video_id), 
            feat=fc_feats.cpu().numpy()
        )
        # cleanup
        shutil.rmtree(dst)
        
class LoadTransformImage(object):
    def __init__(self, model, scale=0.875, random_crop=False,
                 random_hflip=False, random_vflip=False,
                 preserve_aspect_ratio=True):
        self._scale = scale
        self._input_size = model.input_size
        self._mean = model.mean
        self._std = model.std
        
        # 图像预处理
        tfs = []
        # Resize图像大小
        if preserve_aspect_ratio:
            # 将图像短边缩放到 224 / _scale，并保持纵横比
            tfs.append(transforms.Resize(
                int(math.floor(max(self._input_size) / self._scale))))
        else:
            # 将图像缩放到 (224 / _scale, 224 / _scale)，不考虑纵横比
            # (256, 256)
            height = int(self._input_size[0] / self._scale)
            width = int(self._input_size[1] / self._scale)
            tfs.append(transforms.Resize((height, width)))

        if random_crop:
            # 随机裁减
            tfs.append(transforms.RandomCrop(max(self._input_size)))
        else:
            # 中心裁减
            tfs.append(transforms.CenterCrop(max(self._input_size)))

        tfs.append(transforms.ToTensor())
        tfs.append(transforms.Normalize(mean=self._mean, std=self._std))

        self._transform = transforms.Compose(tfs)
        
    def __call__(self, img_path):
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # opencv 转 PIL.Image，且图像需为RGB格式
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # ndarray转为tensor
        img_tensor = self._transform(img)
        return img_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/resnet152', 
                        help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, 
                        default=40,
                        help='how many frames to sampler per video')
    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/train-video', 
                        help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, 
                        default='resnet152',
                        help='the CNN model you want to use to extract_feats')
    
    args = parser.parse_args()
    params = vars(args)
    
    # 创建临时模型，用于提取特征，基于torchvision
    model = TmpModel(params['model'])
    # model = nn.DataParallel(model)
    model = model.cuda()
    print(model)
    load_image_fn = LoadTransformImage(model)
    # 提取视频帧特征
    extract_feats(params, model, load_image_fn)

            