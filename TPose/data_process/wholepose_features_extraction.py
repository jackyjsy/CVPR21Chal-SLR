
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import argparse
import os
import pprint
import glob
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from pose_hrnet import get_pose_net
from torch.autograd import Variable
from config import cfg
from config import update_config

import torchvision.transforms as transforms
from tqdm import tqdm
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="../data/color_test", help="Path to input dataset")
    parser.add_argument("--feature_path",type=str, default="../data/test_features", help="Path to output feature dataset")
    parser.add_argument("--istrain",type=bool, default=False, help="generate training data or not")
    opt = parser.parse_args()
    print(opt)
    videopath = opt.video_path+'/*'
    os.makedirs(opt.feature_path, exist_ok=True)
    lenstr = len(videopath)-2
    with torch.no_grad():
        config = './wholebody_w48_384x384_adam_lr1e-3.yaml'
        cfg.merge_from_file(config)
        device = torch.device("cuda")
        model = get_pose_net(cfg, is_train=False)
        checkpoint = torch.load(
    './wholebody_hrnet_w48_384x384.pth', map_location="cuda:0")
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("start extraction!")
        filelist = list(glob.iglob(videopath))
        for filename in tqdm(filelist):
          output_filename = opt.feature_path+'/'+filename[lenstr:-4] + '.pt'
          frames = []
          frames_flip = []
          cap = cv2.VideoCapture(filename)
          length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          fps = cap.get(cv2.CAP_PROP_FPS)
          index = 0
          space = 0
          num_frame = 0
          while cap.isOpened():
          
################## process image #############################################
            success, image = cap.read()
            if success:
                num_frame += 1
            else:
#              print("Ignoring empty camera frame.")
              # If loading a video, use 'break' instead of 'continue'.
              break
            image = cv2.resize(image,(384,384))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_flip = cv2.flip(image,1)
            image = image.astype(np.float32) / 255.
            image_flip = image_flip.astype(np.float32) / 255.
            means=[0.485, 0.456, 0.406]
            stds=[0.229, 0.224, 0.225]
            for i in range(3):
                image[:, :, i] = image[:, :, i] - means[i]
                image[:, :, i] = image[:, :, i] / stds[i]
                image_flip[:, :, i] = image_flip[:, :, i] - means[i]
                image_flip[:, :, i] = image_flip[:, :, i] / stds[i]
            image = image.transpose((2, 0, 1))
            image_flip = image_flip.transpose((2,0,1))
################## clip videos ################################################
            if length < 60:
                num_to_repeat = int(60/length)
                space = 1
                if 60-length*num_to_repeat>0:
                    space =int(length/(60-length*num_to_repeat))
                else:
                    space = 100000
                if index % space == 0 and index < length - (60 %(60-length)) and space < 60:
                    num_to_repeat += 1
                for i in range(num_to_repeat):
                    frames.append(image)
                    frames_flip.append(image_flip)
                index += 1
                if num_frame == length:
                    for i in range(60-len(frames)):
                        frames.append(image)
                        frames_flip.append(image_flip)
                    break
            elif length == 60:
                frames.append(image)
                frames_flip.append(image_flip)
            else:
                space = int(length/(length-60))
                if index % space == 0 and index < length - (length % (length-60)):
                    index += 1
                    continue
                index += 1
                frames.append(image)
                frames_flip.append(image_flip)
################## feature extraction ################################################
          data = np.array(frames)
          input = Variable(torch.from_numpy(data).cuda())
          out = model(input)
          m = torch.nn.MaxPool2d(3, stride=2,padding=1)
          out = m(out)
          out = m(out)
          selected_indices = [0,71,77,85,89,5,6,7,8,9,10,91,93,95,96,99,100,103,104,107,108,111,112,114,116,117,120,121,124,125,128,129,132]
          newout = out[:,selected_indices,:,:]
          newout = newout.view(1,-1,24,24)
          torch.save(newout,output_filename)
          if opt.istrain:
              data = np.array(frames_flip)
              input = Variable(torch.from_numpy(data).cuda())
              out = model(input)
              out = m(out)
              out = m(out)
              newout = out[:,selected_indices,:,:]
              newout = newout.view(1,-1,24,24)
              output_filename = opt.feature_path+'/'+filename[lenstr:-4] + '_flip.pt'
              torch.save(newout,output_filename)
          if len(frames)!=60:
              break
          cap.release()

if __name__ == '__main__':
    main()
