import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np

"""
Implementation of Sign Language Dataset
"""
class Sign_Isolated(Dataset):
    def __init__(self, data_path, label_path, frames=16, num_classes=226, train=True, transform=None, test_clips=5):
        super(Sign_Isolated, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.test_clips = test_clips
        self.sample_names = []
        self.labels = []
        self.data_folder = []
        label_file = open(label_path, 'r', encoding='utf-8')
        for line in label_file.readlines():
            line = line.strip()
            line = line.split(',')

            self.sample_names.append(line[0])
            self.data_folder.append(os.path.join(data_path, line[0]))
            # print(line[1])
            self.labels.append(int(line[1]))

    def frame_indices_tranform(self, video_length, sample_duration):
        if video_length > sample_duration:
            random_start = random.randint(0, video_length - sample_duration)
            frame_indices = np.arange(random_start, random_start + sample_duration) + 1

        else:
            frame_indices = np.arange(video_length)
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration] + 1
        assert frame_indices.shape[0] == sample_duration
        return frame_indices

    def frame_indices_tranform_test(self, video_length, sample_duration, clip_no=0):
        if video_length > sample_duration:
            start = (video_length - sample_duration) // (self.test_clips - 1) * clip_no
            frame_indices = np.arange(start, start + sample_duration) + 1
        elif video_length == sample_duration:
            frame_indices = np.arange(sample_duration) + 1
        else:
            frame_indices = np.arange(video_length)
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration] + 1

        return frame_indices

    def random_crop_paras(self, input_size, output_size):
        diff = input_size - output_size
        i = random.randint(0, diff)
        j = random.randint(0, diff)
        return i, j, i+output_size, j+output_size

    def read_images(self, folder_path, clip_no=0):
        # assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        images = []
        if self.train:
            index_list = self.frame_indices_tranform(len(os.listdir(folder_path)), self.frames)
            flip_rand = random.random()
            angle = (random.random() - 0.5) * 20
            crop_box = self.random_crop_paras(256, 224)
        else:
            index_list = self.frame_indices_tranform_test(len(os.listdir(folder_path)), self.frames, clip_no)
        
        for i in index_list:
            image = Image.open(os.path.join(folder_path, '{:04d}.jpg').format(i))
            if self.train:
                if flip_rand > 0.5:
                    image = ImageOps.mirror(image)
                image = transforms.functional.rotate(image, angle) 
                image = image.crop(crop_box)
                assert image.size[0] == 224
            else:
                crop_box = (16, 16, 240, 240)
                image = image.crop(crop_box)
                # assert image.size[0] == 224
            if self.transform is not None:
                image = self.transform(image)

            # flow x - y
            image = image[:2,:,:]
            if self.train and flip_rand > 0.5:
                image[0, :, :] = - image[0, :, :]
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension for 3d cnn
        images = images.permute(1, 0, 2, 3)
        # T, C, H, W
        # print(images.shape)
        return images

    def __len__(self):
        return len(self.data_folder)

    def __getitem__(self, idx):
        selected_folder = self.data_folder[idx]
        if self.train:
            images = self.read_images(selected_folder)
        else:
            images = []
            for i in range(self.test_clips):
                images.append(self.read_images(selected_folder, i))
            images = torch.stack(images, dim=0)
            # M, T, C, H, W
        label = torch.LongTensor([self.labels[idx]])
        # print(images.size(), ', ', label.size())
        return {'data': images, 'label': label}
