import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import scipy.io
import os
from torch.autograd import Variable
import random
 
class TorchDataset(Dataset):
    def __init__(self,
    istrain,
    fea_dir,
    isaug=False,# set True for training, False for finetuning
    repeat=1):
        self.load_name = './train_val_split.mat'
        self.mat = scipy.io.loadmat(self.load_name)
        self.istrain = istrain
        self.isaug = isaug
        # file name of training and testing samples
        self.train_file_name = self.mat['train_file_name']
        self.test_file_name = self.mat['test_file_name']
        self.train_label = self.mat['train_label']
        self.test_label = self.mat['test_label']
        self.train_number = self.mat['train_count']
        self.test_number = self.mat['test_count']
        self.train_number = self.train_number[0][0]
        self.test_number = self.test_number[0][0]
        self.fea_label_list = self.read_file()
        if self.istrain:
           random.shuffle(self.fea_label_list)
        self.fea_dir = fea_dir
        self.len = len(self.fea_label_list)
        self.repeat = repeat
        
    def __getitem__(self, i):
        index = i % self.len
        fea_name, label = self.fea_label_list[index]
        fea_path = os.path.join(self.fea_dir, fea_name)
        features = self.load_data(fea_path)
        label=np.array(label)
        return features, label

        
    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.fea_label_list) * self.repeat
        return data_len

    def read_file(self):
        fea_label_list = []
        if self.istrain:
            for idx in range(self.train_number):
              name = self.train_file_name[idx][0][0][0][0]+ '_color'+ '.pt'
              labels = self.train_label[idx][0]
              fea_label_list.append((name, labels))
              name = self.train_file_name[idx][0][0][0][0]+ '_color'+ '_flip.pt'
              labels = self.train_label[idx][0]
              fea_label_list.append((name, labels))
        else:
            for idx in range(self.test_number):
              name = self.test_file_name[idx][0][0][0][0]+ '_color'+ '.pt'
              labels = int(self.test_label[idx][0])
              fea_label_list.append((name, labels))
        return fea_label_list
 
    def load_data(self, path):
        data = torch.load(path, map_location='cpu')
        if self.isaug:
            data = data.view(60,-1,24,24)
            judge = random.randint(0,12)
############## aug on frames ##########################################
            slist = range(0,60)
            if judge>7.5 and judge <11.5:
                rlength = 60 - random.randint(1,29)
                rindex = random.sample(range(0,60),rlength)
                extlist = random.sample(rindex,60-rlength)
                final_list = sorted([*rindex, *extlist])
                slist = np.array(final_list)

            if judge >2.5 and judge <3.5:
                rlength = 60 - random.randint(31,45)
                repeatnum =int (60/rlength)
                extension = 60 - rlength*repeatnum
                rindex = random.sample(range(0,60),rlength)
                extlist = random.sample(rindex,extension)
                rindex = list(np.repeat(np.array(rindex),repeatnum))
                final_list = sorted([*rindex, *extlist])
                slist = np.array(final_list)
            slist = list(slist)
##########################################################################
            if self.istrain:
                data = data[slist,:,:,:]
                data = data.view(-1,24,24)
            else:
                data = data.view(-1,24,24)
        else:
            data = data.view(-1,24,24)
        return data
