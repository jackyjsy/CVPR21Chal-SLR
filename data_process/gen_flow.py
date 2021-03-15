import cv2
import numpy as np
import os
from natsort import natsorted
def crop(image, center, radius, size=512):
    scale = 1.3
    radius_crop = (radius * scale).astype(np.int32)
    center_crop = (center).astype(np.int32)

    rect = (max(0,(center_crop-radius_crop)[0]), max(0,(center_crop-radius_crop)[1]), 
                 min(512,(center_crop+radius_crop)[0]), min(512,(center_crop+radius_crop)[1]))

    image = image[rect[1]:rect[3],rect[0]:rect[2],:]

    if image.shape[0] < image.shape[1]:
        top = abs(image.shape[0] - image.shape[1]) // 2
        bottom = abs(image.shape[0] - image.shape[1]) - top
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT,value=(0,0,0))
    elif image.shape[0] > image.shape[1]:
        left = abs(image.shape[0] - image.shape[1]) // 2
        right = abs(image.shape[0] - image.shape[1]) - left
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT,value=(0,0,0))
    return image

selected_joints = np.concatenate(([0,1,2,3,4,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0) 

folder = 'output_file_val' # 'output_file_train', 'output_file_test'
npy_folder = 'val_npy/npy3' # 'train_npy/npy3', 'test_npy/npy3'
out_folder = 'val_flow_depth' # 'train_flow_depth', 'test_flow_depth', 'train_flow_color', 'val_flow_color', 'test_flow_color' 


for o in natsorted(os.listdir(folder)):
    if os.path.isdir(os.path.join(folder,o)) and 'depth' in o:
        video_name = o[:-6]
        files = []
        current_read_folder = os.path.join(folder,o)
        for name in natsorted(os.listdir(os.path.join(folder,o))):
            if 'flow_x' in name:
                files.append(name)
        print(video_name)
        npy = np.load(os.path.join(npy_folder, video_name + '_color.mp4.npy')).astype(np.float32)
        npy = npy[:, selected_joints, :2]
        npy[:, :, 0] = 512 - npy[:, :, 0]
        xy_max = npy.max(axis=1, keepdims=False).max(axis=0, keepdims=False)
        xy_min = npy.min(axis=1, keepdims=False).min(axis=0, keepdims=False)
        assert xy_max.shape == (2,)
        xy_center = (xy_max + xy_min) / 2 - 20
        xy_radius = (xy_max - xy_center).max(axis=0)
        current_save_folder = os.path.join(out_folder, video_name)
        if not os.path.exists(current_save_folder):
            os.mkdir(current_save_folder)
        
        for f in files:
            img_x = cv2.imread(os.path.join(current_read_folder, f), cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(os.path.join(current_read_folder, f.replace('x', 'y')), cv2.IMREAD_GRAYSCALE)
            print(os.path.join(current_read_folder, '{:04d}.jpg'.format(int(f[7:-4]))))
            img = np.zeros((img_y.shape[0],img_y.shape[1],3), dtype=np.uint8)
            img[:,:,2] = img_x
            img[:,:,1] = img_y
            img[:,:,0] = img_x

            img = cv2.resize(img, (512, 512))
            img = crop(img, xy_center, xy_radius)
            img = cv2.resize(img, (256, 256))
            cv2.imwrite(os.path.join(current_save_folder, '{:04d}.jpg'.format(int(f[7:-4]))), img)

