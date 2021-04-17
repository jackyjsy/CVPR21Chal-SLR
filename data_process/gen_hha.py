import cv2
import numpy as np
import os

# from utils.rgbd_util import *
# from utils.getCameraParam import *
# from getHHA import getHHA
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

folder = 'train' # 'val', 'test'
folder_hha = 'train_hha_2' # 'val_hha_2', 'test_hha_2'
npy_folder = 'train_npy/npy3' # 'val_npy/npy3', 'test_npy/npy3'
out_folder = 'train_hha_2_mask' # 'val_hha_2_mask', 'test_hha_2_mask'



for root, dirs, files in os.walk(folder, topdown=False):
    # step = int(len(files) / 8)
    # files = files[step:step*2]
    # files = files[::-1]
    for name in files:
        if 'depth' in name:
            print(os.path.join(root, name))
            print(os.path.join(npy_folder, name + '.npy'))

            # if os.path.exists(os.path.join(out_folder, name[:-10])):
            #     print('Skipping ', os.path.join(out_folder, name[:-10]))
            #     continue

            cap = cv2.VideoCapture(os.path.join(root, name))
            color_name = name.replace('depth', 'color')
            npy = np.load(os.path.join(npy_folder, color_name + '.npy')).astype(np.float32)
            npy = npy[:, selected_joints, :2]
            npy[:, :, 0] = 512 - npy[:, :, 0]
            xy_max = npy.max(axis=1, keepdims=False).max(axis=0, keepdims=False)
            xy_min = npy.min(axis=1, keepdims=False).min(axis=0, keepdims=False)
            assert xy_max.shape == (2,)
            xy_center = (xy_max + xy_min) / 2 - 20
            xy_radius = (xy_max - xy_center).max(axis=0)
            index = 0

            if not os.path.exists(os.path.join(out_folder, name[:-10])):
                os.makedirs(os.path.join(out_folder, name[:-10]))

            while True:
                ret, frame = cap.read()
                index = index + 1
                if ret:
                    assert frame.shape[2] == 3
                    if os.path.exists(os.path.join(out_folder, name[:-10], '{:04d}.jpg'.format(index))):
                        print('Skipping ', os.path.join(out_folder, name[:-10], '{:04d}.jpg'.format(index)))
                        continue

                    hha = cv2.imread(os.path.join(folder_hha, name[:-10], 'frame{:d}.png'.format(index)))
                    assert hha.shape[2] == 3
                    mask = frame == 0
                    assert mask.shape == hha.shape
                    # hha[mask] = [0, 0, 0]
                    hha[mask] = 0
                    # camera_matrix = getCameraParam('color')
                    # frame = getHHA(camera_matrix, frame, frame)
                    hha = crop(hha, xy_center, xy_radius)

                    hha = cv2.resize(hha, (256,256))

                    cv2.imwrite(os.path.join(out_folder, name[:-10], '{:04d}.jpg'.format(index)), hha)
                    print('Saved, ', os.path.join(out_folder, name[:-10], '{:04d}.jpg'.format(index)))

                else:
                    break

                
                # break
            # break

        # break
