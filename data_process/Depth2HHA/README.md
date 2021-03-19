# Depth2HHA
https://github.com/charlesCXK/Depth2HHA

Please use CVPR21Chal_convert_HHA.m to extract HHA features from AUTSL dataset.

The followings are original readme from the above repo.

----

### Introduction

**First**, I want to thank **<a href='https://github.com/s-gupta'>s-gupta</a>** for his excellent work. Actually, I got some inspiration from his code and fixed some problems in it.

This repo is used to convert Depth images into HHA images. HHA is an encoding method which extract the information in the depth image which was proposed in <a href='https://arxiv.org/pdf/1407.5736.pdf'>Learning Rich Features from RGB-D Images for Object Detection and Segmentation</a>.  In this repo, I use <a href='https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html'>NYU Depth V2</a> dataset as an example.

<br>

**All we need** is: 

1. A depth image
2. A raw-depth image

<br>

----

### Usage

#### 1. Modify **main.m**

```matlab
depth_image_root = './depth'       % dir where depth images are in.
rawdepth_image_root = './rawdepth'       % dir where raw depth images are in.
hha_image_root = './hha'

saveHHA(['img_', mat2str(5000+i)], matrix, depth_image_root, D, RD);
```

***depth_image_root*** is the directory of where depth images are in, and ***rawdepth_image_root*** is the directory of where **raw** depth images are in. The path of HHA can be defined by yourself by modifying parameters of ***saveHHA(imName, C, outDir, D, RD)***. 

`imName` : name of HHA.

`C` : camera matrix

`outDir` : root of HHA images

`D and RD` : depth and  raw-depth images. The raw-depth images are just used as masks, which you can understand from the corresponding code. 

#### 2. Modify saveHHA.m

Look at this line.

```matlab
D = double(D)/1000;        % The unit of the element inside D is 'centimeter'
```

Here, *D* is the depth image. You may confused about the number '1000'. Because when I save the depth data as 'png', I multiply it with 1000. We all know that float number can not be saved as png or jpg, so I scale it. **Anyway**, after this line, the unit of the element in *D* should be 'meter'. It's up to you how to convert it.

#### 3. Others

- You can directly run ***main.m*** to use this repo.
- You can use data that s-gupta supplies:

```shell
wget http://www.cs.berkeley.edu/~sgupta/eccv14/eccv14-data.tgz
tar -xf eccv14-data.tgz
```



----

<!-- ### Results

Pictures below are RGB, Depth, Raw-depth, HHA in turn.

<img src='demo-data/1.png' width='400'>

<img src='demo-data/1_depth.png' width='400'>

<img src='demo-data/1_rawdepth.png' width='400'>

<img src='demo-data/1_hha.png' width='400'> -->