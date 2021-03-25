Instructions to reproduce our results using pretrained models
=====
Here we provide a step-step instruction to reproduce our results using our docker image.
### I: The zip files of pretrained models should be decompressed and move to their folders:

* Conv3D/pretrained.zip needs to be decompressed and put in Conv3D/ folder in our code repo. (contains pretrained RGB model using Chinese Sign Language dataset)
* Conv3D/final_models_finetuned.zip needs to be decompressed and put in Conv3D/ folder in our code repo. (contains pretrained our final models of all modalities)
* SL-GCN/27_2.zip contains models trained using training set only and needs to be decompressed and put under SL-GCN/final_models/ in our code repo.
* SL-GCN/27_2_finetuned.zip contains models trained using training set and validation set which needs to be decompressed and put under SL-GCN/final_models/ in our code repo.
* SSTCN/T_Pose_model_final.pth do not need decompressing and can be put under SSTCN/ folder directly.
------
### II: Nvidia docker image:
Our docker image can be load using the following command:
```
sudo docker run -it --gpus all -v path_to_your_data:/home/smilelab_slr/cvpr2021_allcode/shared_data cvpr2021cha_code /bin/bash
```
------
### III: Procedues to reproduce our results

Here I provide a brief pipeline to help reproduce our results, detailed instruction can be found in our code repo:

#### A. For RGB track: 

1. Please test SL-GCN using the following script. All the config files in SL-GCN/config/test and SL-GCN/config/test_finetuned need to be tested.
```
python main.py --config /path/to/config/file
```

2. Please test the other modalities using:
```
cd Conv3D
python Sign_Isolated_Conv3D_clip_test.py
python Sign_Isolated_Conv3D_flow_clip_test.py
cd SSTCN
python test.py
```

The test results are saved as .pkl files.

3. Ensemble the final results following the instructions in ensemble/ folder.

- ensemble gcn multi-stream results first.
- ensemble rgb results using ensemble_multimodal_rgb.py

------
#### For RGB-D track: 

1. Please test SL-GCN using the following script. All the config files in SL-GCN/config/test and SL-GCN/config/test_finetuned need to be tested.
```
python main.py --config /path/to/config/file
```

2. Please 'cd' to each folder and test the pretrained models using the following scripts:
```
cd Conv3D
python Sign_Isolated_Conv3D_clip_test.py
python Sign_Isolated_Conv3D_flow_clip_test.py
python Sign_Isolated_Conv3D_hha_clip_mask_test.py
python /Sign_Isolated_Conv3D_depth_flow_clip_test.py
cd SSTCN
python test.py
```

The test results are saved as .pkl files.

3. Ensemble the final results following the instructions in ensemble/ folder.

- ensemble gcn multi-stream results first.
- ensemble rgbd results using ensemble_multimodal_rgbd.py

Please note that you need to rename and copy all the generated .pkl files to the ensemble folder.