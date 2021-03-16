# Isolated Sign Language Recognition using Conv3D
This folder contains the training, finetuning code for the following modalities.
## Pretrained models
Pretrained models can be downloaded via [Google Drive](https://drive.google.com/drive/folders/1yn3K_BdnDwqYXL-z03wv1FA0aQL7gdsU?usp=sharing). Those pretrained models can be used to reproduce our final results using the testing code. 

## Requirement
See [requirements.txt](requirements.txt)
## RGB Frames 
The RGB frames modality can be trained, finetuned and tested using the following commands in Conv3D/ folder.  

    python Sign_Isolated_Conv3D_clip.py

    python Sign_Isolated_Conv3D_clip_finetune.py

    python Sign_Isolated_Conv3D_clip_test.py

## Flow Color 
The Flow Color modality can be trained, finetuned and tested using the following commands

    python Sign_Isolated_Conv3D_flow_clip.py

    python Sign_Isolated_Conv3D_flow_clip_funtine.py

    python Sign_Isolated_Conv3D_flow_clip_test.py

## HHA
The HHA modality can be trained, finetuned and tested using the following commands

    python Sign_Isolated_Conv3D_hha_clip_mask.py

    python Sign_Isolated_Conv3D_hha_clip_mask_finetune.py

    python Sign_Isolated_Conv3D_hha_clip_mask_test.py

## Flow Depth 
The Flow Depth modality can be trained, finetuned and tested using the following commands 

    python Sign_Isolated_Conv3D_depth_flow_clip.py

    python Sign_Isolated_Conv3D_depth_flow_clip_finetune.py

    python Sign_Isolated_Conv3D_depth_flow_clip_test.py

## Ensemble
The results .pkl files of the above modalities will be saved in results/ folder. Please rename and copy them to ../ensemble/ folder for model ensemble.
