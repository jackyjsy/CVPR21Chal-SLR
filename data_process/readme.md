# Multi-modal Data Preparation and Processing
## Generate whole-body skeleton keypoints and save as npy
Use pretrained model of whole-body pose estimation to extract 133 landmarks from rgb videos and save as npy files. 

1. Go to wholepose folder, change input_path and output_npy variables as the path of input videos and output npy files.

2. Download pretrained whole-body pose model: [Google Drive](https://drive.google.com/file/d/1f_c3uKTDQ4DR3CrwMSI8qdsTKJvKVt7p/view?usp=sharing)

3. Run `python demo.py`

4. Copy generated npy files to corresponding data folders.

## Generate skeleton features

Use the [../TPose/data_process/wholepose_features_extraction.py](../TPose/data_process/wholepose_features_extraction.py) to extract skeleton features.

## Generate rgb frames from rgb videos
Get frames from RGB videos and crop to 256x256 according to the whole-pose skeletons extracted above.

1. Change folder, npy_folder, out_folder variables accordingly in [gen_frames.py](gen_frames.py).

2. Run `python gen_frames.py`

## Generate flow data from rgb and depth videos
There are two types of flow modality: color flow and depth flow. Those data can be obtained by pretrained Caffe model first. Then combine flow_x and flow_y and crop the combined flow data using [gen_flow.py](gen_flow.py).

1. Obtain raw flow data from videos using docker as described in optical_flow_guidelines.docx

2. Change folder, npy_folder, out_folder variables accordingly in [gen_flow.py](gen_flow.py).

3. Run `python gen_flow.py`

## Generate HHA representation from depth videos

Use matlab code in Depth2HHA_master_mat to extract HHA from depth videos. It takes a long time extracting HHA features. And then crop the hha images and maskout pixels using [gen_hha.py](gen_hha.py).

1. Change input_folder and output_folder and hha_root variables accordingly in CVPR21Chal_convert_HHA.m and run the script.

2. Change  folder, npy_folder, out_folder variables accordingly in [gen_hha.py](gen_hha.py).

3. Run  `python gen_hha.py`