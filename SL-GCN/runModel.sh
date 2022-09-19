#python main.py --config config/sign/train/train_joint.yaml
  #num_class: 28 # AEC=28, PUCP=29 , WLASL=86

  #num_point: 29 # 29 or 71

'''
########## tunning ########### 71

python main.py --experiment_name results/f_71/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.05 --num_class 28 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.01 --num_class 28 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.005 --num_class 28 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.001 --num_class 28 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.0005 --num_class 28 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.0001 --num_class 28 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 

python main.py --experiment_name results/f_71/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.05 --num_class 86 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.01 --num_class 86 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.005 --num_class 86 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.001 --num_class 86 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.0005 --num_class 86 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.0001 --num_class 86 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 

python main.py --experiment_name results/f_71/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.05 --num_class 29 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.01 --num_class 29 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.005 --num_class 29 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.001 --num_class 29 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.0005 --num_class 29 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_71/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.0001 --num_class 29 --keypoints_number 71 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 

########## tunning ########### 29

python main.py --experiment_name results/f_29/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.05 --num_class 28 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.01 --num_class 28 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.005 --num_class 28 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.001 --num_class 28 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.0005 --num_class 28 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/AEC/cris_wholepose_AEC --mode_train tunning --database AEC --keypoints_model wholepose --base_lr 0.0001 --num_class 28 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/AEC--wholepose-Train.hdf5 --testing_set_path ../../../dataset_original/AEC--wholepose-Val.hdf5 --config config/sign/train/train_joint.yaml 

python main.py --experiment_name results/f_29/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.05 --num_class 86 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.01 --num_class 86 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.005 --num_class 86 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.001 --num_class 86 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.0005 --num_class 86 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/WLASL/cris_mediapipe_WLASL --mode_train tunning --database WLASL --keypoints_model mediapipe --base_lr 0.0001 --num_class 86 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/WLASL--mediapipe-Train.hdf5 --testing_set_path ../../../dataset_original/WLASL--mediapipe-Val.hdf5 --config config/sign/train/train_joint.yaml 

python main.py --experiment_name results/f_29/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.05 --num_class 29 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.01 --num_class 29 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.005 --num_class 29 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.001 --num_class 29 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.0005 --num_class 29 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 
python main.py --experiment_name results/f_29/PUCP/cris_openpose_PUCP --mode_train tunning --database PUCP --keypoints_model openpose --base_lr 0.0001 --num_class 29 --keypoints_number 29 --num_epoch 500 --training_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Train.hdf5 --testing_set_path ../../../dataset_original/PUCP_PSL_DGI156--openpose-Val.hdf5 --config config/sign/train/train_joint.yaml 

'''
# AEC   AEC  PUCP PUCP WASL WASL
# 0.05	0.1	 0.05	0.1	 0.1	0.05

########### 5 ###########
### POINTS 71 ###
python main.py --seed 5 --experiment_name results/71/AEC/wholepose-AEC-s-42 --database AEC --keypoints_model wholepose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/71/AEC/mediapipe-AEC-s-42 --database AEC --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/71/AEC/openpose-AEC-s-42  --database AEC --keypoints_model openpose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 5 --experiment_name results/71/PUCP/wholepose-PUCP-s-42 --database PUCP --keypoints_model wholepose --base_lr 0.1 --keypoints_number 71 --num_epoch 400 --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/71/PUCP/mediapipe-PUCP-s-42 --database PUCP --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/71/PUCP/openpose-PUCP-s-42  --database PUCP --keypoints_model openpose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 5 --experiment_name results/71/WLASL/wholepose-WLASL-s-42 --database WLASL --keypoints_model wholepose --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/71/WLASL/mediapipe-WLASL-s-42 --database WLASL --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/71/WLASL/openpose-WLASL-s-42  --database WLASL --keypoints_model openpose --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
### POINTS 29 ###
python main.py --seed 5 --experiment_name results/29/AEC/wholepose-AEC-s-42 --database AEC --keypoints_model wholepose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/29/AEC/mediapipe-AEC-s-42 --database AEC --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/29/AEC/openpose-AEC-s-42  --database AEC --keypoints_model openpose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 5 --experiment_name results/29/PUCP/wholepose-PUCP-s-42 --database PUCP --keypoints_model wholepose --base_lr 0.05 --keypoints_number 29 --num_epoch 400 --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/29/PUCP/mediapipe-PUCP-s-42 --database PUCP --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/29/PUCP/openpose-PUCP-s-42  --database PUCP --keypoints_model openpose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 5 --experiment_name results/29/WLASL/wholepose-WLASL-s-42 --database WLASL --keypoints_model wholepose --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/29/WLASL/mediapipe-WLASL-s-42 --database WLASL --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/29/WLASL/openpose-WLASL-s-42  --database WLASL --keypoints_model openpose --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

########### 15 ###########
### POINTS 71 ###
python main.py --seed 15 --experiment_name results/71/AEC/wholepose-AEC-s-42 --database AEC --keypoints_model wholepose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/71/AEC/mediapipe-AEC-s-42 --database AEC --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/71/AEC/openpose-AEC-s-42  --database AEC --keypoints_model openpose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 15 --experiment_name results/71/PUCP/wholepose-PUCP-s-42 --database PUCP --keypoints_model wholepose --base_lr 0.1 --keypoints_number 71 --num_epoch 400 --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/71/PUCP/mediapipe-PUCP-s-42 --database PUCP --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/71/PUCP/openpose-PUCP-s-42  --database PUCP --keypoints_model openpose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 15 --experiment_name results/71/WLASL/wholepose-WLASL-s-42 --database WLASL --keypoints_model wholepose --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/71/WLASL/mediapipe-WLASL-s-42 --database WLASL --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/71/WLASL/openpose-WLASL-s-42  --database WLASL --keypoints_model openpose --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
### POINTS 29 ###
python main.py --seed 15 --experiment_name results/29/AEC/wholepose-AEC-s-42 --database AEC --keypoints_model wholepose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/29/AEC/mediapipe-AEC-s-42 --database AEC --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/29/AEC/openpose-AEC-s-42  --database AEC --keypoints_model openpose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 15 --experiment_name results/29/PUCP/wholepose-PUCP-s-42 --database PUCP --keypoints_model wholepose --base_lr 0.05 --keypoints_number 29 --num_epoch 400 --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/29/PUCP/mediapipe-PUCP-s-42 --database PUCP --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/29/PUCP/openpose-PUCP-s-42  --database PUCP --keypoints_model openpose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 15 --experiment_name results/29/WLASL/wholepose-WLASL-s-42 --database WLASL --keypoints_model wholepose --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/29/WLASL/mediapipe-WLASL-s-42 --database WLASL --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 15 --experiment_name results/29/WLASL/openpose-WLASL-s-42  --database WLASL --keypoints_model openpose --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

########### 25 ###########
### POINTS 71 ###
python main.py --seed 25 --experiment_name results/71/AEC/wholepose-AEC-s-42 --database AEC --keypoints_model wholepose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/71/AEC/mediapipe-AEC-s-42 --database AEC --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/71/AEC/openpose-AEC-s-42  --database AEC --keypoints_model openpose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 25 --experiment_name results/71/PUCP/wholepose-PUCP-s-42 --database PUCP --keypoints_model wholepose --base_lr 0.1 --keypoints_number 71 --num_epoch 400 --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/71/PUCP/mediapipe-PUCP-s-42 --database PUCP --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/71/PUCP/openpose-PUCP-s-42  --database PUCP --keypoints_model openpose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 25 --experiment_name results/71/WLASL/wholepose-WLASL-s-42 --database WLASL --keypoints_model wholepose --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/71/WLASL/mediapipe-WLASL-s-42 --database WLASL --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/71/WLASL/openpose-WLASL-s-42  --database WLASL --keypoints_model openpose --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
### POINTS 29 ###
python main.py --seed 25 --experiment_name results/29/AEC/wholepose-AEC-s-42 --database AEC --keypoints_model wholepose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/29/AEC/mediapipe-AEC-s-42 --database AEC --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/29/AEC/openpose-AEC-s-42  --database AEC --keypoints_model openpose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 25 --experiment_name results/29/PUCP/wholepose-PUCP-s-42 --database PUCP --keypoints_model wholepose --base_lr 0.05 --keypoints_number 29 --num_epoch 400 --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/29/PUCP/mediapipe-PUCP-s-42 --database PUCP --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/29/PUCP/openpose-PUCP-s-42  --database PUCP --keypoints_model openpose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 25 --experiment_name results/29/WLASL/wholepose-WLASL-s-42 --database WLASL --keypoints_model wholepose --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/29/WLASL/mediapipe-WLASL-s-42 --database WLASL --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 25 --experiment_name results/29/WLASL/openpose-WLASL-s-42  --database WLASL --keypoints_model openpose --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

########### 35 ###########
### POINTS 71 ###
python main.py --seed 35 --experiment_name results/71/AEC/wholepose-AEC-s-42 --database AEC --keypoints_model wholepose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/71/AEC/mediapipe-AEC-s-42 --database AEC --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/71/AEC/openpose-AEC-s-42  --database AEC --keypoints_model openpose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 35 --experiment_name results/71/PUCP/wholepose-PUCP-s-42 --database PUCP --keypoints_model wholepose --base_lr 0.1 --keypoints_number 71 --num_epoch 400 --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/71/PUCP/mediapipe-PUCP-s-42 --database PUCP --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/71/PUCP/openpose-PUCP-s-42  --database PUCP --keypoints_model openpose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 35 --experiment_name results/71/WLASL/wholepose-WLASL-s-42 --database WLASL --keypoints_model wholepose --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/71/WLASL/mediapipe-WLASL-s-42 --database WLASL --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/71/WLASL/openpose-WLASL-s-42  --database WLASL --keypoints_model openpose --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
### POINTS 29 ###
python main.py --seed 35 --experiment_name results/29/AEC/wholepose-AEC-s-42 --database AEC --keypoints_model wholepose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/29/AEC/mediapipe-AEC-s-42 --database AEC --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/29/AEC/openpose-AEC-s-42  --database AEC --keypoints_model openpose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 35 --experiment_name results/29/PUCP/wholepose-PUCP-s-42 --database PUCP --keypoints_model wholepose --base_lr 0.05 --keypoints_number 29 --num_epoch 400 --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/29/PUCP/mediapipe-PUCP-s-42 --database PUCP --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/29/PUCP/openpose-PUCP-s-42  --database PUCP --keypoints_model openpose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 35 --experiment_name results/29/WLASL/wholepose-WLASL-s-42 --database WLASL --keypoints_model wholepose --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/29/WLASL/mediapipe-WLASL-s-42 --database WLASL --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 35 --experiment_name results/29/WLASL/openpose-WLASL-s-42  --database WLASL --keypoints_model openpose --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1


########### 45 ###########
### POINTS 71 ###
python main.py --seed 45 --experiment_name results/71/AEC/wholepose-AEC-s-42 --database AEC --keypoints_model wholepose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/71/AEC/mediapipe-AEC-s-42 --database AEC --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/71/AEC/openpose-AEC-s-42  --database AEC --keypoints_model openpose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 45 --experiment_name results/71/PUCP/wholepose-PUCP-s-42 --database PUCP --keypoints_model wholepose --base_lr 0.1 --keypoints_number 71 --num_epoch 400 --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/71/PUCP/mediapipe-PUCP-s-42 --database PUCP --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/71/PUCP/openpose-PUCP-s-42  --database PUCP --keypoints_model openpose --base_lr 0.1 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 45 --experiment_name results/71/WLASL/wholepose-WLASL-s-42 --database WLASL --keypoints_model wholepose --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/71/WLASL/mediapipe-WLASL-s-42 --database WLASL --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/71/WLASL/openpose-WLASL-s-42  --database WLASL --keypoints_model openpose --base_lr 0.05 --keypoints_number 71 --num_epoch 400  --mode_train sustentacion1
### POINTS 29 ###
python main.py --seed 45 --experiment_name results/29/AEC/wholepose-AEC-s-42 --database AEC --keypoints_model wholepose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/29/AEC/mediapipe-AEC-s-42 --database AEC --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 5 --experiment_name results/29/AEC/openpose-AEC-s-42  --database AEC --keypoints_model openpose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 45 --experiment_name results/29/PUCP/wholepose-PUCP-s-42 --database PUCP --keypoints_model wholepose --base_lr 0.05 --keypoints_number 29 --num_epoch 400 --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/29/PUCP/mediapipe-PUCP-s-42 --database PUCP --keypoints_model mediapipe --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/29/PUCP/openpose-PUCP-s-42  --database PUCP --keypoints_model openpose --base_lr 0.05 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1

python main.py --seed 45 --experiment_name results/29/WLASL/wholepose-WLASL-s-42 --database WLASL --keypoints_model wholepose --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/29/WLASL/mediapipe-WLASL-s-42 --database WLASL --keypoints_model mediapipe --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
python main.py --seed 45 --experiment_name results/29/WLASL/openpose-WLASL-s-42  --database WLASL --keypoints_model openpose --base_lr 0.1 --keypoints_number 29 --num_epoch 400  --mode_train sustentacion1
