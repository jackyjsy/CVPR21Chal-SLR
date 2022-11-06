#!/bin/bash

#########################################################
#python main.py --config config/sign/train/train_joint.yaml
  #num_class: 28 # AEC=28, PUCP=29 , WLASL=86
  #num_point: 29 # 29 or 71 or 51

#declare -a points=(51 51 51)
#declare -a lrs=(0.05 0.05 0.05)
#declare -a datasets=("PUCP" "AEC" "WLASL")

declare -a points=(29 51 71 29 51 71 29 51 71)
declare -a lrs=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)
declare -a datasets=("AEC" "AEC" "AEC" "PUCP" "PUCP" "PUCP" "WLASL" "WLASL" "WLASL")
declare -a model_version=(0 1 2 3 4 5 6)

'''
for i in 0
do
  for j in 1 2 3 4 5 6 7 8
  do 
      python main.py --seed $i --experiment_name "results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model wholepose --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 600  --mode_train optimizacion_analysis_aec_29_v5
      python main.py --seed $i --experiment_name "results/${points[j]}/${datasets[j]}/mediapipe-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model mediapipe --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 600  --mode_train optimizacion_analysis_aec_29_v5 --weights "save_models/results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i/wholepose-${datasets[j]}-${points[j]}-$i-init.pt"
      python main.py --seed $i --experiment_name "results/${points[j]}/${datasets[j]}/openpose-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model openpose --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 600  --mode_train optimizacion_analysis_aec_29_v5 --weights "save_models/results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i/wholepose-${datasets[j]}-${points[j]}-$i-init.pt"
  done
done
'''
#  for j in 4 5 6 7 8
'''
for i in 0
do
  for j in 6 7 8 5 4 3 2 1 0
  do 
      python main.py --seed $i --experiment_name "results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model wholepose --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 1200  --mode_train optimizacion_analysis_v4
      python main.py --seed $i --experiment_name "results/${points[j]}/${datasets[j]}/mediapipe-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model mediapipe --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 1200  --mode_train optimizacion_analysis_v4 --weights "save_models/results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i/wholepose-${datasets[j]}-${points[j]}-$i-init.pt"
      python main.py --seed $i --experiment_name "results/${points[j]}/${datasets[j]}/openpose-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model openpose --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 1200  --mode_train optimizacion_analysis_v4 --weights "save_models/results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i/wholepose-${datasets[j]}-${points[j]}-$i-init.pt"
  done
done
'''

######## get number parameters ############
for w in 0 1 2 3 4 5 6 # model version
do 
  for i in 0 # seed
  do
    for j in 0 1 2 3 4 5 6 7 8 # dataset-keypoint
    do 
        python main.py --seed $i --model_version ${model_version[w]} --experiment_name "results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model wholepose --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 1  --mode_train "optimizacion_analysis_get_ratio_v${model_version[w]}"
        python main.py --seed $i --model_version ${model_version[w]} --experiment_name "results/${points[j]}/${datasets[j]}/mediapipe-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model mediapipe --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 1  --mode_train "optimizacion_analysis_get_ratio_v${model_version[w]}" --weights "save_models/results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i/wholepose-${datasets[j]}-${points[j]}-$i-init.pt"
        python main.py --seed $i --model_version ${model_version[w]} --experiment_name "results/${points[j]}/${datasets[j]}/openpose-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model openpose --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 1  --mode_train "optimizacion_analysis_get_ratio_v${model_version[w]}" --weights "save_models/results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i/wholepose-${datasets[j]}-${points[j]}-$i-init.pt"
    done
  done
done