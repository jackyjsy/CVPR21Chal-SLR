#!/bin/bash

#########################################################
#python main.py --config config/sign/train/train_joint.yaml
  #num_class: 28 # AEC=28, PUCP=29 , WLASL=86
  #num_point: 29 # 29 or 71 or 51

declare -a points=(51 51 51)
declare -a lrs=(0.05 0.05 0.05)
declare -a datasets=("PUCP" "AEC" "WLASL")



for i in 0 5 15 25 35
do
  for j in 0 1 2
  do 
      python main.py --seed $i --experiment_name "results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model wholepose --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 400  --mode_train neurips_51points_v4
      python main.py --seed $i --experiment_name "results/${points[j]}/${datasets[j]}/mediapipe-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model mediapipe --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 400  --mode_train neurips_51points_v4 --weights "save_models/results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i/wholepose-${datasets[j]}-${points[j]}-$i-init.pt"
      python main.py --seed $i --experiment_name "results/${points[j]}/${datasets[j]}/openpose-${datasets[j]}-s-$i" --database ${datasets[j]} --keypoints_model openpose --base_lr ${lrs[j]} --keypoints_number ${points[j]} --num_epoch 400  --mode_train neurips_51points_v4 --weights "save_models/results/${points[j]}/${datasets[j]}/wholepose-${datasets[j]}-s-$i/wholepose-${datasets[j]}-${points[j]}-$i-init.pt"
  done
done

