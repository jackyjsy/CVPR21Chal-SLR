# Skeleton Based Sign Language Recognition

We prepare this runModel.sh file to run the model.

Note: If you do not use WandB please change to False "wandbFlag" variable value in main.py (line 34)

## main.py

* --seed => set the seed to train [used 5, 15, 25, 35 and 45]
* --experiment_name => The name where the model will be saved
* --database => part of the name of the HDF5 file that will be used to retrieve the imput data (e.x: AEC, PUCP-DGI156, WLASL)
* --keypoints_model => part of the name of the HDF5 file that will be used to retrieve the imput data (e.x: mediapipe, wholepose, openpose)
* --base_lr => learning rate 
* --keypoints_number => number of keypoints used (29 or 71)
* --num_epoch => number of epochs 
* --mode_train => to show in wandb the number of parameters (always write: numero_parametros"
* --cleaned => add this to use the cleaned data from Connecting points repository


---------------

To automatize our work we create this .sh file
modify it as you need

# in windows
```
bash runModel.sh
```
# In Linux
```
sh runModel.sh
```
