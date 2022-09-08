# Skeleton Based Sign Language Recognition

## To use with [Connecting Points](https://github.com/JoeNatan30/ConnectingPoints) repository

### Generate smile-lab data split (from the split of connecting points) and Smile-lab model variable preparation to train
1. Run "data_gen/getConnectingPoint.py" in data_gen folder (Not forget to modify "kpModel", "numPoints" and "dataset" variable)

2. Modify "num_point", "num_class" and "device" variable of the yaml file "/config/sign/train/train_joint.yaml" as it is needed (same as setted in the previous step)

3. Modify "num_node" variable in sign_27

4. Go to "if __name__ == '__main__':" section of main.py (in SL-GCN folder) and modify "config" paremeters

5. run
```
python main.py --config config/sign/train/train_joint.yaml
```

Note: if you don't have a wandb account, you need to set "wandbFlag" variable of "main.py" to False and modify the code to have reports 

---------------------------
