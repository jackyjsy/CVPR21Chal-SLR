# Skeleton Based Sign Language Recognition
## Data preparation
``cd data_gen/``
## Train
``python main.py --config config/sign/train/train_joint.yaml``

``python main.py --config config/sign/train/train_bone.yaml``

``python main.py --config config/sign/train/train_joint_motion.yaml``

``python main.py --config config/sign/train/train_bone_motion.yaml``
## Finetune
``python main.py --config config/sign/finetune/train_joint.yaml``

``python main.py --config config/sign/finetune/train_bone.yaml``

``python main.py --config config/sign/finetune/train_joint_motion.yaml``

``python main.py --config config/sign/finetune/train_bone_motion.yaml``
## Test
``python main.py --config config/sign/finetune/test_joint.yaml``

``python main.py --config config/sign/finetune/test_bone.yaml``

``python main.py --config config/sign/finetune/test_joint_motion.yaml``

``python main.py --config config/sign/finetune/test_bone_motion.yaml``