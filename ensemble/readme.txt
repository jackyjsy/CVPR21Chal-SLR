1. Copy results from GCN (joint, bone, joint motion, bone motion) to gcn/ folder.

2. Ensemble gcn results by running

goto gcn/ follow the readme.txt instructions

3. Copy RGB, Flow, HHA, Depth flow results from Conv3D/ folder. Copy feature results from TPose (feature) folder. Rename them accordingly.

4. Ensemble RGB/RGBD results by running

python ensemble_multimodal_rgb.py

python ensemble_multimodal_rgbd.py

We provide the .pkl files we used to obtain our final results as well.