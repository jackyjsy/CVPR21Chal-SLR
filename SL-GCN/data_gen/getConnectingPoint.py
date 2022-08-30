import pickle
import sys
import numpy as np
import pandas as pd
import os
import h5py
import pandas as pd
sys.path.extend(['../'])

max_body_true = 1
max_frame = 150
num_channels = 2

def get_mp_keys(points):
    tar = np.array(points.mp_pos)-1
    return list(tar)

def get_op_keys(points):
    tar = np.array(points.op_pos)-1
    return list(tar)

def get_wp_keys(points):
    tar = np.array(points.wb_pos)-1
    return list(tar)

def read_data(path, model_key_getter):
    data = []
    classes = []
    videoName = []

    with h5py.File(path, "r") as f:
        for index in f.keys():
            classes.append(f[index]['label'][...].item().decode('utf-8'))
            videoName.append(f[index]['video_name'][...].item().decode('utf-8'))
            data.append(f[index]["data"][...])

    points = pd.read_csv("../points.csv")

    tar = model_key_getter(points)

    data = [d[:,:,tar] for d in data]

    meaning = {v:k for (k,v) in enumerate(sorted(set(classes)))}

    retrive_meaning = {k:v for (k,v) in enumerate(sorted(set(classes)))}
    
    labels = [meaning[label] for label in classes]

    return labels, videoName, data, retrive_meaning
    

def gendata(data_path,  out_path, model_key_getter, part='train', config='27'):

    data=[]
    sample_names = []

    labels, sample_names, data , retrive_meaning = read_data(data_path, model_key_getter)
    fp = np.zeros((len(labels), max_frame, 71, num_channels, max_body_true), dtype=np.float32)

    for i, skel in enumerate(data):

        skel = np.array(skel)
        skel = np.moveaxis(skel,1,2)
        skel = skel # *256
        
        if skel.shape[0] < max_frame:
            L = skel.shape[0]

            fp[i,:L,:,:,0] = skel

            rest = max_frame - L
            num = int(np.ceil(rest / L))
            pad = np.concatenate([skel for _ in range(num)], 0)[:rest]
            fp[i,L:,:,:,0] = pad

        else:
            L = skel.shape[0]

            fp[i,:,:,:,0] = skel[:max_frame,:,:]

    
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_names, labels), f)

    fp = np.transpose(fp, [0, 3, 1, 2, 4])
    print(fp.shape)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

    with open('{}/meaning.pkl'.format(out_path), 'wb') as f:
        pickle.dump(retrive_meaning, f)




if __name__ == '__main__':

    points= '1'
    out_folder='../data/sign/'
    out_path = os.path.join(out_folder, points)

    kp_model = 'wholepose' # openpose wholepose mediapipe
    dataset = "WLASL" # WLASL PUCP_PSL_DGI156 AEC

    model_key_getter = {'mediapipe': get_mp_keys,
                        'openpose': get_op_keys,
                        'wholepose': get_wp_keys}

    if not os.path.exists(out_path):
        os.makedirs(out_path)


    print('\n',kp_model,'\n')

    part = "train"
    data_path = f'../../../ConnectingPoints/split/{dataset}--{kp_model}-v2-Train.hdf5'
    gendata(data_path, out_path, model_key_getter[kp_model], part=part, config=points)
    
    print(out_path)
    part = "val"
    data_path = f'../../../ConnectingPoints/split/{dataset}--{kp_model}-v2-Val.hdf5'
    
    gendata(data_path, out_path, model_key_getter[kp_model], part=part, config=points)
