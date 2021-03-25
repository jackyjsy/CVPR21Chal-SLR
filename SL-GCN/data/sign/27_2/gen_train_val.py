import numpy as np

parts = {'joint', 'bone2', 'joint_motion', 'bone2_motion'}

for part in parts:
    print(part)
    data_train = np.load('train_data_{}.npy'.format(part))
    data_val = np.load('val_data_{}.npy'.format(part))

    data_train_val = np.concatenate((data_train, data_val), axis=0)
    print(data_train_val.shape)


    np.save('train_val_data_{}.npy'.format(part), data_train_val)