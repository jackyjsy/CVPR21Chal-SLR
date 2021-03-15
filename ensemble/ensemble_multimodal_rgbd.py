import argparse
import pickle

import numpy as np
from tqdm import tqdm

label = open('test_labels_pseudo.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('test_gcn_w_val_finetune.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('test_rgb_w_val_finetune.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('test_flow_color_w_val_finetune.pkl','rb')
r3 = list(pickle.load(r3).items())
r4 = open('test_feature_w_val_finetune.pkl', 'rb')
r4 = list(pickle.load(r4).items())
r5 = open('test_hha_w_val_finetune.pkl', 'rb')
r5 = list(pickle.load(r5).items())
r6 = open('test_flow_depth_w_val_finetune.pkl', 'rb')
r6 = list(pickle.load(r6).items())


alpha = [1.0,1.4,0.5,0.4,0.5,0.4]  # gcn, rgb_multi2, flow_color2, skeleton feature, hha, flow_depth
# alpha = [1.0,0.9,0.4,0.4,0.4,0.1]  # gcn, rgb_multi2, flow_color2, skeleton feature, hha, flow_depth

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
with open('predictions.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        name5, r55 = r5[i]
        name6, r66 = r6[i]
        assert name == name1 == name2 == name3
        score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3] + r55*alpha[4] + r66*alpha[5]) / np.array(alpha).sum() 
        score = score.squeeze()
        rank_5 = score.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(score)
        scores.append(score)
        preds.append(r)
        right_num += int(r == int(l))
        total_num += 1
        f.write('{}, {}\n'.format(name, r))
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(total_num)
    print('top1: ', acc)
    print('top5: ', acc5)

f.close()

# with open('./val_score.pkl', 'wb') as f:
#     score_dict = dict(zip(names, scores))
#     pickle.dump(score_dict, f)