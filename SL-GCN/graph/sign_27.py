import sys

sys.path.extend(['../'])
from graph import tools
import pandas as pd

num_node = 29 # 29 or 71
points = pd.read_csv(f"points_{num_node}.csv")
ori = points.origin
tar = points.tarjet

inward_ori_index = [(o,t) for o, t in zip(ori, tar)]


print("NUM OF NODES:", num_node)
self_link = [(i, i) for i in range(num_node)]

'''
inward_ori_index = [(5, 6), (5, 7),
                    (6, 8), (8, 10), (7, 9), (9, 11), 
                    (12,13),(12,14),(12,16),(12,18),(12,20),
                    (14,15),(16,17),(18,19),(20,21),
                    (22,23),(22,24),(22,26),(22,28),(22,30),
                    (24,25),(26,27),(28,29),(30,31),
                    (10,12),(11,22)]


inward_ori_index = [(1, 2), (1, 3), (2, 4), (4, 6), (3, 5), (5, 7),

                    (6, 8),
                    (8, 9), (9, 10), (10, 11), (11, 12),
                    (8, 13), (13, 14), (14, 15), (15, 16),
                    (8, 17), (17, 18), (18, 19), (19, 20),
                    (8, 21), (21, 22), (22, 23), (23, 24),
                    (8, 25), (25, 26), (26, 27), (27, 28), 

                    (7, 29),
                    (29, 30), (30, 31), (31, 32), (32, 33),
                    (29, 34), (34, 35), (35, 36), (36, 37),
                    (29, 38), (38, 39), (39, 40), (40, 41),
                    (29, 42), (42, 43), (43, 44), (44, 45),
                    (29, 46), (46, 47), (47, 48), (48, 49)
                    ]
'''

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
