import sys

sys.path.extend(['../'])
from graph import tools

num_node = 27
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(5, 6), (5, 7),
                    (6, 8), (8, 10), (7, 9), (9, 11), 
                    (12,13),(12,14),(12,16),(12,18),(12,20),
                    (14,15),(16,17),(18,19),(20,21),
                    (22,23),(22,24),(22,26),(22,28),(22,30),
                    (24,25),(26,27),(28,29),(30,31),
                    (10,12),(11,22)]

inward = [(i - 5, j - 5) for (i, j) in inward_ori_index]
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
