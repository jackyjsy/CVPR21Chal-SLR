import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from model.dropSke import DropBlock_Ske
from model.dropT import DropBlockT_1d


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, num_point=25, block_size=41):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))
        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
                                      3, 1, num_point, num_point]), dtype=torch.float32, requires_grad=True).repeat(1, groups, 1, 1), requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * num_subset, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * num_subset, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant(self.Linear_bias, 1e-6)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(
            eye_array), requires_grad=False, device='cuda'), requires_grad=False)  # [c,25,25]

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        learn_A = self.DecoupleA.repeat(
            1, self.out_channels // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(
            learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = torch.einsum(
            'nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size, stride=1, residual=True, attention=True):
        super(TCN_GCN_unit, self).__init__()
        num_jpts = A.shape[-1]
        self.gcn1 = unit_gcn(in_channels, out_channels, A, groups, num_point)
        self.tcn1 = unit_tcn(out_channels, out_channels,
                             stride=stride, num_point=num_point)
        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
                              3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'), requires_grad=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(
                in_channels, out_channels, kernel_size=1, stride=stride)
        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)
        self.attention = attention
        if attention:
            print('Attention Enabled!')
            self.sigmoid = nn.Sigmoid()
            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)
            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x, keep_prob):
        y = self.gcn1(x)
        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)
            
        y = self.tcn1(y, keep_prob, self.A)
        x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(y + x_skip)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, groups=8, block_size=41, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64, A, groups, num_point,
                               block_size, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.l3 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.l4 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.l5 = TCN_GCN_unit(
            64, 128, A, groups, num_point, block_size, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.l7 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.l8 = TCN_GCN_unit(128, 256, A, groups,
                               num_point, block_size, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.l10 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, keep_prob=0.9):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x, 1.0)
        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        x = self.l7(x, keep_prob)
        x = self.l8(x, keep_prob)
        x = self.l9(x, keep_prob)
        x = self.l10(x, keep_prob)

        # N*M,C,T,V
        c_new = x.size(1)

        # print(x.size())
        # print(N, M, c_new)

        # x = x.view(N, M, c_new, -1)
        x = x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
