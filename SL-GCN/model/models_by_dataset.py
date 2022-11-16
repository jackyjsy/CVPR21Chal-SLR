#AEC-29 v1


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, groups=8, block_size=41, graph=None, graph_args=dict(), in_channels=2,model_version=1):
        super(Model, self).__init__()
        self.model_version = model_version
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            self.graph.num_node = num_point

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        
        if self.model_version == 1:
            self.l1 = TCN_GCN_unit(in_channels, 32, A, groups, num_point,block_size, residual=False)
            self.l2 = TCN_GCN_unit(32, 32, A, groups,num_point, block_size, stride=2)
            self.fc = nn.Linear(32, num_class)
        
        if self.model_version == 2:
            self.l1 = TCN_GCN_unit(in_channels, 16, A, groups, num_point,block_size, residual=False)
            self.l2 = TCN_GCN_unit(16, 32, A, groups,num_point, block_size, stride=2)
            self.fc = nn.Linear(32, num_class)
        
        if self.model_version == 3:
            self.l1 = TCN_GCN_unit(in_channels, 32, A, groups, num_point,block_size, residual=False)
            self.fc = nn.Linear(32, num_class)
        
        if self.model_version == 4:
            self.l1 = TCN_GCN_unit(in_channels, 16, A, groups, num_point,block_size, residual=False)
            self.fc = nn.Linear(16, num_class)
        
        
        
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, keep_prob=0.9):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        if self.model_version == 1:
            x = self.l1(x, 1.0)
            x = self.l2(x, keep_prob)
        if self.model_version == 2:
            x = self.l1(x, 1.0)
            x = self.l2(x, keep_prob)
        if self.model_version == 3:
            x = self.l1(x, 1.0)
        if self.model_version == 4:
            x = self.l1(x, 1.0)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
    