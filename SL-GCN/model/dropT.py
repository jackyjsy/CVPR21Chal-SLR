import torch
import torch.nn.functional as F
from torch import nn

class DropBlockT_1d(nn.Module):
    def __init__(self, block_size=7):
        super(DropBlockT_1d, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n,c,t,v = input.size()

        input_abs = torch.mean(torch.mean(torch.abs(input),dim=3),dim=1).detach()
        input_abs = (input_abs/torch.sum(input_abs)*input_abs.numel()).view(n,1,t)
        gamma = (1. - self.keep_prob) / self.block_size
        input1 = input.permute(0,1,3,2).contiguous().view(n,c*v,t)
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1,c*v,1)
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        mask = (1 - Msum).to(device=input.device, dtype=input.dtype)
        return (input1 * mask * mask.numel() /mask.sum()).view(n,c,v,t).permute(0,1,3,2)
