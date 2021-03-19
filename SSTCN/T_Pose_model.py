import math
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
# groups == joints_number
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups,
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
class swish(nn.Module):
    def __init__(self, inplace=True):
        super(swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
    
class Attention(nn.Module):
    def __init__(self, channel, mid,groups):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, mid, 1, padding=0,groups=groups, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, channel, 1, padding=0,groups=groups, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

        
def conv1x1(in_channels, out_channels, groups=133):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1,
        padding = 0)
        
def conv3x3(in_channels, out_channels, groups=1,stride=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        groups=groups,
        stride=stride,
        padding = 1)

def get_inplanes():
    return [64, 128, 256, 512]



class TemporalDownsampleBlock(nn.Module):

    def __init__(self, in_planes, planes,joints_number,mid, relu = True, stride=1, downsample=None):
        super().__init__()
        # in_planes and planes are both n*joints_number, n is integer
        self.joints_number = joints_number
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            in_planes,
            planes,
            self.joints_number,
            mid,
            batch_norm=True,
            relu=relu
            )
    def _make_grouped_conv1x1(self, in_channels, out_channels, groups, mid,
        batch_norm=True, relu=False):

        modules = OrderedDict()
        attention = Attention(in_channels, mid, groups=groups)
        modules['attention'] = attention
        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['swish'] = swish()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv
            

    def forward(self, x):
        out = self.g_conv_1x1_compress(x)
        return out
        

class FrameDownsampleBlock(nn.Module):

    def __init__(self, in_planes, planes, frames,mid, relu = True, stride=1, downsample=None):
        super().__init__()
        # in_planes and planes are both n*joints_number, n is integer
        self.frames = frames

        self.g_conv_3x3_compress = self._make_grouped_conv3x3(
            in_planes,
            planes,
            self.frames,
            mid,
            stride,
            batch_norm=True,
            relu=relu
            )

    def _make_grouped_conv3x3(self, in_channels, out_channels, groups, mid, stride,
        batch_norm=True, relu=False):

        modules = OrderedDict()
        attention = Attention(in_channels, mid, groups=groups)
        modules['attention'] = attention
        conv = conv3x3(in_channels, out_channels, groups=groups,stride=stride)
        modules['conv3x3'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['swish'] = swish()
            #modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv
            


    def forward(self, x):
        out = self.g_conv_3x3_compress(x)
        return out


class T_Pose_model(nn.Module):

    def __init__(self,frames_number,joints_number,n_classes=226):
        super().__init__()
        self.in_channels = frames_number
        self.joints_number = joints_number
        self.final_frames_number = frames_number

        self.bn = nn.BatchNorm2d(self.in_channels*self.joints_number)
        self.swish = swish()
        self.t1downsample = TemporalDownsampleBlock(self.in_channels,frames_number,1,10)
        self.t2downsample = TemporalDownsampleBlock(frames_number,self.final_frames_number,1,10,relu = False)
        self.f1downsample = FrameDownsampleBlock(self.final_frames_number*joints_number,
        self.final_frames_number*joints_number,joints_number,joints_number*10)
        self.f2downsample = FrameDownsampleBlock(self.final_frames_number*joints_number,
        self.final_frames_number*joints_number,joints_number,joints_number*10,relu = False)
        self.f3downsample = FrameDownsampleBlock(self.final_frames_number*joints_number,
        self.final_frames_number*joints_number,self.final_frames_number,self.final_frames_number*10)
        self.f4downsample = FrameDownsampleBlock(self.final_frames_number*joints_number,
        self.final_frames_number*joints_number,self.final_frames_number,self.final_frames_number*10, relu = False)
        self.f5downsample = FrameDownsampleBlock(self.final_frames_number*joints_number,
        self.final_frames_number*joints_number,1,10*10)
        self.f6downsample = FrameDownsampleBlock(self.final_frames_number*joints_number,30*joints_number,1,10*10)
        self.dropout = nn.Dropout2d(0.333)
        self.fc1 = nn.Linear(990, n_classes)
    def forward(self, x):
        batchsize,num_channels, height, width = x.data.size()
        x = self.bn(x)
        x = self.swish(x)
        res = x
        x = x.view(-1,self.in_channels,self.joints_number*height,width)
        x = self.t1downsample(x)
        x = self.t2downsample(x)
        x = x.view(-1,self.final_frames_number*self.joints_number,height,width)
        x = res + x
        x = self.swish(x)
        res = x
        x = channel_shuffle(x,self.final_frames_number)
        x = self.f1downsample(x)
        x = self.dropout(x)
        x = self.f2downsample(x)
        x = channel_shuffle(x,self.joints_number)
        x = x + res
        x = self.swish(x)
        res = x
        x = self.f3downsample(x)
        x = self.dropout(x)
        x = self.f4downsample(x)
        x = x + res
        x = self.swish(x)

        x = self.f5downsample(x)
        x = self.dropout(x)
        x = self.f6downsample(x)
        x = F.avg_pool2d(x, x.data.size()[-2:])
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')#, nonlinearity='relu')
                #nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
