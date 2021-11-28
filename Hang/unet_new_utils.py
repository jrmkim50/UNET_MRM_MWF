# Code from: /disk/hz459/coding/ms_lesion_segmentatation/lesion_seg/

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from numpy.random import seed

from torch.autograd import Variable
from torch.nn import Parameter

import pytorch_lightning as pl


class convBlockVGG_ND(pl.LightningModule):

    def __init__(self, 
        num_channels = [9, 32], 
        is_batchnorm = True,
        dimension = 3,
        kernel_size = (3,3,1), #used to be 3,3,1; 3,3,3
        stride = 1,
        padding = (1,1,0) #used to be 1,1,0; 1,1,1
    ):
        super(convBlockVGG_ND, self).__init__()
        self.num_channels = num_channels
        self.is_batchnorm = is_batchnorm
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        num_groups = 32 if num_channels[0] < 64 or num_channels[1] < 64 else 64

        if self.dimension == 1:
            conv_ND = nn.Conv1d
            normType = nn.BatchNorm1d if is_batchnorm else nn.GroupNorm
        elif self.dimension == 2:
            conv_ND = nn.Conv2d
            normType = nn.BatchNorm3d if is_batchnorm else nn.GroupNorm
        elif self.dimension == 3:
            conv_ND = nn.Conv3d
            normType = nn.BatchNorm3d if is_batchnorm else nn.GroupNorm

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                conv_ND(self.num_channels[0], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                normType(self.num_channels[1]), 
                nn.ReLU(inplace=True) #inplace=True if using ReLU
            )
            self.conv2 = nn.Sequential(
                conv_ND(self.num_channels[1], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                normType(self.num_channels[1]), 
                nn.ReLU(inplace=True) #inplace=True if using ReLU
            )
        else:
            self.conv1 = nn.Sequential(
                conv_ND(self.num_channels[0], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                normType(num_groups, self.num_channels[1]),
                nn.ReLU(inplace=True) #inplace=True if using ReLU
            )
            self.conv2 = nn.Sequential(
                conv_ND(self.num_channels[1], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                normType(num_groups, self.num_channels[1]),
                nn.ReLU(inplace=True) #inplace=True if using ReLU
            )

    def forward(self, inputs):
        outputs = self.conv2(self.conv1(inputs))
        return outputs

class unetConv3d(pl.LightningModule):

    def __init__(self, in_channels, 
                 out_channels, is_batchnorm, 
                 kernel_size = (3,3,1), padding = (3,3,1), 
                 conv_type = 'vgg'):
        super(unetConv3d, self).__init__()
        self.convBlocks = []
        
        if conv_type == 'vgg':
            convBlock1 = convBlockVGG_ND(
                num_channels = [in_channels, out_channels],
                is_batchnorm = is_batchnorm,
                kernel_size = kernel_size, padding = padding,
                dimension = 3
            )
            self.convBlocks.append(convBlock1)

        self.convolution = nn.Sequential(*self.convBlocks)

    def forward(self, inputs):
        outputs = self.convolution(inputs)
        return outputs
    
class upsample3d(pl.LightningModule):

    def __init__(self, in_size, out_size):
        super(upsample3d, self).__init__()
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear')

    def forward(self, x):
        return self.up(x)

class pad3d(pl.LightningModule):
    def __init__(self):
        super(pad3d, self).__init__()
    
    def forward(self, leftIn, rightIn):
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[4]-rShape[4], 0, lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)
        pad = torch.nn.ConstantPad3d(padding, 0)
        rightIn = pad(rightIn)
        return leftIn, rightIn

class cat3d(pl.LightningModule):
    def __init__(self):
        super(cat3d, self).__init__()
    
    def forward(self, leftIn, rightIn):
        lrCat = torch.cat([leftIn, rightIn], 1)
        return lrCat

class padConcate3d(pl.LightningModule):
    def __init__(self):
        super(padConcate3d, self).__init__()

    def forward(self, leftIn, rightIn):
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[4]-rShape[4], 0, lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)
        pad = torch.nn.ConstantPad3d(padding, 0)
        rightIn = pad(rightIn)
        lrCat = torch.cat([leftIn, rightIn], 1)
        return lrCat 
        
class unetUp3d(pl.LightningModule):

    def __init__(self, in_size, out_size, 
                 is_batchnorm, kernel_size = (3,3,1), padding = (1,1,0)):
        
        super(unetUp3d, self).__init__()
        
        self.conv = unetConv3d(in_size, out_size, is_batchnorm, kernel_size, padding)
        self.up = nn.Upsample(scale_factor = (2,2,2), mode = 'nearest')

    def forward(self, leftIn, rightIn):
        rightIn = self.up(rightIn)
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[4]-rShape[4], 0, lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)
        
        pad = torch.nn.ConstantPad3d(padding, 0)
        rightIn = pad(rightIn)

        lrCat = torch.cat([leftIn, rightIn], 1).type_as(leftIn)
        output = self.conv(lrCat)
        return output