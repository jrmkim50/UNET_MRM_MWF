# Code from: /disk/hz459/coding/ms_lesion_segmentatation/lesion_seg/

import torch.nn as nn

from Hang.utils_u_1 import unetConv3d, unetUp3d


class unet3d(nn.Module):
    def __init__(self, 
        feature_scale = 2, 
        n_classes = 5, 
        is_deconv = True, 
        in_channels = 9, 
        is_batchnorm = True,
        is_hpool = True,
    ):
        super(unet3d, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv3d(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size = 2, padding = 1)

        self.conv2 = unetConv3d(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size = 2, padding = 1)

        self.conv3 = unetConv3d(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size = 2, padding = 1)

        self.conv4 = unetConv3d(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size = 2, padding = 1)

        self.center = unetConv3d(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp3d(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3d(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs):
        
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final