import torch
import pytorch_lightning as pl
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
import matplotlib as matp
import numpy as np
import scipy.io as sio
from Hang.utils_u_groupnorm_pytorchLightning import *
from utils import *
import array as arr
from Hang.unet3dPersonalGroupNorm_pytorchLightning import unet3dpp
from torch.utils import data
from numpy import zeros
import time as time
import pdb
import nibabel as nib
import torchvision.transforms
import random

random.seed(5)

class Dataset_Generator(Dataset):
        def __init__(self, signal_data, signal_label, patches_mask):
            self.data = (signal_data.astype(float))
            self.label = (signal_label)
            self.mask = patches_mask
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            signal_graph = self.data[idx]/(self.data[idx][0] + 1e-16)
            signal_label = self.label[idx]
            patches_mask = self.mask[idx]
            return (torch.tensor(signal_graph).float(), torch.tensor(signal_label).float(), torch.tensor(patches_mask).float())
        
class Dataset_Generator_VALID(Dataset):
        def __init__(self, signal_data, signal_label, mask, transform=None):
            self.data = (signal_data.astype(float))
            self.label = (signal_label)
            self.mask = mask
            self.transform = transform
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            signal_graph = self.data[idx]/(self.data[idx][0] + 1e-16)
            signal_label = self.label[idx]
            mask = self.mask[idx]
            if self.transform is not None:
                idx = random.randrange(0,len(self.transform))
                signal_graph, signal_label, mask = self.transform[idx]([torch.tensor(signal_graph), 
                                                                   torch.tensor(signal_label), 
                                                                   torch.tensor(mask)])
            return (torch.tensor(signal_graph).float(), torch.tensor(signal_label).float(), torch.tensor(mask).float())

class Rotate(object):

    def __init__(self, output_angle):
        self.output_angle = output_angle

    def __call__(self, sample):
        x, y, mask = sample
        if (self.output_angle == 90):
            x = x.transpose(1, 2).flip(1)
            y = y.transpose(1, 2).flip(1)
            mask = mask.transpose(1, 2).flip(1)
        elif (self.output_angle == 180):
            x = x.flip(1).flip(2)
            y = y.flip(1).flip(2)
            mask = mask.flip(1).flip(2)
        elif (self.output_angle == 270):
            x = x.transpose(1, 2).flip(2)
            y = y.transpose(1, 2).flip(2)
            mask = mask.transpose(1, 2).flip(2)

        return x, y, mask


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): This is a tuple of length 3. Desired output size. If int, square crop
            is made. Example output size is (100,100,100)
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        x, y, mask = sample
        top = np.random.randint(0, x.shape[1] - self.output_size[0])
        left = np.random.randint(0, x.shape[2] - self.output_size[1])
        back = np.random.randint(0, x.shape[3] - self.output_size[2])
        x = x[:,top: top + self.output_size[0], left: left + self.output_size[1], back: back + self.output_size[2]]
        y = y[:,top: top + self.output_size[0], left: left + self.output_size[1], back: back + self.output_size[2]]
        mask = mask[:,top: top + self.output_size[0], left: left + self.output_size[1], back: back + self.output_size[2]]

        return x, y, mask

class Rescale(object):
    def __init__(self, scale_size):
        self.scale_size = scale_size
    def __call__(self, sample):
        x, y, mask = sample
        up = nn.Upsample(size=self.scale_size)
        x = up(x[None])[0]
        y = up(y[None])[0]
        mask = up(mask[None])[0]
        return x, y, mask
    
