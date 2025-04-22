###### Kaushik Dutta
###### Model definition with self attention block

### Importing the necessary packages
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import torch
import pytorch_lightning as pl
from monai.data import PersistentDataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, ToTensord, RandRotated, RandZoomd, RandFlipd, RandGaussianSmoothd, RandAdjustContrastd, RandGaussianSharpend
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint

### Model Architecture
### Model Architecture with Batch Normalization
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeezed = self.global_avg_pool(x).view(batch_size, channels)
        excitation = self.fc2(self.fc1(squeezed))
        excitation = self.sigmoid(excitation).view(batch_size, channels, 1, 1)
        return x * excitation


class CorrNet_3Layer(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels1, hidden_channels2, hidden_channels3, activation_function1):
        super(CorrNet_3Layer, self).__init__()
        
        #### Encoding for x
        self.conv_x1 = nn.Conv2d(input_channels, hidden_channels1, kernel_size=3, padding=1)
        self.bn_x1 = nn.BatchNorm2d(hidden_channels1)  # Batch Norm layer
        self.activation_x1 = activation_function1
        
        self.conv_x2 = nn.Conv2d(hidden_channels1, hidden_channels2, kernel_size=3, padding=1)
        self.bn_x2 = nn.BatchNorm2d(hidden_channels2)  # Batch Norm layer
        self.activation_x2 = activation_function1
        
        self.conv_x3 = nn.Conv2d(hidden_channels2, hidden_channels3, kernel_size=3, padding=1)
        self.bn_x3 = nn.BatchNorm2d(hidden_channels3)  # Batch Norm layer
        self.activation_x3 = activation_function1
        
        #### Encoding for y
        self.conv_y1 = nn.Conv2d(input_channels, hidden_channels1, kernel_size=3, padding=1)
        self.bn_y1 = nn.BatchNorm2d(hidden_channels1)  # Batch Norm layer
        self.activation_y1 = activation_function1
        
        self.conv_y2 = nn.Conv2d(hidden_channels1, hidden_channels2, kernel_size=3, padding=1)
        self.bn_y2 = nn.BatchNorm2d(hidden_channels2)  # Batch Norm layer
        self.activation_y2 = activation_function1
        
        self.conv_y3 = nn.Conv2d(hidden_channels2, hidden_channels3, kernel_size=3, padding=1)
        self.bn_y3 = nn.BatchNorm2d(hidden_channels3)  # Batch Norm layer
        self.activation_y3 = activation_function1
        
        ### Define SE blocks for attention
        self.se_x3 = SEBlock(hidden_channels3)
        self.se_y3 = SEBlock(hidden_channels3)
        
        ### Concat and Fuse Attention Maps
        self.concat_conv = nn.Conv2d(hidden_channels3, hidden_channels3, kernel_size=3, padding=1)
        
        #### Decoding for x and y
        self.conv_out_x1 = nn.Conv2d(hidden_channels3, hidden_channels2, kernel_size=3, padding=1)
        self.bn_out_x1 = nn.BatchNorm2d(hidden_channels2)  # Batch Norm layer
        
        self.conv_out_y1 = nn.Conv2d(hidden_channels3, hidden_channels2, kernel_size=3, padding=1)
        self.bn_out_y1 = nn.BatchNorm2d(hidden_channels2)  # Batch Norm layer
        
        self.conv_out_x2 = nn.Conv2d(hidden_channels2, hidden_channels1, kernel_size=3, padding=1)
        self.bn_out_x2 = nn.BatchNorm2d(hidden_channels1)  # Batch Norm layer
        
        self.conv_out_y2 = nn.Conv2d(hidden_channels2, hidden_channels1, kernel_size=3, padding=1)
        self.bn_out_y2 = nn.BatchNorm2d(hidden_channels1)  # Batch Norm layer
        
        self.conv_out_x3 = nn.Conv2d(hidden_channels1, output_channels, kernel_size=3, padding=1)
        self.conv_out_y3 = nn.Conv2d(hidden_channels1, output_channels, kernel_size=3, padding=1)

    def forward(self, x, y):
    
        #### Encoder ####
        x = self.conv_x1(x)
        x = self.bn_x1(x)
        x = self.activation_x1(x)
        
        x = self.conv_x2(x)
        x = self.bn_x2(x)
        x = self.activation_x2(x)
        
        x = self.conv_x3(x)
        x = self.bn_x3(x)
        x = self.activation_x3(x)
        x_att = self.se_x3(x)
        
        y = self.conv_y1(y)
        y = self.bn_y1(y)
        y = self.activation_y1(y)
        
        y = self.conv_y2(y)
        y = self.bn_y2(y)
        y = self.activation_y2(y)
        
        y = self.conv_y3(y)
        y = self.bn_y3(y)
        y = self.activation_y3(y)
        y_att = self.se_y3(y)
        
        #### Hidden Layer ####
        fused_concat = (x_att + y_att) / 2
        fused = self.concat_conv(fused_concat)
        
        #### Decoder ####
        hz = self.conv_out_x1(fused)
        hz = self.bn_out_x1(hz)
        hz = self.activation_x1(hz)
        
        hz = self.conv_out_x2(hz)
        hz = self.bn_out_x2(hz)
        hz = self.activation_x1(hz)
        
        hz = self.conv_out_x3(hz)
        hz = self.activation_x1(hz)
        
        hz_concat = torch.cat((hz, hz), dim=1)
        
        ### New Block without activation
        z_recon = hz_concat
       
        return z_recon, x_att, y_att, fused


