####### Calculate the Loss i.e. correlation and reconstruction

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import torch
import pytorch_lightning as pl
from monai.data import PersistentDataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, ToTensord, RandRotated, RandZoomd, RandFlipd, RandGaussianSmoothd, RandAdjustContrastd, RandGaussianSharpend
#import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint



class CorrelationCalculator(nn.Module):
    def __init__(self):
        super(CorrelationCalculator, self).__init__()

    def forward(self, x, y):
        """
        Calculate the correlation loss between two tensors.
        
        Args:
        x (torch.Tensor): A tensor of shape (batch_size, n).
        y (torch.Tensor): A tensor of shape (batch_size, n).
        
        Returns:
        torch.Tensor: The correlation loss.
        """
        # Subtract mean
        x_mean = x - torch.mean(x, dim=1, keepdim=True)
        y_mean = y - torch.mean(y, dim=1, keepdim=True)
        
        # Calculate the numerator (covariance)
        numerator = torch.sum(x_mean * y_mean, dim=1)
        
        # Calculate the denominator (standard deviations multiplied)
        x_std = torch.sqrt(torch.sum(x_mean ** 2, dim=1))
        y_std = torch.sqrt(torch.sum(y_mean ** 2, dim=1))
        denominator = x_std * y_std
        
        # Calculate the correlation coefficient for each sample in the batch
        correlation = numerator / denominator
        
        # Calculate the mean correlation across the batch
        mean_correlation = torch.mean(correlation)
        
        return mean_correlation
        
               
######### Defining the class to calculate the correlation
class CalculateLoss(nn.Module):
    def __init__(self, loss_terms, lambda_param):
        super(CalculateLoss, self).__init__()
        self.loss_terms = loss_terms
        self.lambda_param = lambda_param
        self.mae_loss = nn.L1Loss()
        self.corr_loss = CorrelationCalculator()

    def forward(self, z_recon, x_attention, y_attention,inputs_1, inputs_2):
        if len(self.loss_terms) == 0:
            raise ValueError("Must pass at least one loss term of l1, l2, l3, l4")
        inputs = torch.cat((inputs_1, inputs_2), axis = 1)
        inputs_1_only = torch.cat((inputs_1, torch.zeros_like(inputs_2)), axis = 1)
        inputs_2_only = torch.cat((torch.zeros_like(inputs_1), inputs_2), axis = 1)

        ### Sorting out the Loss terms
        l1 = self.mae_loss(z_recon, inputs)
        l2 = self.mae_loss(z_recon, inputs_1_only)
        l3 = self.mae_loss(z_recon, inputs_2_only)
        l4 = self.corr_loss(x_attention, y_attention)

        if self.loss_terms == '1111':
            total_loss = l1+l2+l3-self.lambda_param*l4
        elif self.loss_terms == '1110':
            total_loss = l1+l2+l3
        elif self.loss_terms == '1100':
            total_loss = l1+l2
        elif self.loss_terms == '1010':
            total_loss = l1+l3
        elif self.loss_terms == '1000':
            total_loss = l1
        else:
            print('Enter Proper Code for Calculating the Loss')
        return total_loss,l1,l2,l3,l4