###### Kaushik Dutta
###### Implementation of Correlational AutoEncoder (No Attention Module included)

######## Importing the necessary packages 
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import torch
import pytorch_lightning as pl
from monai.data import PersistentDataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, ToTensord, RandRotated, RandZoomd, RandFlipd, RandGaussianSmoothd, RandAdjustContrastd, RandGaussianSharpend, RandLambdaD, ResizeD
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from data_corr_encoder import train_data_loading
#from model_corr_encoder_1 import Encoder, Decoder, CorrNet_3Layer
from model_corr_encoder_noattention import CorrNet_3Layer
from loss_function import CalculateLoss
import argparse

parser = argparse.ArgumentParser(description="Correlational AutoEncoder Training")
parser.add_argument("--hidden_layer", type=int, default=16, help="Size of the hidden layer")
args = parser.parse_args()

######## Load the data
he_data, ihc_data = train_data_loading()
print('Shape of IHC List==',len(ihc_data))
print('Data Loading Done')
####### Convert the list to dictionary
train_data = [{"he": he, "ihc": ihc} for he, ihc in zip(he_data, ihc_data)]
train_transforms = Compose([
    ToTensord(keys=["he", "ihc"]),
    RandGaussianSmoothd(keys=["he", "ihc"], sigma_x=(0.5, 0.85), sigma_y=(0.5, 0.85), prob = 0.35),
    RandAdjustContrastd(keys=["he", "ihc"], prob = 0.45, gamma=(0.7, 1.3)),
    RandGaussianSharpend(keys=["he", "ihc"], prob = 0.45),   
    RandLambdaD(
        keys=["ihc"],
        func=lambda x: torch.zeros_like(x),
        prob=0.5
    ),  # Zero out 'ihc' with 50% probability
    ResizeD(keys=["he", "ihc"], spatial_size= (256, 256)),
])
train_dataset = PersistentDataset(data=train_data, transform=train_transforms, cache_dir="cache_train")
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
print('Persistent DataLoader Done')

######################### TRAINING BLOCK ###############################        
#### Training Loop using Pytorch Lightning
class Training_CorrNet(pl.LightningModule):
    def __init__(self, model, learning_rate, max_epochs):
        super(Training_CorrNet, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.criterion = CalculateLoss(loss_terms = '1111', lambda_param = 1)

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        he, ihc = batch["he"], batch["ihc"]
        #z_recon, hz, x_att, y_att = self(he, ihc)
        ##For autoencoder
        z_recon, x, y, hidden = self(he, ihc)
        loss,l1,l2,l3,l4 = self.criterion(z_recon, x, y, he, ihc)
        
#        z_recon_he = z_recon[0, :, :]  
#        z_recon_ihc = z_recon[1, :, :]  
#        correlation_he = F.cosine_similarity(z_recon_he.flatten(1), he.flatten(1)).mean()
#        correlation_ihc = F.cosine_similarity(z_recon_ihc.flatten(1), ihc.flatten(1)).mean()
        
        #self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log_dict({'Train_Loss_Total': loss, 'Self_Recon': l1, 'Cross_Recon': l2+l3, 'Correlation_Loss': l4}, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'Train_Loss_Total'}   
        
##### Final Training
#### Define the network parameters

max_epochs = 100
lambda_corr = 1.5
hidden_layer = args.hidden_layer

model = CorrNet_3Layer(3,3,hidden_layer//4,hidden_layer//2,hidden_layer, activation_function1= nn.LeakyReLU())
# Define the learning rate
learning_rate = 0.0001
# Instantiate the Lightning module
lightning_model = Training_CorrNet(model, learning_rate, max_epochs)
# Create a PyTorch Lightning trainer
trainer = pl.Trainer(max_epochs=max_epochs, accelerator = 'gpu', devices = 2, strategy='ddp_find_unused_parameters_true', enable_checkpointing=True)
# Train the model
trainer.fit(lightning_model, train_loader)        
