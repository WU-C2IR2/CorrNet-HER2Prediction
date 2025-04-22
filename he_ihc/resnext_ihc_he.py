#### KAUSHIK DUTTA ####
#### VisionTransformer for classification - Using Training from Scratch for IHC and H&E Combined #####
##### This code Block along with implementing a ViT also inplements a Confidence Loss Function which gives Confidence Level for our prediction

#### Importing the necessary packages
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl
import timm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
import contextlib
import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import Accuracy

##################################################################################### DATASET ###################################################################################################

# Load IHC images
with open('//ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/ihc_only/ihc_train_data.pkl', 'rb') as f:
    ihc_images = pickle.load(f)

# Load H&E images
with open('//ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/ihc_only/he_train_data.pkl', 'rb') as f:
    he_images = pickle.load(f)

# Load the labels
with open('//ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/ihc_only/labels_train.pkl', 'rb') as f:
    labels_list = pickle.load(f)

# Encode labels and convert to one-hot
label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(labels_list)
one_hot_targets = torch.nn.functional.one_hot(torch.from_numpy(targets), num_classes=4)

#####################################################################################
# TRANSFORMS
#####################################################################################

# Training data augmentation and resizing
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(1024, scale=(0.8, 1.0)),  # Assuming images are 1024x1024
    transforms.Resize((256, 256))
])

# Validation resizing
val_transforms = transforms.Compose([
    transforms.Resize((256, 256))
])

#####################################################################################
# DATASET AND DATALOADER
#####################################################################################

class CustomImageDataset(Dataset):
    def __init__(self, ihc_images, he_images, targets, transform):
        self.ihc_images = ihc_images
        self.he_images = he_images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.ihc_images)

    def __getitem__(self, idx):
        # Load IHC image and H&E image
        ihc_image = torch.from_numpy(self.ihc_images[idx]).float()
        he_image = torch.from_numpy(self.he_images[idx]).float()
        
        # Concatenate along the channel dimension (IHC has 3 channels, H&E has 3 channels)
        combined_image = torch.cat((ihc_image, he_image), dim=0)  # Final shape: (6, H, W)
        
        # Apply transforms if any
        if self.transform:
            combined_image = self.transform(combined_image)

        target = self.targets[idx]
        return combined_image, target
        
# Split data into training and validation sets
X_train_ihc, X_test_ihc, y_train, y_test = train_test_split(ihc_images, one_hot_targets, test_size=0.2, random_state=42)
X_train_he, X_test_he = train_test_split(he_images, test_size=0.2, random_state=42)

# Training dataset and dataloader
train_dataset = CustomImageDataset(X_train_ihc, X_train_he, y_train, transform=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Validation dataset and dataloader
validation_dataset = CustomImageDataset(X_test_ihc, X_test_he, y_test, transform=val_transforms)
validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

##################################################################################### MODEL ARCHITECTURE ################################################################################
class ResNeXtClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, learning_rate=0.0001):
        super(ResNeXtClassifier, self).__init__()
        # Initialize ResNeXt from scratch
        self.model = timm.create_model('resnext50_32x4d', pretrained=False, num_classes=num_classes, in_chans=6)
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred.float(), y.float())
        acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        confidence_scores = torch.max(torch.softmax(y_pred, dim=1), dim = 1)[0]
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=True, logger=True)
        self.log('train_confidence', confidence_scores.mean(), prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred.float(), y.float())
        acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        confidence_scores = torch.max(torch.softmax(y_pred, dim=1), dim = 1)[0]
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_confidence', confidence_scores.mean(), prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        return [optimizer], [scheduler]

#################################################################################### RUNNING THE NETWORK (TRAINING) #################################################################
### Callbacks and Trainer
checkpoint_callback = ModelCheckpoint(
    save_top_k=30,
    monitor="val_acc",
    mode="max",
    dirpath="/ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/Classification/he_ihc/saved_weights/ResNext_Scratch/",
    filename="resnext_normal_ce-{epoch:02d}-{val_loss:.2f}"
)

# Initialize the model and trainer
max_epochs = 150
resnext_model = ResNeXtClassifier(num_classes=4, learning_rate=0.0001)

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator='gpu',
    devices=2,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=10,
    strategy='ddp_find_unused_parameters_true'
)

trainer.fit(resnext_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
