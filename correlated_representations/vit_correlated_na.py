#### KAUSHIK DUTTA ####
#### EfficientNet for classification - Using Training from Scratch for IHC and H&E Combined #####
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
import argparse

parser = argparse.ArgumentParser(description="Hidden Layer Classifier Training")
parser.add_argument("--hidden_layer", type=int, default=16, help="Size of the hidden layer")
args = parser.parse_args()

##################################################################################### DATASET ###################################################################################################
with open('/ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/Classification/correlated_representations/z_recon/no_attention/hz_train_'+ str(args.hidden_layer) +'_na.pkl', 'rb') as f:
    images = pickle.load(f)

### Loading the Training Labels for now
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
    #transforms.RandomResizedCrop(1024, scale=(0.8, 1.0)),  # Assuming images are 1024x1024
    transforms.Resize((256, 256))
])

# Validation resizing
val_transforms = transforms.Compose([
    transforms.Resize((256, 256))
])

### To pre-process the Datasets
class CustomImageDataset(Dataset):
    def __init__(self, images, targets, transform):
        self.images = images
        self.targets = targets
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        min_val = torch.min(image)
        max_val = torch.max(image)
        image = (image - min_val) / (max_val - min_val)

        #if isinstance(image, np.ndarray):
        #image = torch.from_numpy(image)
        # Squeeze the tensor to remove unnecessary dimensions
        image = torch.squeeze(image)
        target = self.targets[idx]
        # Apply the transformations if any
        if self.transform:
            image = self.transform(image)
        return image, target
### Train and Validation
X_train, X_test, y_train, y_test = train_test_split(images, one_hot_targets, test_size=0.2, random_state=42)
### Initialize the train and validation Dataloaders
batch_size = 4
## Training
train_dataset = CustomImageDataset(X_train, y_train, transform=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

## Validation 
validation_dataset = CustomImageDataset(X_test, y_test, transform=val_transforms)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

##################################################################################### MODEL ARCHITECTURE ################################################################################
class VitClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, learning_rate=0.0001, hidden_layer = 16):
        super(VitClassifier, self).__init__()
        # Initialize ResNeXt from scratch
        self.model = timm.create_model('vit_mediumd_patch16_reg4_gap_256', pretrained=False, num_classes=num_classes, in_chans=6)
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred.float(), y.float())
        acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred.float(), y.float())
        acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

### Callbacks and Trainer
checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="val_acc",
    mode="max",
    dirpath="/ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/Classification/correlated_representations/saved_weights/vit_na/new/",
    filename="vit_finetune_all_layer-{epoch:02d}-{val_loss:.2f}"+"__"+str(args.hidden_layer)
)        

# Initialize the model and trainer
max_epochs = 200

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator='gpu',
    devices=2,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=10,
    strategy='ddp_find_unused_parameters_true'
)
vit_model = VitClassifier(num_classes=4, learning_rate=0.0001, hidden_layer = args.hidden_layer)
trainer.fit(vit_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)