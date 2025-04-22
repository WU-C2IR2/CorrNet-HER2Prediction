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



############################################################################################ LOSS FUNCTION ########################################################################################

#### Defining the Confidence Loss and the Focal Loss (Task Loss)
##### Confidence Loss
class CustomLossWithConfidence(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, class_weights = None):
        super(CustomLossWithConfidence, self).__init__()
        self.alpha = alpha  # Reward coefficient
        self.beta = beta    # Penalty coefficient
        self.class_weights = class_weights
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.nll_loss = nn.NLLLoss(reduction='none')

    def forward(self, predictions, targets):
        # predictions: raw logits (batch_size x num_classes)
        # targets: one-hot encoded labels (batch_size x num_classes)
        # Get softmax probabilities
        softmax_probs = predictions
        confidence_scores = torch.max(torch.softmax(predictions, dim=1), dim = 1)[0]
        # Predicted class (the one with the highest softmax probability)
        predicted_class = torch.argmax(softmax_probs, dim=1)
        # Correct class based on one-hot encoded targets
        correct_class = torch.argmax(targets, dim=1)
        
        # Standard Cross-Entropy Loss
        ce_loss = self.cross_entropy_loss(softmax_probs, correct_class)
        custom_loss = ce_loss 
        return custom_loss, confidence_scores
                
##################################################################################### MODEL ARCHITECTURE ################################################################################

class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassifierHead, self).__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        logits = self.linear(x)        # Shape: (Batch_size, num_classes)
        return logits

# Define your LightningModule
class VitClassifierWithConfNet(pl.LightningModule):
    def __init__(self, num_classes=4, learning_rate=0.0001, alpha = 1.0, beta = 1.0, class_weights = None):
        super(VitClassifierWithConfNet, self).__init__()
        
        # Initialize ViT model from scratch with a valid model name
        self.vit = timm.create_model(
            'vit_mediumd_patch16_reg4_gap_256',  # Ensure this model exists in timm
            pretrained=False,
            num_classes=0,           # Remove existing classification head
            in_chans=6
        )
        
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights
        
        # Modified Classification Head with Softmax
        self.classifier = ClassifierHead(in_features=self.vit.num_features, num_classes=num_classes)
        
        self.confidence_loss_fn = CustomLossWithConfidence(alpha=self.alpha, beta = self.beta, class_weights = self.class_weights)
        
        # Accuracy Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.validation_conf_scores = []
        self.validation_correctness = []
        # Log hyperparameters
        self.save_hyperparameters()
    
    def forward(self, x):
        # Extract features using ViT
        features = self.vit.forward_features(x)  # Shape: [batch_size, num_features, ...]
        #print(features.shape)
        
        # Apply Global Average Pooling if features have more than 2 dimensions
        if features.dim() > 2:
            features = features.mean(dim=1)  # Shape: [batch_size, num_features]
        
        # Classification probabilities
        class_probs = self.classifier(features)  # Shape: [batch_size, num_classes]
        # Confidence score
        #confidence_score = self.confidence_net(features)  # Shape: [batch_size, 1]
        
        return class_probs
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        # if y.dim() > 1 and y.size(1) > 1:
        #     y = torch.argmax(y, dim=1)
        #     #print(f"Converted y shape: {y.shape}")  # Should be torch.Size([batch_size])
        # else:
        #     y = y.view(-1)
        #     #print(f"Reshaped y shape: {y.shape}")    # Ensure it's torch.Size([batch_size])

        # Forward pass
        class_probs = self(x)
        
        # Compute Confidence Loss
        loss, conf_score = self.confidence_loss_fn(class_probs, y)
        
        # Compute accuracy
        preds = torch.argmax(class_probs, dim=1)
        y_target = torch.argmax(y, dim=1)
        acc = self.train_accuracy(preds, y_target)
        
        # Log losses and accuracy
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train Confidence', conf_score.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Confidence STD', conf_score.std(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        class_probs = self(x)

        # if y.dim() > 1 and y.size(1) > 1:
        #     y = torch.argmax(y, dim=1)
        #     #print(f"Converted y shape: {y.shape}")  # Should be torch.Size([batch_size])
        # else:
        #     y = y.view(-1)
        #     #print(f"Reshaped y shape: {y.shape}")    # Ensure it's torch.Size([batch_size])

        
        # Compute Confidence Loss
        loss, conf_score = self.confidence_loss_fn(class_probs, y)
        
        # Compute accuracy
        preds = torch.argmax(class_probs, dim=1)
        y_target = torch.argmax(y, dim=1)
        acc = self.val_accuracy(preds, y_target)
        
        # Log losses and accuracy
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Confidence Mean', conf_score.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Confidence STD', conf_score.std(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        #scheduler = CosineAnnealingLR(optimizer, T_max=10)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        return [optimizer], [scheduler]
        
#################################################################################### RUNNING THE NETWORK (TRAINING) #################################################################

# Convert one-hot encoded y_train to class indices
y_train_indices = torch.argmax(y_train, dim=1).numpy()
# Now you can use np.bincount
class_counts = np.bincount(y_train_indices)
class_weights = 1.0 / class_counts  # Inverse of the frequency
class_weights = class_weights / class_weights.sum()  # Normalize
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)


# Define the base log directory locally
base_log_dir = Path("/ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/Classification/he_ihc")
# Define TensorBoard log directory and Checkpoint directory
tb_log_dir = base_log_dir / "tb_logs" / "ViT_Scratch"
checkpoint_dir = base_log_dir / "saved_weights" / "ViT_Scratch/new/"

# Create directories if they don't exist
tb_log_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Initialize TensorBoard Logger
tb_logger = TensorBoardLogger(
    save_dir=tb_log_dir.parent,  # "tb_logs"
    name=tb_log_dir.name          # "ViT_Scratch"
)

# Initialize ModelCheckpoint Callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_top_k=30,
    dirpath=checkpoint_dir,
    filename="vit_confnet_normal_ce-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}"
)

# Initialize TensorBoard Logger
#tb_logger = TensorBoardLogger("tb_logs", name="vit_with_confnet")

# Initialize the model and trainer
max_epochs = 200
vit_confnet_model = VitClassifierWithConfNet(num_classes=4, learning_rate=0.001, alpha = 0.5, beta = 0.5, class_weights = class_weights_tensor)

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator='gpu',
    devices=2,
    callbacks=[checkpoint_callback],
    logger = tb_logger,
    check_val_every_n_epoch=5,
    strategy='ddp_find_unused_parameters_true'  # Ensure parameters are correctly tracked
)

# Start Training
trainer.fit(vit_confnet_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

