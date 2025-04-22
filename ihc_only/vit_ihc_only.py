#### KAUSHIK DUTTA ####
#### VisionTransformer for classification - Using Training from Scratch for IHC Only #####
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import Accuracy

##################################################################################### DATASET ###################################################################################################

#### Loading the Training, Validation and Testing Data 
with open('//ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/ihc_only/ihc_train_data.pkl', 'rb') as f:
    images = pickle.load(f)

### Loading the Training Labels for now
with open('//ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/ihc_only/labels_train.pkl', 'rb') as f:
    labels_list = pickle.load(f)
    
### Dataset and DataLoader Creation
label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(labels_list)
one_hot_targets = torch.nn.functional.one_hot(torch.from_numpy(targets), num_classes=4)

##### Transforms added to the train and validation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(1024, scale=(0.8, 1.0)),  # Assuming images are 1024x1024
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #transforms.ToTensor(),  # Convert numpy array to tensor
    transforms.Resize((256, 256)),
    #transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for one channel, adjust for your data
])
 
# Validation might not require augmentations, only normalization
val_transforms = transforms.Compose([
    #transforms.ToTensor(),
    transforms.Resize((256, 256)),
    #transforms.Normalize(mean=[0.5], std=[0.5])
])

### Split the data into train and validation and create dataloaders
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
        #if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
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
batch_size = 32
## Training
train_dataset = CustomImageDataset(X_train, y_train, transform=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

## Validation 
validation_dataset = CustomImageDataset(X_test, y_test, transform=val_transforms)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


############################################################################################ LOSS FUNCTION ########################################################################################

#### Defining the Confidence Loss and the Focal Loss (Task Loss)
##### Confidence Loss
class ConfidenceLoss(nn.Module):
    def __init__(self, lambda_confidence, alpha, gamma = 2.0, beta = 0.5, reduction = 'mean'):
        super(ConfidenceLoss, self).__init__()
        self.lambda_confidence = lambda_confidence
        self.nll_loss = nn.NLLLoss()
        #self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
    
    def focal_loss(self, logits, targets):
        """
        Compute Focal Loss using alpha (class weights) and gamma.
        """
        # Apply softmax to the logits to get class probabilities
        predictions = F.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)

        # Convert targets to one-hot if not already
        if targets.ndim == 1:
            labels_onehot = F.one_hot(targets, num_classes=predictions.shape[1]).float()
        else:
            labels_onehot = targets

        self.alpha = self.alpha.to(logits.device)
        # Compute the focal loss
        p_t = torch.sum(labels_onehot * predictions, dim=1)  # Shape: (batch_size,)
        alpha_t = torch.sum(labels_onehot * self.alpha, dim=1)  # Shape: (batch_size,)
        
        focal_loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-12)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

    def forward(self, logits, confidence_logit, targets):
        """
        Args:
            logits: Prediction logits from the model of shape (batch_size, num_classes)
            confidence_logit: Confidence logit from the model of shape (batch_size, 1)
            targets: Ground truth labels, one-hot encoded of shape (batch_size, num_classes)
        
        Returns:
            Total loss: combination of task loss and confidence loss
        """
        # Apply softmax to the prediction logits to get class probabilities
        predictions = F.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)
        
        # Apply sigmoid to the confidence logit to get confidence estimate
        confidence = torch.sigmoid(confidence_logit).squeeze()  # Shape: (batch_size,)

        # Ensure targets is one-hot encoded
        # Step 2: Ensure targets are one-hot encoded
        
        if targets.ndim == 1 or targets.shape[1] != pred_original.shape[1]:
            labels_onehot = F.one_hot(targets, num_classes=predictions.shape[1]).float()
        else:
            labels_onehot = targets
        # Interpolate between the predictions and the target distribution
        # Step 3: Compute the confidence score (sigmoid for binary confidence)
        #confidence = torch.sigmoid(confidence_logit).squeeze()  # Shape: (batch_size,)
        
        # Step 4: Create Bernoulli variable `b`
        b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).cuda()  # Shape: (batch_size,)
        
        # Step 5: Compute `conf` based on the combination of `confidence` and `b`
        conf = confidence * b + (1 - b)  # Shape: (batch_size,)
        
        # Step 6: Compute `pred_new` by combining `pred_original` and `labels_onehot`
        pred_new = predictions * conf.unsqueeze(1).expand_as(predictions) + labels_onehot * (1 - conf.unsqueeze(1).expand_as(labels_onehot))
        
        # Step 7: Take the logarithm of `pred_new` for final loss computation
        pred_new = torch.log(pred_new + 1e-12)  # Add a small epsilon for numerical stability
        
        target_indices = torch.argmax(labels_onehot, dim=1)
        # Step 8: Calculate task loss (negative log-likelihood loss)
        #task_loss = self.nll_loss(pred_new, target_indices) 
        task_loss = self.focal_loss(pred_new, labels_onehot)

        # Calculate confidence loss (Binary Cross-Entropy where target is always 1)
        confidence_loss = -torch.log(confidence + 1e-12)  # Shape: (batch_size,)
        confidence_loss = confidence_loss.mean()  # Averaging over the batch

        # Adjust lambda_confidence based on beta
        if confidence_loss > self.beta:
            lambda_adjusted = self.lambda_confidence / 0.95
        else:
            lambda_adjusted = self.lambda_confidence / 1.05

        # Total loss is the weighted sum of task loss and confidence loss
        #total_loss = task_loss + self.lambda_confidence * confidence_loss
        total_loss = task_loss + lambda_adjusted  * confidence_loss
        return total_loss
                
##################################################################################### MODEL ARCHITECTURE ################################################################################

class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassifierHead, self).__init__()
        self.linear = nn.Linear(in_features, num_classes)
        #self.softmax = nn.Softmax(dim=1)  # Apply Softmax across classes

    def forward(self, x):
        logits = self.linear(x)        # Shape: (Batch_size, num_classes)
        #probs = self.softmax(logits)   # Shape: (Batch_size, num_classes)
        return logits

class ConfNet(nn.Module):
    def __init__(self, num_classes):
        super(ConfNet, self).__init__()
        self.fc = nn.Linear(num_classes, 1)  # Output a single confidence score
        
    def forward(self, x):
        confidence_logit = self.fc(x)  # Shape: [batch_size, 1]
        return confidence_logit

# Define your LightningModule
class VitClassifierWithConfNet(pl.LightningModule):
    def __init__(self, num_classes=4, learning_rate=0.0001, lambda_confidence = 0.5, alpha = None, gamma = 2.0, beta = 0.5):
        super(VitClassifierWithConfNet, self).__init__()
        
        # Initialize ViT model from scratch with a valid model name
        self.vit = timm.create_model(
            'vit_mediumd_patch16_reg4_gap_256',  # Adjusted model name for timm
            pretrained=False,
            num_classes=0,           # Remove existing classification head
            in_chans=3
        )
        
        self.learning_rate = learning_rate
        self.lambda_confidence = lambda_confidence
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        
        # Modified Classification Head with Softmax
        self.classifier = ClassifierHead(in_features=self.vit.num_features, num_classes=num_classes)
        
        # Confidence Network
        self.confidence_net = ConfNet(num_classes=self.vit.num_features)
        
        # Confidence Loss function with focal loss (assuming you have this implemented)
        self.confidence_loss_fn = ConfidenceLoss(lambda_confidence, alpha, gamma, beta, reduction='mean')
        
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
        
        # Apply Global Average Pooling if features have more than 2 dimensions
        if features.dim() > 2:
            features = features.mean(dim=1)  # Shape: [batch_size, num_features]
        
        # Classification probabilities
        class_probs = self.classifier(features)  # Shape: [batch_size, num_classes]
        
        # Confidence score
        confidence_score = self.confidence_net(features)  # Shape: [batch_size, 1]
        
        return class_probs, confidence_score
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        if y.dim() > 1 and y.size(1) > 1:
            y = torch.argmax(y, dim=1)  # Handle one-hot encoded labels
        else:
            y = y.view(-1)  # Ensure y is of shape [batch_size]

        # Forward pass
        class_probs, conf_score = self(x)
        
        # Compute Confidence Loss
        loss = self.confidence_loss_fn(class_probs, conf_score, y)
        
        # Compute accuracy
        preds = torch.argmax(class_probs, dim=1)
        acc = self.train_accuracy(preds, y)
        
        # Log losses and accuracy
        self.log('train_loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        if y.dim() > 1 and y.size(1) > 1:
            y = torch.argmax(y, dim=1)  # Handle one-hot encoded labels
        else:
            y = y.view(-1)  # Ensure y is of shape [batch_size]
        
        # Forward pass
        class_probs, conf_score = self(x)
        
        # Compute Confidence Loss
        loss = self.confidence_loss_fn(class_probs, conf_score, y)
        
        # Compute accuracy
        preds = torch.argmax(class_probs, dim=1)
        acc = self.val_accuracy(preds, y)

        # Determine correctness: 1 for correct, -1 for incorrect
        correctness = (preds == y).float() * 2 - 1  # 1 if correct, -1 if incorrect

        # Collect confidence scores and correctness
        self.validation_conf_scores.append(conf_score.detach())
        self.validation_correctness.append(correctness.detach())
        
        # Log losses and accuracy
        self.log('val_loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def on_validation_epoch_end(self):
        """
        Computes the Mean Effective Confidence (MEC) at the end of the validation epoch.
        """
        # Concatenate all collected confidence scores and correctness
        conf_scores = torch.cat(self.validation_conf_scores, dim=0)  # Shape: [n_samples, 1]
        correctness = torch.cat(self.validation_correctness, dim=0)  # Shape: [n_samples]

        # Normalize CSi: (CSi - min) / (max - min)
        min_conf = conf_scores.min()
        max_conf = conf_scores.max()
        if max_conf - min_conf > 1e-6:
            norm_conf_scores = (conf_scores - min_conf) / (max_conf - min_conf)
        else:
            # Avoid division by zero; set all normalized scores to 0
            norm_conf_scores = torch.zeros_like(conf_scores)

        # Flatten norm_conf_scores to shape [n_samples]
        norm_csi = norm_conf_scores.squeeze(1)  # Shape: [n_samples]

        # Compute MEC
        ci = correctness  # Shape: [n_samples]
        mec = torch.mean(ci * norm_csi)  # Scalar

        # Log MEC
        self.log('val_mec', mec, on_epoch=True, prog_bar=True, logger=True)

        # Reset the lists for the next epoch
        self.validation_conf_scores = []
        self.validation_correctness = []
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
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
base_log_dir = Path("/ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/Classification/ihc_only")
# Define TensorBoard log directory and Checkpoint directory
tb_log_dir = base_log_dir / "tb_logs" / "ViT_Scratch"
checkpoint_dir = base_log_dir / "saved_weights" / "ViT_Scratch"

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
    save_top_k=5,
    dirpath=checkpoint_dir,
    filename="vit_confnet-{epoch:02d}-{val_acc:.2f}"
)

# Initialize TensorBoard Logger
#tb_logger = TensorBoardLogger("tb_logs", name="vit_with_confnet")

# Initialize the model and trainer
max_epochs = 100
vit_confnet_model = VitClassifierWithConfNet(num_classes=4, learning_rate=0.0001, lambda_confidence = 0.75, alpha = class_weights_tensor, gamma = 2, beta = 0.3)

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

