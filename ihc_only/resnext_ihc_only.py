#### KAUSHIK DUTTA ####
#### ResNeXt for classification - Training from Scratch for IHC Only #####
#### This code block implements a ResNeXt model and a Confidence Loss Function for the prediction.

#### Importing the necessary packages
import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl
import timm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import Accuracy
from pathlib import Path

##################################################################################### DATASET ###################################################################################################

#### Loading the Training, Validation and Testing Data 
with open('//ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/ihc_only/ihc_train_data.pkl', 'rb') as f:
    images = pickle.load(f)

### Loading the Training Labels
with open('//ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/ihc_only/labels_train.pkl', 'rb') as f:
    labels_list = pickle.load(f)
    
### Dataset and DataLoader Creation
label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(labels_list)
one_hot_targets = torch.nn.functional.one_hot(torch.from_numpy(targets), num_classes=4)

##### Data transformations for training and validation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  
    transforms.Resize((256, 256)),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
])

### Custom Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, images, targets, transform):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        image = torch.squeeze(image)
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        return image, target

### Train and Validation data split
X_train, X_test, y_train, y_test = train_test_split(images, one_hot_targets, test_size=0.2, random_state=42)

### Initialize DataLoaders
batch_size = 32

train_dataset = CustomImageDataset(X_train, y_train, transform=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = CustomImageDataset(X_test, y_test, transform=val_transforms)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

############################################################################################ LOSS FUNCTION ########################################################################################

#### Defining the Confidence Loss with Focal Loss
class ConfidenceLoss(nn.Module):
    def __init__(self, lambda_confidence, alpha, gamma=2.0, beta=0.5, reduction='mean'):
        super(ConfidenceLoss, self).__init__()
        self.lambda_confidence = lambda_confidence
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def focal_loss(self, logits, targets):
        predictions = F.softmax(logits, dim=1)
        if targets.ndim == 1:
            labels_onehot = F.one_hot(targets, num_classes=predictions.shape[1]).float()
        else:
            labels_onehot = targets
        self.alpha = self.alpha.to(logits.device)
        p_t = torch.sum(labels_onehot * predictions, dim=1)
        alpha_t = torch.sum(labels_onehot * self.alpha, dim=1)
        focal_loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-12)

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

    def forward(self, logits, confidence_logit, targets):
        predictions = F.softmax(logits, dim=1)
        confidence = torch.sigmoid(confidence_logit).squeeze()

        if targets.ndim == 1:
            labels_onehot = F.one_hot(targets, num_classes=predictions.shape[1]).float()
        else:
            labels_onehot = targets

        b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).cuda()
        conf = confidence * b + (1 - b)
        pred_new = predictions * conf.unsqueeze(1).expand_as(predictions) + labels_onehot * (1 - conf.unsqueeze(1).expand_as(labels_onehot))
        pred_new = torch.log(pred_new + 1e-12)
        
        task_loss = self.focal_loss(pred_new, labels_onehot)

        confidence_loss = -torch.log(confidence + 1e-12).mean()

        lambda_adjusted = self.lambda_confidence / 0.95 if confidence_loss > self.beta else self.lambda_confidence / 1.05

        return task_loss + lambda_adjusted * confidence_loss
                
##################################################################################### MODEL ARCHITECTURE ################################################################################

class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassifierHead, self).__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.linear(x)
        probs = self.softmax(logits)
        return probs

class ConfNet(nn.Module):
    def __init__(self, num_classes):
        super(ConfNet, self).__init__()
        self.fc = nn.Linear(num_classes, 1)

    def forward(self, x):
        return self.fc(x)

# Define the LightningModule
class ResNeXtClassifierWithConfNet(pl.LightningModule):
    def __init__(self, num_classes=4, learning_rate=0.0001, lambda_confidence=0.5, alpha=None, gamma=2.0, beta=0.5):
        super(ResNeXtClassifierWithConfNet, self).__init__()

        # Initialize ResNeXt model
        self.resnext = timm.create_model(
            'resnext50_32x4d',  # ResNeXt model
            pretrained=False,
            num_classes=0,  # Set num_classes to 0 for feature extraction
            in_chans=3
        )
        
        self.learning_rate = learning_rate
        self.lambda_confidence = lambda_confidence
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

        self.classifier = ClassifierHead(in_features=self.resnext.num_features, num_classes=num_classes)
        self.confidence_net = ConfNet(num_classes=num_classes)

        self.confidence_loss_fn = ConfidenceLoss(lambda_confidence, alpha, gamma, beta, reduction='mean')
        
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.validation_conf_scores = []
        self.validation_correctness = []

    def forward(self, x):
        features = self.resnext.forward_features(x)
        if features.dim() > 2:
            features = features.mean(dim=1)

        class_probs = self.classifier(features)
        confidence_score = self.confidence_net(class_probs)

        return class_probs, confidence_score

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1) if y.dim() > 1 else y.view(-1)

        class_probs, conf_score = self(x)
        loss = self.confidence_loss_fn(class_probs, conf_score, y)

        preds = torch.argmax(class_probs, dim=1)
        acc = self.train_accuracy(preds, y)

        self.log('train_loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1) if y.dim() > 1 else y.view(-1)

        class_probs, conf_score = self(x)
        loss = self.confidence_loss_fn(class_probs, conf_score, y)

        preds = torch.argmax(class_probs, dim=1)
        acc = self.val_accuracy(preds, y)

        correctness = (preds == y).float() * 2 - 1
        self.validation_conf_scores.append(conf_score.detach())
        self.validation_correctness.append(correctness.detach())

        self.log('val_loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self):
        if self.validation_conf_scores:
            avg_conf = torch.cat(self.validation_conf_scores).mean().item()
            avg_correct = torch.cat(self.validation_correctness).mean().item()
            self.log('avg_confidence', avg_conf)
            self.log('avg_correctness', avg_correct)

        # Clear the lists for the next validation epoch
        self.validation_conf_scores.clear()
        self.validation_correctness.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

##################################################################################### TRAINING ######################################################################################

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
tb_log_dir = base_log_dir / "tb_logs" / "ResNextNet_Scratch"
checkpoint_dir = base_log_dir / "saved_weights" / "ResNext_Scratch"

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
    filename="resnextnet_confnet-{epoch:02d}-{val_acc:.2f}"
)

# Initialize TensorBoard Logger
#tb_logger = TensorBoardLogger("tb_logs", name="vit_with_confnet")

# Initialize the model and trainer
max_epochs = 100
vit_confnet_model = ResNeXtClassifierWithConfNet(num_classes=4, learning_rate=0.0001, lambda_confidence = 0.75, alpha = class_weights_tensor, gamma = 2, beta = 0.3)

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