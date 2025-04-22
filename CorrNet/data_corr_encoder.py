##### Kaushik Dutta
##### Prprocess and load the data for training the CorrNet-AE

###### Importing the necessary Packages
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

##### Builder Functions

# Calculate the max and min value of the whole dataset
#def calc_max_min_normalization(lists_arrays):
#    max_value = -np.inf
#    min_value = np.inf
#    for lists in lists_arrays:
#        max_value = max(max_value, np.max(lists))
#        min_value = min(min_value, np.min(lists))
#    normalized_arrays = [(array - min_value) / (max_value - min_value) for array in lists_arrays]
#    return normalized_arrays
    
def normalize_rgb_image(lists_arrays):
    normalized_arrays = [array / 255.0 for array in lists_arrays]
    return normalized_arrays


#### Defining the datasets 
he_data = []
ihc_data = []
her2_scores = []   
    
###### Read the data from Dataset 1
#
#filepath_he = '/home/kslabuser1/BCI_dataset/HER2/TrainValAB/trainA/'
#filepath_ihc = '/home/kslabuser1/BCI_dataset/HER2/TrainValAB/trainB/'
#he_lists = os.listdir(filepath_he)
#ihc_lists = os.listdir(filepath_ihc)
#
#for he_list, ihc_list in zip(he_lists, ihc_lists):
#    he_path_final = os.path.join(filepath_he, he_list)
#    ihc_path_final = os.path.join(filepath_ihc, ihc_list)
#    if not os.path.exists(he_path_final) and os.path.exists(ihc_path_final):
#        print("File not found")
#    else:
#        he_img = cv2.imread(he_path_final, cv2.IMREAD_COLOR)
#        if he_img is None:
#            print(f"Failed to read the image: {he_path_final}")
#        else:
#            he_img = cv2.cvtColor(he_img, cv2.COLOR_RGB2GRAY)
#            he_img = np.expand_dims(he_img, axis=0)
#            he_data.append(he_img.astype(np.float32))
#
#        ihc_img = cv2.imread(ihc_path_final, cv2.IMREAD_COLOR)
#        if ihc_img is None:
#            print(f"Failed to read the image: {ihc_path_final}")
#        else:
#            ihc_img = cv2.cvtColor(ihc_img, cv2.COLOR_RGB2GRAY)
#            ihc_img = np.expand_dims(ihc_img, axis=0)
#            ihc_data.append(ihc_img.astype(np.float32))
#        
#
###### Read the data from Dataset 3
#
#filepath_he3 = '/home/kslabuser1/BCI_dataset/HER2/TrainValAB/valA/'
#filepath_ihc3 = '/home/kslabuser1/BCI_dataset/HER2/TrainValAB/valB/'
#he_lists3 = os.listdir(filepath_he3)
#ihc_lists3 = os.listdir(filepath_ihc3)
#
#for he_list, ihc_list in zip(he_lists3, ihc_lists3):
#    he_path_final = os.path.join(filepath_he3, he_list)
#    ihc_path_final = os.path.join(filepath_ihc3, ihc_list)
#    if not os.path.exists(he_path_final) and os.path.exists(ihc_path_final):
#        print("File not found")
#    else:
#        he_img = cv2.imread(he_path_final, cv2.IMREAD_COLOR)
#        if he_img is None:
#            print(f"Failed to read the image: {he_path_final}")
#        else:
#            he_img = cv2.cvtColor(he_img, cv2.COLOR_RGB2GRAY)
#            he_img = np.expand_dims(he_img, axis=0)
#            he_data.append(he_img.astype(np.float32))
#
#        ihc_img = cv2.imread(ihc_path_final, cv2.IMREAD_COLOR)
#        if ihc_img is None:
#            print(f"Failed to read the image: {ihc_path_final}")
#        else:
#            ihc_img = cv2.cvtColor(ihc_img, cv2.COLOR_RGB2GRAY)
#            ihc_img = np.expand_dims(ihc_img, axis=0)
#            ihc_data.append(ihc_img.astype(np.float32))
        
##### Read the data from Dataset 2

filepath_he2 = '/ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/BCI_dataset/HE/train/'
filepath_ihc2 = '/ceph/chpc/home-nfs/d.kaushik/Vision_Transformer/BCI_dataset/IHC/train/'
he_lists2 = os.listdir(filepath_he2)
ihc_lists2 = os.listdir(filepath_ihc2)

her2_scores = []
for he_list, ihc_list in zip(he_lists2, ihc_lists2):
    he_path_final = os.path.join(filepath_he2, he_list)
    ihc_path_final = os.path.join(filepath_ihc2, ihc_list)
    if not os.path.exists(he_path_final) and os.path.exists(ihc_path_final):
        print("File not found")
    else:
        he_img = cv2.imread(he_path_final, cv2.IMREAD_COLOR)
        if he_img is None:
            print(f"Failed to read the image: {he_path_final}")
        else:
            #he_img = cv2.cvtColor(he_img, cv2.COLOR_RGB2GRAY)
            #he_img = np.expand_dims(he_img, axis=0)
            #he_img = he_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
            he_img = np.transpose(he_img, (2, 0, 1))  # Convert from (H, W, C) to (C, H, W)
            he_data.append(he_img.astype(np.float32))
            #### Extracting the HER2 Classification Score 
            extracted_part = he_list.split('_')[2].split('.')[0]
            her2_scores.append(extracted_part)

        ihc_img = cv2.imread(ihc_path_final, cv2.IMREAD_COLOR)
        if ihc_img is None:
            print(f"Failed to read the image: {ihc_path_final}")
        else:
            #ihc_img = cv2.cvtColor(ihc_img, cv2.COLOR_RGB2GRAY)
            #ihc_img = np.expand_dims(ihc_img, axis=0)
            ihc_img = np.transpose(ihc_img, (2, 0, 1))
            ihc_data.append(ihc_img.astype(np.float32))
        
#he_data = calc_max_min_normalization(he_data)
#ihc_data = calc_max_min_normalization(ihc_data)
he_data = normalize_rgb_image(he_data)
ihc_data = normalize_rgb_image(ihc_data)

#### Defining a function to return the arrays
def train_data_loading():
    return he_data, ihc_data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    