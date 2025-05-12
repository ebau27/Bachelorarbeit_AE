# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:55:07 2025

@author: Eva Bauer 

Dataset: https://figshare.unimelb.edu.au/collections/Biomechanics_and_energetics_of_neurotypical_un_constrained_walking_Bacek2023_/6887854/1
"""

import os
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, RandomSampler, ConcatDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from PIL import Image
from tqdm import tqdm

from SequenceLoader import Loader
import Autoencoder as ae
import utils as ult
from Trainer import Trainer

from sklearn.metrics import roc_auc_score

#%% DataLoader
"""
- Load data and devide it into a Train, Evaluation split (80:20) 
- For the Test Dataset 2 Steps from every File and Sub8 are held back 
- The dataset only contains health and unlabeled data tensors [128,32,3]
"""

path = os.path.join(os.getcwd(), 'Data', 'human_walking_dataset')              
dataset = Loader(path, condition='normal')
orth_dataset = Loader(path, condition='orthosis')


#%% Test Set laden
test_dataset = Loader(path, condition='normal', split='test')
test_orth_dataset = Loader(path, condition='orthosis', split='test')

#%% Hyperparameter
maxpool = True
block_num = 2
batch_size = 32
lat_dim = 91
lr = 0.0013
weight_decay = 3.644e-5
dropout_rate = 0.222
phy_weight = 1e-5

epochs = 400

#%% Train, Valid, Performance Split
# Calculate the split sizes
train_size = int(0.8 * len(dataset))
val_size = int(0.75 * (len(dataset) - train_size))
perform_size = len(dataset) - train_size - val_size
spare_size = len(orth_dataset) - perform_size


# Randomly assign the Tensors to the splitsets (always the same assignment)
generator = torch.Generator().manual_seed(42)
train_subset, val_subset, perform_subset_norm = random_split(dataset, [train_size, val_size, perform_size], generator=generator)
spare_subset, perform_subset_orth = random_split(orth_dataset, [spare_size, perform_size], generator=generator)

perform_subset = ConcatDataset([perform_subset_norm, perform_subset_orth])

#TestSet
test_spare_size = len(test_dataset) - len(test_orth_dataset)
if test_spare_size > 0:
    test_spare, test_dataset_split = random_split(test_dataset, [abs(test_spare_size), len(test_dataset)-test_spare_size])
    test_subset = ConcatDataset([test_dataset_split, test_orth_dataset])
elif test_spare_size < 0:
    test_spare, test_orth_dataset_split = random_split(test_orth_dataset, [abs(test_spare_size), len(test_orth_dataset)+test_spare_size])
    test_subset = ConcatDataset([test_dataset, test_orth_dataset_split])
else:
    test_subset = ConcatDataset([test_dataset, test_orth_dataset])

# Load the batches 
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
perform_loader = DataLoader(perform_subset, batch_size=1, sampler=RandomSampler(perform_subset, generator=generator))
test_loader = DataLoader(test_subset, batch_size=1, sampler=RandomSampler(test_subset, generator=generator))
#%% Model Einstellungen

model = ae.adaptResNetAE(latent_dim=lat_dim, maxpool=True, block_num=block_num, dropout_rate=dropout_rate)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

train = Trainer(model=model, 
                train_loader=train_loader, 
                valid_loader=val_loader,
                perform_loader=perform_loader,
                criterion=criterion, 
                optimizer=optimizer, 
                end_epoch=epochs, 
                valid_epoch=20,
                name='Pendel_AE',
                phy=phy_weight,
                save=True)


#%% Modell trainieren 
train.fit()

