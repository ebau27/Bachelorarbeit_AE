# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:55:07 2025

@author: Eva
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

#Load entire Dataset a. preprocess (check Basic_Funktion - get_sequenceList)
#path = os.path.join(os.getcwd(), 'Data', 'human_walking_dataset')
#dataset = Loader(path, condition='normal')
#orth_dataset = Loader(path, condition='orthosis')


#torch.save(dataset, 'Data/norm_dataset.pt')
#torch.save(orth_dataset, 'Data/orth_dataset.pt')


#%% Daten Laden 
dataset = torch.load('Data/norm_dataset_neu.pt')
orth_dataset = torch.load('Data/orth_dataset_neu.pt')

#%% Test Set laden
#test_dataset = Loader(path, condition='normal', split='test')
#test_orth_dataset = Loader(path, condition='orthosis', split='test')

#torch.save(test_dataset, 'Data/test_dataset.pt')
#torch.save(test_orth_dataset, 'Data/test_orth_dataset.pt')

#%%
test_dataset = torch.load('Data/test_dataset_neu.pt')
test_orth_dataset = torch.load('Data/test_orth_dataset_neu.pt')


#%%
#sub8_dataset = Loader(path, condition='normal', unknown=True)
#sub8_orth_dataset = Loader(path, condition='orthosis', unknown=True)

#torch.save(sub8_dataset, 'Data/sub8_dataset.pt')
#torch.save(sub8_orth_dataset, 'Data/sub8_orth_dataset.pt')

#%%
#sub8_dataset = torch.load('Data/norm_dataset.pt')
#sub8_orth_dataset = torch.load('Data/orth_dataset.pt')
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

# #Sub8Set
# sub8_spare_size = len(sub8_dataset) - len(sub8_orth_dataset)
# if sub8_spare_size > 0:
#     sub8_spare, sub8_dataset_split = random_split(sub8_dataset, [abs(sub8_spare_size), len(sub8_dataset)-sub8_spare_size])
#     sub8_subset = ConcatDataset([sub8_dataset_split, sub8_orth_dataset])
# elif sub8_spare_size < 0:
#     sub8_spare, sub8_orth_dataset_split = random_split(sub8_orth_dataset, [abs(sub8_spare_size), len(sub8_orth_dataset)+sub8_spare_size])
#     sub8_subset = ConcatDataset([sub8_dataset, sub8_orth_dataset_split])
# else:
#     sub8_subset = ConcatDataset([sub8_dataset, sub8_orth_dataset])

# Load the batches 
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
perform_loader = DataLoader(perform_subset, batch_size=1, sampler=RandomSampler(perform_subset, generator=generator))
test_loader = DataLoader(test_subset, batch_size=1, sampler=RandomSampler(test_subset, generator=generator))
#sub8_loader = DataLoader(sub8_subset, batch_size=1, sampler=RandomSampler(sub8_subset, generator=generator))

#%% Model Einstellungen

model = ae.adaptResNetAE(latent_dim=lat_dim, maxpool=True, block_num=block_num, dropout_rate=dropout_rate)

#PATH = os.path.join(os.getcwd(),'Models','2025_05_03_11_14_10_Pendel_AE.pth')
#checkpoint = torch.load(PATH, weights_only=True)

#PATH = os.path.join(os.getcwd(),'Models','2025_04_29_12_11_44_naive_AE.pth')
#checkpoint = torch.load(PATH, weights_only=True)

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
                #checkpoint=checkpoint,
                save=True)

#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#%% Modell trainieren 
train.fit()


#%% 
model.eval()


#%% Recon Naive Clip (NORMAL)
model.eval()
steps,_ = next(iter(val_loader))

step = steps.permute(0,3,2,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
step = step.to(device)

recon = model(step)

show_step = step[4].permute(2,1,0)
show_recon = recon[4].permute(2,1,0)


fig = plt.figure(figsize=(6,10))
ax = plt.subplot(projection='3d')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ani = ult.model_animation([show_step, show_recon], fig, ax, color=[['original','blue','green'], ['reconstructed', 'red', 'orange']])
#ani.save('Imgs/naive_AE_recon.gif', dpi=500)
plt.tight_layout()
plt.show()


#%% Recon Naive Clip (ABNORMAL)
model.eval()
label = 1

while label == 1:
    print(label)
    steps, label = next(iter(perform_loader))

step = steps.permute(0,3,2,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
step = step.to(device)

recon = model(step)

show_step = step[0].permute(2,1,0)
show_recon = recon[0].permute(2,1,0)


fig = plt.figure(figsize=(6,10))
ax = plt.subplot(projection='3d')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ani = ult.model_animation([show_step, show_recon], fig, ax, color=[['abnormal','blue','green'], ['reconstructed', 'red', 'orange']])
#ani.save('Imgs/pendel_AE_recon_norm.gif', dpi=500)
plt.tight_layout()
plt.show()

#%% Recon Naive Clip (TEST)
model.eval()
label = 0

while label == 0:
    print(label)
    steps, label = next(iter(test_loader))

step = steps.permute(0,3,2,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
step = step.to(device)

recon = model(step)

show_step = step[0].permute(2,1,0)
show_recon = recon[0].permute(2,1,0)


fig = plt.figure(figsize=(6,10))
ax = plt.subplot(projection='3d')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ani = ult.model_animation([show_step, show_recon], fig, ax, color=[['original','blue','green'], ['reconstructed', 'red', 'orange']])
#ani.save('Imgs/pendel_AE_recon_norm_test.gif', dpi=500)
plt.tight_layout()
plt.show()

#%% Recon Naive Clip (SUB8)
model.eval()
label = 0

while label == 0:
    print(label)
    steps, label = next(iter(sub8_loader))

step = steps.permute(0,3,2,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
step = step.to(device)

recon = model(step)

show_step = step[0].permute(2,1,0)
show_recon = recon[0].permute(2,1,0)


fig = plt.figure(figsize=(6,10))
ax = plt.subplot(projection='3d')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ani = ult.model_animation([show_step, show_recon], fig, ax, color=[['original','blue','green'], ['reconstructed', 'red', 'orange']])
#ani.save('Imgs/naive_AE_recon_norm_test.gif', dpi=500)
plt.tight_layout()
plt.show()
#%% Recon Naive 2D

fig, (ax1,ax2) = plt.subplots(2,1, figsize=(30,15))

for j, frame in enumerate(show_step):
    if (j%20) == 0:
        x = []
        y = []
        z = []
        for i,marker in enumerate(frame):
            x.append(float(marker[0])-float(j)*0.05)
            y.append(float(marker[1])-float(j)*0.05)
            z.append(float(marker[2]))
            
        ax1.scatter(y,z, c='#0343df')
        ax2.scatter(x,z, c='#0343df')
        
        for label,part in ult.MARKER_LOC.items():
            for con in part:
                lx = [x[con[1]],x[con[0]]]
                ly = [y[con[1]],y[con[0]]]
                lz = [z[con[1]],z[con[0]]]
                ax1.plot(ly,lz, c='#008000')
                ax2.plot(lx,lz, c='#008000')
                
for j, frame in enumerate(show_recon):
    if (j%20) == 0:
        x = []
        y = []
        z = []
        for i,marker in enumerate(frame):
            x.append(float(marker[0])-float(j)*0.05)
            y.append(float(marker[1])-float(j)*0.05)
            z.append(float(marker[2]))
            
        ax1.scatter(y,z, c='red')
        ax2.scatter(x,z, c='red')

        for label,part in ult.MARKER_LOC.items():
            for con in part:
                lx = [x[con[1]],x[con[0]]]
                ly = [y[con[1]],y[con[0]]]
                lz = [z[con[1]],z[con[0]]]
                ax1.plot(ly,lz, c='orange')
                ax2.plot(lx,lz, c='orange')

ax1.invert_xaxis()
ax2.invert_xaxis()

plt.tight_layout()
#plt.gca().invert_xaxis()
#plt.gca().set_axis_off()
plt.savefig('Imgs/recon_abnormal_pendel_2d_test_short', bbox_inches='tight', dpi=500)
plt.show()


#%% Pseudo Naive AE normal 
show_step_t = show_step.transpose(1,0)
show_recon_t = show_recon.transpose(1,0)

show_step_t = show_step_t.cpu().detach().numpy()
show_recon_t = show_recon_t.cpu().detach().numpy()

dif = show_step_t - show_recon_t

ult.plot_pseudo([(show_step_t+1)*0.5, (show_recon_t+1)*0.5, 1-abs(dif)], 
            ['Original Stride', 'Reconstructed Stride', 'Difference (Inverted)'], 
            [True, True, True],
            save='Imgs/Pseudo_Img_pendel_recon_abnormal_test_short',
            fig_num=1)

#%% AUC Naive
with torch.no_grad():
    pred = []
    real = []
    for steps, labels in test_loader:
        # Steps throught model - only one Step at a time (with labels)
        steps = steps.permute(0,3,2,1)
        steps = steps.to(device)
    
        recon = model(steps)
    
        # calculate Loss
        data_loss = criterion(recon, steps)
        phy_loss = ult.phy_loss(recon)
        
        loss = data_loss + phy_weight*phy_loss
        
        # Append Rec_Error and Label
        pred.append(loss.item())
        real.append(labels.item())

    ## Calculate Area under Curve (AUC) for Receiver Operating Characteristic (ROC)
auc_val = roc_auc_score(real, pred)
p_auc = round(auc_val,5)

pred = np.array(pred)
real = np.array(real)

error_normal = [pred[i].item() for i in np.where(real == 0)[0]]
error_patho = [pred[i].item() for i in np.where(real == 1)[0]]


plt.figure(figsize=(8, 5))

plt.hist(error_normal, bins=20, alpha=0.5, label='Normal', color='blue')
plt.hist(error_patho, bins=20, alpha=0.5, label='Abnormal', color='red')

plt.xlabel('Reconstruction Loss')
plt.ylabel('Frequency')
plt.title('Distribution of the Reconstruction Loss (Normal vs. Abnormal) PIAE')
plt.legend()
plt.text(0.002, 40, f'AUC={p_auc}')
plt.savefig('Imgs/verteilung_fehler_pendel_test_short', bbox_inches='tight', dpi=500)
plt.show()

#%% AUC Sym
def calc_sym_factor(in_step):
    sym_factor = 0.0

    n_frames = in_step.shape[3]

    Mitte = in_step[:,:,:22,:].mean(dim=2)
    #Mitte = in_step.mean(dim=2)
    Mitte = ult.close_loop(Mitte)
    Mitte = Mitte.squeeze()

    X = torch.fft.fft(Mitte)

    for i, X_i in enumerate(X):
        c_list = []
        phi_list = []
    
        max_harm = min(6, n_frames-1)
        for k in range(1,max_harm+1):
            re_k = X_i[k].real
            im_k = X_i[k].imag
        
            a_k = (-2.0 / n_frames) * im_k
            b_k = (2.0 / n_frames) * re_k
        
            c_k = torch.sqrt(a_k**2 + b_k**2)
            phi_k = (torch.pi/2.0)*torch.sign(b_k) - torch.atan(a_k / (b_k+1e-8))
        
            c_list.append(c_k)
            phi_list.append(phi_k)
        
        if i == 0:
            S = (c_list[0] + c_list[2] + c_list[4]) / sum(c_list)
        else:
            S = (c_list[1] + c_list[3] + c_list[5]) / sum(c_list)
        
        sym_factor += (1.0-S)    
        
    return sym_factor

    
#%%
with torch.no_grad():
    pred = []
    real = []
    for steps, labels in perform_loader:
        # Steps throught model - only one Step at a time (with labels)
        steps = steps.permute(0,3,2,1)
        steps = steps.to(device)
    
        recon = model(steps)
        
        # calculate Loss
        loss = criterion(recon, steps)
        
        # calculate Sym
        sym = calc_sym_factor(steps)
        #print(labels.item(), loss.item(), sym.item())
        
        dif_factor = loss + 1e-3*sym
        
        # Append Rec_Error and Label
        pred.append(dif_factor.item())
        real.append(labels.item())


## Calculate Area under Curve (AUC) for Receiver Operating Characteristic (ROC)
auc_val = roc_auc_score(real, pred)
p_auc = round(auc_val,5)

pred = np.array(pred)
real = np.array(real)

error_normal = [pred[i].item() for i in np.where(real == 0)[0]]
error_patho = [pred[i].item() for i in np.where(real == 1)[0]]


plt.figure(figsize=(8, 5))

plt.hist(error_normal, bins=20, alpha=0.5, label='Normal', color='blue')
plt.hist(error_patho, bins=20, alpha=0.5, label='Gestört', color='red')

plt.xlabel('Rekonstruktionsfehler')
plt.ylabel('Häufigkeit')
plt.title('Verteilung der Rekonstruktionsfehler (Normal vs. Gestört) Symmetrie')
plt.legend()
plt.text(0.0030, 100, f'AUC={p_auc}')
#plt.savefig('Imgs/verteilung_fehler_sym')
plt.show()

#%% Lissajous Model 
model = ae.expanded_AE_model(latent_dim=lat_dim, maxpool=True, block_num=block_num, dropout_rate=dropout_rate, load_coef=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

PATH = os.path.join(os.getcwd(),'Models','04_30_20_37_55_Lissajous.pth')
checkpoint = torch.load(PATH, weights_only=True)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#%% 
model.eval()
label = 0

while label == 0:
    print(label)
    steps, label = next(iter(test_loader))

step = steps.permute(0,3,2,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
step = step.to(device)

t_norm = torch.linspace(0, 2*torch.pi, 128, device=device).expand(steps.shape[0], -1)
recon, lis_mitte = model(step, t_norm)

step_mitte = step.mean(dim=2)
step_mitte = ult.close_loop(step_mitte)

recon_mitte = recon.mean(dim=2)
recon_mitte = ult.close_loop(recon_mitte)

show_step = step[0].permute(2,1,0)
show_recon = recon[0].permute(2,1,0)

fig = plt.figure()
ax = plt.subplot(projection='3d')
plt.plot(recon_mitte[0,0,:].cpu().detach().numpy(), recon_mitte[0,1,:].cpu().detach().numpy(), recon_mitte[0,2,:].cpu().detach().numpy(), c='orange', label='reconstructed')
plt.plot(step_mitte[0,0,:].cpu().detach().numpy(), step_mitte[0,1,:].cpu().detach().numpy(), step_mitte[0,2,:].cpu().detach().numpy(), c='blue', label='original')
#plt.plot(step_mitte_c[0,0,:].cpu().detach().numpy(), step_mitte_c[0,1,:].cpu().detach().numpy(), step_mitte_c[0,2,:].cpu().detach().numpy(), c='blue')
plt.plot(lis_mitte[0,0,:].cpu().detach().numpy(), lis_mitte[0,1,:].cpu().detach().numpy(), lis_mitte[0,2,:].cpu().detach().numpy(), c='green', label='approx. Lissajous')
plt.legend(loc='upper left')
plt.savefig('Imgs/LisAE_test_normal_gCOM', dpi=500)
plt.show()


fig = plt.figure(figsize=(6,10))
ax = plt.subplot(projection='3d')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ani = ult.model_animation([show_step, show_recon], fig, ax, color=[['original','blue','green'], ['reconstructed', 'red', 'orange']])
ani.save('Imgs/LisAE_recon_norm_test.gif', dpi=500)
plt.tight_layout()
plt.show()

#%% Recon Naive 2D

fig, (ax1,ax2) = plt.subplots(2,1, figsize=(30,15))

for j, frame in enumerate(show_step):
    if (j%20) == 0:
        x = []
        y = []
        z = []
        for i,marker in enumerate(frame):
            x.append(float(marker[0])-float(j)*0.05)
            y.append(float(marker[1])-float(j)*0.05)
            z.append(float(marker[2]))
            
        ax1.scatter(y,z, c='#0343df')
        ax2.scatter(x,z, c='#0343df')
        
        for label,part in ult.MARKER_LOC.items():
            for con in part:
                lx = [x[con[1]],x[con[0]]]
                ly = [y[con[1]],y[con[0]]]
                lz = [z[con[1]],z[con[0]]]
                ax1.plot(ly,lz, c='#008000')
                ax2.plot(lx,lz, c='#008000')
                
for j, frame in enumerate(show_recon):
    if (j%20) == 0:
        x = []
        y = []
        z = []
        for i,marker in enumerate(frame):
            x.append(float(marker[0])-float(j)*0.05)
            y.append(float(marker[1])-float(j)*0.05)
            z.append(float(marker[2]))
            
        ax1.scatter(y,z, c='red')
        ax2.scatter(x,z, c='red')

        for label,part in ult.MARKER_LOC.items():
            for con in part:
                lx = [x[con[1]],x[con[0]]]
                ly = [y[con[1]],y[con[0]]]
                lz = [z[con[1]],z[con[0]]]
                ax1.plot(ly,lz, c='orange')
                ax2.plot(lx,lz, c='orange')

ax1.invert_xaxis()
ax2.invert_xaxis()

plt.tight_layout()
#plt.gca().invert_xaxis()
#plt.gca().set_axis_off()
plt.savefig('Imgs/recon_abnormal_LisAE_2d_test', bbox_inches='tight', dpi=500)
plt.show()


#%% Pseudo Naive AE normal 
show_step_t = show_step.transpose(1,0)
show_recon_t = show_recon.transpose(1,0)

show_step_t = show_step_t.cpu().detach().numpy()
show_recon_t = show_recon_t.cpu().detach().numpy()

dif = show_step_t - show_recon_t

ult.plot_pseudo([(show_step_t+1)*0.5, (show_recon_t+1)*0.5, 1-abs(dif)], 
            ['Original Stride', 'Reconstructed Stride', 'Difference (Inverted)'], 
            [True, True, True],
            save='Imgs/Pseudo_Img_LisAE_recon_normal_test',
            fig_num=1)

#%% AUC Naive
alpha = 10
beta = 0.1
with torch.no_grad():
    pred = []
    real = []
    for steps, labels in test_loader:
        # Steps throught model - only one Step at a time (with labels)
        steps = steps.permute(0,3,2,1)
        steps = steps.to(device)
    
        t_norm = torch.linspace(0, 2*torch.pi, 128, device=device).expand(steps.shape[0], -1)
        
        recon, liss_mitte = model(steps, t_norm)
        
        data_mitte = steps.mean(dim=2)
        data_mitte = ult.close_loop(data_mitte)
        
        data_loss = criterion(recon, steps)
        liss_loss = criterion(liss_mitte, data_mitte)
        
        loss = alpha*data_loss + beta*liss_loss
        
        # Append Rec_Error and Label
        pred.append(loss.item())
        real.append(labels.item())

    ## Calculate Area under Curve (AUC) for Receiver Operating Characteristic (ROC)
auc_val = roc_auc_score(real, pred)
p_auc = round(auc_val,5)

pred = np.array(pred)
real = np.array(real)

error_normal = [pred[i].item() for i in np.where(real == 0)[0]]
error_patho = [pred[i].item() for i in np.where(real == 1)[0]]


plt.figure(figsize=(8, 5))

plt.hist(error_normal, bins=20, alpha=0.5, label='Normal', color='blue')
plt.hist(error_patho, bins=20, alpha=0.5, label='Abnormal', color='red')

plt.xlabel('Reconstruction Loss')
plt.ylabel('Frequency')
plt.title('Distribution of the Reconstruction Loss (Normal vs. Abnormal) Lissajous PIAE')
plt.legend()
plt.text(0.03, 20, f'AUC={p_auc}')
plt.savefig('Imgs/verteilung_fehler_LisAE_test', bbox_inches='tight', dpi=500)
plt.show()
#%% 
# def phy_loss(recon_batch):
#     dif = []
#     for step in recon_batch:
#         angle_COB_r = []
#         angle_COB_l = []
#         for frame in range(step.size(dim=2)):
#             COB_r = torch.mean(step[:,[4,5,6,7,8,9,10,11,12],frame], axis=1)
#             COB_l = torch.mean(step[:,[13,14,15,16,17,18,19,20,21],frame], axis=1)
        
#             G = torch.tensor([0.0, 0.0, -1.0], device=device)
        
#             angle_COB_r.append(ult.calc_joint_angle(COB_r, G))
#             angle_COB_l.append(ult.calc_joint_angle(COB_l, G))
        
#         dif.append(ult.calc_simularity(angle_COB_l, angle_COB_r))
        
        
#     return torch.mean(torch.tensor(dif))


# def calc_joint_angle(v1,v2):
#     dot_product = (v1 * v2).sum(dim=1)
    
#     norm_v1 = v1.norm(dim=1)
#     norm_v2 = v2.norm(dim=1)
    
#     cos_angle = dot_product / (norm_v1 * norm_v2 + 1e-8)  
#     cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
#     angle_rad = torch.acos(cos_angle)
#     angle_deg = torch.rad2deg(angle_rad)
    
#     return angle_deg


# def phy_loss_new(recon_batch):
#     COB_r = recon_batch[:, :, [4,5,6,7,8,9,10,11,12], :].mean(dim=2)
#     COB_l = recon_batch[:, :, [13,14,15,16,17,18,19,20,21], :].mean(dim=2)
    
#     G = torch.tensor([0.0, 0.0, -1.0], device=recon_batch.device).view(1,3,1)
    
#     angle_r = calc_joint_angle(COB_r, G) 
#     angle_l = calc_joint_angle(COB_l, G)
    
#     angle_r_shifted = torch.roll(angle_r, shifts=angle_r.size(1)//2, dims=1)
    
#     dif = (angle_l - angle_r_shifted)**2 
    
#     return dif.mean()

# #%% 
# model = ae.adaptResNetAE(maxpool=True,block_num=4)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# train_loss = []

# progress = tqdm(train_loader)
# for steps,_ in progress:
#     # Steps throught model
#     steps = steps.permute(0,3,2,1)
#     steps = steps.to(device)
#     recon = model(steps)
    
#     # calculate Loss
#     data_loss = criterion(recon, steps)
#     physic_loss = ult.phy_loss(recon)
#     loss = data_loss + 0.00001*physic_loss
#     train_loss.append(loss.item())
    
#     # Backpropagation
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# #%%
# plt.figure()
# plt.plot(train_loss)
# plt.show()

# #%%
# step = recon[0]
# dif = []

# for step in recon:
#     angle_COB_r = []
#     angle_COB_l = []
#     for frame in range(step.size(dim=2)):
#         COB_r = torch.mean(step[:,[4,5,6,7,8,9,10,11,12],frame], axis=1)
#         COB_l = torch.mean(step[:,[13,14,15,16,17,18,19,20,21],frame], axis=1)
    
#         G = torch.tensor([0.0, 0.0, -1.0])
    
#         angle_COB_r.append(ult.calc_joint_angle(COB_r, G))
#         angle_COB_l.append(ult.calc_joint_angle(COB_l, G))
    
#     dif.append(ult.calc_simularity(angle_COB_l, angle_COB_r))
    



# #%%
# model.eval()
# steps,_ = next(iter(val_loader))

# step = steps.permute(0,3,2,1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# step = step.to(device)

# recon = model(step)

# show_step = step[0].permute(2,1,0)
# show_recon = recon[0].permute(2,1,0)



# fig = plt.figure(figsize=(6,10))
# ax = plt.subplot(projection='3d')
# ani = ult.model_animation([show_step, show_recon], fig, ax, color=[['original','blue','green'], ['recon', 'red', 'orange']])
# #ani.save('Imgs/mittel_normal_gestört_schritt.gif')
# plt.show()

# #%% 
# test = []
# real = []

# with torch.no_grad():
#     for steps,label in perform_loader:
#         steps = steps.permute(0,3,2,1)
#         steps = steps.to(device)
#         recon = model(steps)
    
#         loss = criterion(recon, steps)
#         test.append(loss.item())
#         real.append(label.item())
        
# #%%
# from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# auc_val = roc_auc_score(real, test)

# fpr, tpr, thresholds = roc_curve(real, test)

# roc_auc = auc(fpr, tpr)


# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonale (Zufallsmodell)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC-Kurve')
# plt.legend(loc="lower right")
# plt.show()


# #%%
# def plot_pseudo(input_tensor:list, label:list, color:list, save=None, fig_num=1):
#     Marker_Label = ult.Marker_Label
    
#     plt.figure(num=fig_num, figsize=(20,10), dpi=100)
    
#     for idx,tensor in enumerate(input_tensor):
#         plt.subplot(1,len(input_tensor),idx+1)
#         if color[idx]:
#             plt.imshow(tensor, aspect='auto')
#         else:
#             plt.imshow(tensor, cmap='gray_r', vmin=0, vmax=0.2, aspect='auto')
#             #plt.imshow(tensor, cmap='gray_r', aspect='auto')
        
#         plt.colorbar()
#         plt.yticks(ticks=range(len(Marker_Label)), labels=Marker_Label)
#         plt.title(label[idx])
#         plt.xlabel('Frames')
#         plt.ylabel('Marker')
    
#     if save is not None:
#         plt.savefig(save)
#     plt.tight_layout()
#     plt.show()
    
# trans_step = step.transpose(1,0)
# trans_rec = recon_st.transpose(1,0)

# plot_pseudo([(trans_step+1)*0.5, (trans_rec+1)*0.5], ['step','reconstep'], [True, True])