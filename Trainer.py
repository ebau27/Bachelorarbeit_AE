# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 12:44:54 2025

@author: Eva Bauer
"""

from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc

import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import utils as ult

import PIL.Image
from torchvision.transforms import ToTensor


class Trainer:
    def __init__(self, model, train_loader, valid_loader, perform_loader, criterion, optimizer, end_epoch, valid_epoch, checkpoint=None, name='', save=False, phy=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.perform_loader = perform_loader
        
        
        self.model_code = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_") + name
        
        self.save = save
        if self.save:
            
            TBOARD = os.path.join(os.getcwd(), f'TensorBoard/{name}/exp_{self.model_code}')
            if not os.path.exists(TBOARD):
                os.makedirs(TBOARD)
            
            shutil.rmtree(TBOARD)
            self.writer = SummaryWriter(TBOARD)
            
        self.start_epoch = 0
        
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            print('Model geladen!')
        
        self.end_epoch = end_epoch
        self.valid_epoch = valid_epoch
        
        self.phy = phy
            


    def train_step(self, epoch):
        loss_list = []
        phy_list = []
        self.model.train()
        for steps,_ in self.train_loader:
            steps = steps.permute(0,3,2,1)
            steps = steps.to(self.device)
            recon = self.model(steps)
            
            if self.phy is None:
                loss = self.criterion(recon, steps)
                
            else:
                phy_loss = ult.phy_loss(recon)
                data_loss = self.criterion(recon, steps)
                phy_list.append(self.phy*phy_loss.item())
                loss = data_loss + self.phy*phy_loss
                
            loss_list.append(loss.item())
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if self.save:
            self.writer.add_scalar('Train Loss', round(np.mean(loss_list), 5), global_step=epoch)
            if len(phy_list) != 0:
                self.writer.add_scalar('Physic Loss', round(np.mean(phy_list), 5), global_step=epoch)
        return 
    
    
    @torch.no_grad()
    def valid_step(self, epoch):
        self.model.eval()
        
        loss_list = []
        for steps,_ in self.valid_loader:
            steps = steps.permute(0,3,2,1)
            steps = steps.to(self.device)
            recon = self.model(steps)
            
            if self.phy is None:
                loss = self.criterion(recon, steps)
                
            else:
                phy_loss = ult.phy_loss(recon)
                data_loss = self.criterion(recon, steps)
                loss = data_loss + self.phy*phy_loss
            
            loss_list.append(loss.item())
        
        if self.save:
            self.writer.add_scalar('Validation Loss', round(np.mean(loss_list), 5), global_step=epoch)
            
        return 
    
    
    @torch.no_grad()
    def performance_Metric(self, epoch):
        self.model.eval()
        
        pred = []
        real = []
        for steps, labels in self.perform_loader:
            steps  = steps.permute(0,3,2,1)
            steps = steps.to(self.device)
            recon = self.model(steps)
            
            if self.phy is None:
                loss = self.criterion(recon, steps)
                
            else:
                phy_loss = ult.phy_loss(recon)
                data_loss = self.criterion(recon, steps)
                loss = data_loss + self.phy*phy_loss
            
            pred.append(loss.item())
            real.append(labels.item())
        
        auc_val = roc_auc_score(real, pred)
        
        
        pred = np.array(pred)
        real = np.array(real)

        error_normal = [pred[i].item() for i in np.where(real == 0)[0]]
        error_patho = [pred[i].item() for i in np.where(real == 1)[0]]
        
        plot_buf = ult.gen_histogramm(error_normal, error_patho)

        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image)
        
        if self.save:
            self.writer.add_image('Histogramm', image, global_step=epoch)
            self.writer.add_scalar('Area under Curve', round(auc_val, 3), global_step=epoch)
            
        return 
            
        

    def fit(self):    
        progress = tqdm(range(self.start_epoch,self.end_epoch))
        self.files = []
        for epoch in progress:
            self.train_step(epoch)
    
            if (epoch+1)%self.valid_epoch == 0:
                self.valid_step(epoch)
                self.performance_Metric(epoch)
                
                
        if self.save:
            ult.save_model(epoch, self.model, self.optimizer, f'Models/{self.model_code}.pth')
            self.writer.close()
