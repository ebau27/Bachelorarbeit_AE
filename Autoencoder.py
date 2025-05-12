# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:18:16 2024

@author: Eva Bauer
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision

from adapt_ResNet_Decoder import adaptDecoder
from adapt_ResNet_Encoder import adaptEncoder


    
class adaptResNetAE(nn.Module):
    def __init__(self, latent_dim=32, input_dim=[3,32,128], block_num=4, maxpool=True, dropout_rate=0.0):
        super().__init__()
        self.encoder = adaptEncoder(latent_dim=latent_dim, maxpool=maxpool, block_num=block_num, dropout_rate=dropout_rate)
        
        test = torch.Tensor(1,input_dim[0],input_dim[1],input_dim[2])
        test_x = self.encoder.basic(test)
        test_x = self.encoder.encode_layer(test_x)
        #print(f'test {test_x.shape}')
        h_size, w_size = test_x.shape[2], test_x.shape[3]
        in_dim, layer_dims, fc_dim = self.encoder.check_layer_dim()
        self.decoder = adaptDecoder(latent_dim=latent_dim, in_dim=in_dim, layer_dims=layer_dims, fc_dim=fc_dim, rev_pool=[h_size, w_size], maxpool=maxpool) 
        
        
    def forward(self, x):
        # von (N,3,50,100) zu (N,3,64,128)
        #x = nn.functional.pad(x, (14,14,7,7), mode='constant', value=0) #links, rechts, oben, unten
        #print(x.shape)
        x_lat = self.encoder(x)
        x_decoded = self.decoder(x_lat)
        #print(x_decoded.shape)
        # zurück zu (N,3,50,100)
        #x_decoded = x_decoded[:, :, 7:-7, 14:-14]
        
        return x_decoded
    
class Lissajous_fit(nn.Module):
    def __init__(self, n_harm=6):
        super().__init__()
        self.n_harm = n_harm
        
        self.a0_x = nn.Parameter(torch.zeros(1))
        self.a_x  = nn.Parameter(torch.zeros(n_harm))
        self.b_x  = nn.Parameter(torch.zeros(n_harm))
        
        self.a0_y = nn.Parameter(torch.zeros(1))
        self.a_y  = nn.Parameter(torch.zeros(n_harm))
        self.b_y  = nn.Parameter(torch.zeros(n_harm))
        
        self.a0_z = nn.Parameter(torch.zeros(1))
        self.a_z  = nn.Parameter(torch.zeros(n_harm))
        self.b_z  = nn.Parameter(torch.zeros(n_harm))
        
    def forward(self, t):
        B, N = t.shape
        
        x_r = self.a0_x + 0.0*t
        y_r = self.a0_y + 0.0*t
        z_r = self.a0_z + 0.0*t
        
        for i in range(1, self.n_harm+1):
            it = i * t
            x_r += self.a_x[i-1]*torch.sin(it) + self.b_x[i-1]*torch.cos(it)
            y_r += self.a_y[i-1]*torch.sin(it) + self.b_y[i-1]*torch.cos(it)
            z_r += self.a_z[i-1]*torch.sin(it) + self.b_z[i-1]*torch.cos(it)

        return torch.stack([x_r, y_r, z_r], dim=1)
    

class expanded_AE_model(nn.Module):
    def __init__(self, latent_dim=32, input_dim=[3,32,128], block_num=4, maxpool=True, dropout_rate=0.0, n_harm=6, load_coef=True):
        super().__init__()
        self.encoder = adaptEncoder(latent_dim=latent_dim, maxpool=maxpool, block_num=block_num, dropout_rate=dropout_rate)
        
        test = torch.Tensor(1,input_dim[0],input_dim[1],input_dim[2])
        test_x = self.encoder.basic(test)
        test_x = self.encoder.encode_layer(test_x)
        #print(f'test {test_x.shape}')
        h_size, w_size = test_x.shape[2], test_x.shape[3]
        in_dim, layer_dims, fc_dim = self.encoder.check_layer_dim()
        self.decoder = adaptDecoder(latent_dim=latent_dim, in_dim=in_dim, layer_dims=layer_dims, fc_dim=fc_dim, rev_pool=[h_size, w_size], maxpool=maxpool) 
        
        self.lissajous = Lissajous_fit(n_harm=n_harm)
        
        if load_coef:
            with torch.no_grad():
                c = torch.load("Data/lissajous_coeffs.pt", weights_only=True)
                self.lissajous.a0_x.copy_ = torch.tensor(c['x']['a0'], dtype=torch.float32)
                self.lissajous.a_x.copy_  = torch.tensor(c['x']['a'],  dtype=torch.float32)
                self.lissajous.b_x.copy_  = torch.tensor(c['x']['b'],  dtype=torch.float32)

                self.lissajous.a0_y.copy_ = torch.tensor(c['y']['a0'], dtype=torch.float32)
                self.lissajous.a_y.copy_  = torch.tensor(c['y']['a'],  dtype=torch.float32)
                self.lissajous.b_y.copy_  = torch.tensor(c['y']['b'],  dtype=torch.float32)

                self.lissajous.a0_z.copy_ = torch.tensor(c['z']['a0'], dtype=torch.float32)
                self.lissajous.a_z.copy_  = torch.tensor(c['z']['a'],  dtype=torch.float32)
                self.lissajous.b_z.copy_  = torch.tensor(c['z']['b'],  dtype=torch.float32)
        
        
    def forward(self, X, t):
        # X.shape (Batch_size, Channel, Joints, Frames)
        
        x_lat = self.encoder(X)
        x_decoded = self.decoder(x_lat)
        
        liss_batch = self.lissajous(t)

        
        return x_decoded, liss_batch
    
    
    
    
    
    