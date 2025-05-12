# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:47:56 2024

@author: Eva Bauer
"""

import torch
import torch.nn as nn
import torchvision


## Gerade nur für Input Tensor Dimension (batch, 3, 64, 128) 
class adaptEncoder(nn.Module):
    def __init__(self, latent_dim=32, maxpool=True, block_num=4, dropout_rate=0.0):
        super().__init__()
        
        resnet = torchvision.models.resnet18(weights=None)
        
        basic_layers = [resnet.conv1, 
                        resnet.bn1, 
                        resnet.relu]
        
        if maxpool:
            basic_layers.append(resnet.maxpool)
            
        self.basic = nn.Sequential(*basic_layers)
        
        ende = -4+block_num-2
        assert (ende > -6) and (ende <= -2)
        
        self.encode_layer = torch.nn.Sequential(*(list(resnet.children())[4:ende]))
        
        _,_,fc_dim = self.check_layer_dim()

        self.latent_block = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),
                                          nn.Flatten(),
                                          nn.Dropout(dropout_rate),
                                          nn.Linear(fc_dim, latent_dim),
                                          nn.ReLU())
        
    
        
    def check_layer_dim(self):
        resnet = torchvision.models.resnet18(weights=None)
        in_dim = resnet.conv1.in_channels
        layer_dims = []
        fc_dim = resnet.conv1.out_channels
        
        for name, layer in self.encode_layer.named_modules():
            if name[-7:] == '0.conv1':
                layer_dims.append(layer.in_channels)
                fc_dim = layer.out_channels
        
        return in_dim, layer_dims, fc_dim
    
    
    def forward(self,x):
        x_basic = self.basic(x)
        x_enc = self.encode_layer(x_basic)
        x_lat = self.latent_block(x_enc)
        return x_lat
    
    
