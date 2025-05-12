# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:50:09 2024

@author: Eva
mit Input von https://github.com/eleannavali/resnet-18-autoencoder/tree/main
und ChatGPT CAE optimierungstipps (11.12.24 10:58) (InvertBasicBlock)
"""


#import torch
import torch.nn as nn
#import numpy as np
#import torchvision
    
## von ChatGPT CAE optimierungstipps (11.12.24 10:58)
class InvertBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        
        self.upsample = upsample
        
        # Transponierte Convolution: Vergrößert die räumlichen Dimensionen
        self.conv1 = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=2 if upsample else 1, 
            padding=1, 
            output_padding=1 if upsample else 0,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Zweite Convolution: Behält die räumlichen Dimensionen bei
        self.conv2 = nn.ConvTranspose2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut-Verbindung: Anpassung der Kanalanzahl
        if upsample:
            self.upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=2, 
                    output_padding=1, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.upsample_layer = None

    def forward(self, x):
        # Hauptpfad
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut
        shortcut = self.upsample_layer(x) if self.upsample else x

        # Residual-Verbindung
        out += shortcut
        out = self.relu(out)

        return out
    
    
    
    
class adaptDecoder(nn.Module):
    def __init__(self, latent_dim:int, in_dim:int, layer_dims:list, fc_dim:int, rev_pool:list, maxpool=True):
        super().__init__()
        self.fc_dim = fc_dim
        
        self.rev_latent_layer = nn.Linear(latent_dim, self.fc_dim)
        self.rev_avgpool = nn.ConvTranspose2d(in_channels=self.fc_dim, out_channels=self.fc_dim, kernel_size=(rev_pool[0], rev_pool[1]), stride=(rev_pool[0], rev_pool[1]))
        
        
        
        if len(layer_dims) > 1: 
            layers = [self.make_layer(in_channel=self.fc_dim, out_channel=layer_dims[-1])]
            for i in range(1,len(layer_dims)-1):
                layers.append(self.make_layer(in_channel=layer_dims[-i], out_channel=layer_dims[-(i+1)]))
        
            layers.append(self.make_layer(in_channel=layer_dims[1], out_channel=layer_dims[0], block=['basic', 'basic']))
        
        else:
            layers = [self.make_layer(in_channel=self.fc_dim, out_channel=layer_dims[0], block=['basic', 'basic'])]
        
        
        if maxpool:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        
        layers.append(nn.ConvTranspose2d(layer_dims[0], in_dim, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=1, bias=False))
        layers.append(nn.BatchNorm2d(in_dim))
        layers.append(nn.Tanh())
        
        self.decode_layer = nn.Sequential(*layers)
        

    def make_layer(self, in_channel, out_channel, block:list = ['upsample', 'basic']):
        layers = []
        
        upsampleBlock = InvertBlock(in_channel, out_channel, upsample=True)
        basicBlock = InvertBlock(out_channel, out_channel, upsample=False)
        
        for layer in block:
            if layer == 'upsample': 
                layers.append(upsampleBlock)
            elif layer == 'basic':
                layers.append(basicBlock)
            
        return nn.Sequential(*layers)
        
    
    def forward(self,x):
        x = self.rev_latent_layer(x)
        x = x.view(x.shape[0], self.fc_dim, 1, 1)
        x = self.rev_avgpool(x)
        #print(x.shape)
        x = self.decode_layer(x)
        return x
    
