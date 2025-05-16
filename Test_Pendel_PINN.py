# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:46:29 2024

PINN für Pendel Simulation
Orientiert an Experiment von https://swarnadeepseth.github.io/pendulum_PINN.html

@author: Eva Bauer 
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

g = 9.81
l = 0.025
theta_0 = np.pi/4

## Daten für kleine theta
def pendel_schwingung(t):
    return theta_0*np.cos(np.sqrt(g/l)*t)

time = torch.linspace(0, 1, 500, dtype=torch.float32).reshape(-1,1)
theta = pendel_schwingung(time)

x_physics = torch.linspace(0,1,30, dtype=torch.float32).reshape(-1,1).requires_grad_(True)



## Trainingsdaten  
x_data = time[0:200:20]
y_data = theta[0:200:20]

## Fully connected neural network
class NN(nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM):
        super().__init__()

        self.fc_ih = nn.Sequential(
                        nn.Linear(INPUT_DIM, HIDDEN_DIM),
                        nn.Tanh())
        
        self.fc_hh = nn.Sequential(
                        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                        nn.ReLU()) 
        
        self.fc_ho = nn.Sequential(
                        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
                        nn.Tanh())

    def forward(self, x):
        x_hidden = self.fc_ih(x)
        x_hidden = self.fc_hh(x_hidden)
        x_out = self.fc_ho(x_hidden)
        return x_out

## Training Visualisieren
def save_gif_PIL(outfile, files, fps=5, loop=0):
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def plot_result(time, theta, pred_theta, x_data, y_data, step, loss, show=False):
    plt.figure(figsize=(8,4))
    plt.plot(time, theta, c='grey', label='Theta, small Angle')
    plt.plot(time, pred_theta, c='blue', label='Theta')
    plt.scatter(x_data, y_data, c='orange', label='Training Data')
    plt.legend(loc=(1.01, 0.34), frameon=False)
    plt.title(f'Training step: {step}, Loss: {loss:.3f}')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)
    if show:
        plt.show()

## Training 
model = NN(INPUT_DIM=1, OUTPUT_DIM=1, HIDDEN_DIM=36)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
STEPS = 20000

loss_list = []
files = []
progress_bar = tqdm(range(STEPS), total=STEPS)
for i in progress_bar:
    
    ## Naive loss
    pred_y = model(x_data)
    naive_loss = torch.mean((pred_y - y_data)**2)
    
    ## Physics loss
    pred_y_phy = model(x_physics)
    dx = torch.autograd.grad(pred_y_phy, x_physics, torch.ones_like(pred_y_phy), create_graph=True)[0]
    dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0]
    
    residual = dx2 + (g / l) * torch.sin(pred_y_phy)
    phy_loss = torch.mean(residual**2)
    
    loss = naive_loss + 1e-3*phy_loss
    loss_list.append(loss.detach().numpy())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if ((i+1) % 100 == 0) and (i>10000):
        pred_theta = model(time).detach()
        
        plot_result(time, theta, pred_theta, x_data, y_data, i+1, loss, show=False)
        file = f'Plots/pinn{i+1}.8i.png'
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor='white')
        plt.close()
        
        files.append(file)
        
save_gif_PIL('pendel_pinn.gif', files)
        
    
pred_y = model(time).detach()
plot_result(time, theta, pred_y, x_data, y_data, i+1, loss_list[-1], show=True)


