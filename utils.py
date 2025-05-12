# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:44:16 2025

@author: Eva Bauer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import torch
import os
import csv
import scipy
from scipy.signal import butter, filtfilt

import io


Leg_len = {'Sub8':0.91,
           'Sub9':0.82,
           'Sub10':0.95,
           'Sub13':0.87,
           'Sub14':0.87,
           'Sub15':0.78,
           'Sub17':0.84,
           'Sub18':0.93}

# Markerverbindungen zum Plotten der Messung
MARKER_LOC = {'r_foot':[[9,10],[12,9],[9,11],[11,10],[10,12],[11,12]], 
              'r_leg':[[9,8],[12,8],[10,8],[8,7],[6,7],[7,5],[5,6],[7,4]],
              'hip':[[6,4],[4,5],[4,0],[4,1],[0,1],[1,2],[2,3],[3,0],[0,2],[3,1],[13,3],[13,2]],
              'l_foot':[[19,20],[20,18],[18,21],[19,21],[20,21],[19,18]],
              'l_leg':[[18,17],[19,17],[21,17],[17,16],[16,14],[16,15],[14,15],[16,13],[15,13],[14,13]],
              'torso':[[0,23],[1,22],[2,24],[3,23],[23,24],[23,22], [22,24]],
              'r_arm':[[25,26],[26,27],[27,28],[25,22]],
              'l_arm':[[29,30],[30,31],[29,24]]}

Marker_Label = ['RASI','RPSI','LPSI','LASI','RHIPExt','RLKN','RMKN','RTIB','RTIBExt2','RLM','RMM','RTOE','RCAL','LHIPExt','LLKN','LMKN','LTIB','LTIBExt2','LLM','LMM','LTOE','LCAL','RSter','MidSterUp','LSter','RArmElbow','RArmElbowExt','RArmWristExt','RArmWrist','LArmElbow','LArmMid','LArmWrist']
DEL_MARKER = ['Sub Frame', 'MidSterDown_X', 'MidSterDown_Y', 'MidSterDown_Z', 'RHIP_X', 'RHIP_Y', 'RHIP_Z', 'LHIP_X', 'LHIP_Y', 'LHIP_Z', 'LTIBExt1_X', 'LTIBExt1_Y', 'LTIBExt1_Z', 'RTIBExt1_X', 'RTIBExt1_Y', 'RTIBExt1_Z']
    
check_col = ['Frame', 'RASI_X', 'RASI_Y', 'RASI_Z', 'RPSI_X', 'RPSI_Y', 'RPSI_Z', 'LPSI_X', 'LPSI_Y', 'LPSI_Z', 'LASI_X', 'LASI_Y', 'LASI_Z', 'RHIPExt_X', 'RHIPExt_Y', 'RHIPExt_Z', 'RLKN_X', 'RLKN_Y', 'RLKN_Z', 'RMKN_X', 'RMKN_Y', 'RMKN_Z', 'RTIB_X', 'RTIB_Y', 'RTIB_Z', 'RTIBExt2_X', 'RTIBExt2_Y', 'RTIBExt2_Z', 'RLM_X', 'RLM_Y', 'RLM_Z', 'RMM_X', 'RMM_Y', 'RMM_Z', 'RTOE_X', 'RTOE_Y', 'RTOE_Z', 'RCAL_X', 'RCAL_Y', 'RCAL_Z', 'LHIPExt_X', 'LHIPExt_Y', 'LHIPExt_Z', 'LLKN_X', 'LLKN_Y', 'LLKN_Z', 'LMKN_X', 'LMKN_Y', 'LMKN_Z', 'LTIB_X', 'LTIB_Y', 'LTIB_Z', 'LTIBExt2_X', 'LTIBExt2_Y', 'LTIBExt2_Z', 'LLM_X', 'LLM_Y', 'LLM_Z', 'LMM_X', 'LMM_Y', 'LMM_Z', 'LTOE_X', 'LTOE_Y', 'LTOE_Z', 'LCAL_X', 'LCAL_Y', 'LCAL_Z', 'RSter_X', 'RSter_Y', 'RSter_Z', 'MidSterUp_X', 'MidSterUp_Y', 'MidSterUp_Z', 'LSter_X', 'LSter_Y', 'LSter_Z', 'RArmElbow_X', 'RArmElbow_Y', 'RArmElbow_Z', 'RArmElbowExt_X', 'RArmElbowExt_Y', 'RArmElbowExt_Z', 'RArmWristExt_X', 'RArmWristExt_Y', 'RArmWristExt_Z', 'RArmWrist_X', 'RArmWrist_Y', 'RArmWrist_Z', 'LArmElbow_X', 'LArmElbow_Y', 'LArmElbow_Z', 'LArmMid_X', 'LArmMid_Y', 'LArmMid_Z', 'LArmWrist_X', 'LArmWrist_Y', 'LArmWrist_Z']

#Header einlesen u. erstellen 
def get_column(path):
    markers = pd.read_csv(path, sep=',', header=None, skiprows=2, nrows=2).to_numpy()
    name_markers = [markers[1,0], markers[1,1]]

    for idx in range(2,len(markers[0])):
        if pd.isna(markers[1,idx]):
            continue
        
        if pd.notna(markers[0,idx]):
            name = markers[0,idx].split(':')[1]
        
        name_markers.append(name + '_' + markers[1,idx])
        
    return name_markers


# Zeitliche normalisierung ein Schritt über 128 Frames
def time_norm(sequence, target_len=128):
    norm_sequence = pd.DataFrame(columns=sequence.columns)
    for spalte in sequence.columns:
        target = np.linspace(0, 1, target_len)
        origin = np.linspace(0, 1, len(sequence[spalte]))
        interpolate = scipy.interpolate.interp1d(origin, sequence[spalte], kind='linear')
        norm_sequence[spalte] = interpolate(target)
    return norm_sequence


# Dataframe zu x,y,z numpy Array
def df_2_np(df, koordinate):
    columns = [col for col in df.columns if col.endswith(koordinate)]
    return df[columns].values


#Glätten der Daten mit butterworth (ChatGPT-11.02.25)
def zero_lag_butter(df, fs=100.0, cutoff=6.0, order=4):
    nyq = 0.5 * fs
   
   # Frequenz normalisieren
    normal_cutoff = cutoff/nyq 
   
   # Butterworth-Filterkoeffizienten (Tiefpass)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
   # Zero-Lag-Filterung mit filtfilt
    for col in df.columns:
        df[col] = filtfilt(b, a, df[col], axis=0)
    
    return df


def normalise(tensor):
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    tensor_norm = 2*((tensor - min_val)/(max_val - min_val))-1
    return tensor_norm

# sucht nach Max aller Messungen um zwischen -1:1 zu normalisieren (nur einmal, dann speichert den Wert)
def calc_max_abs_scale():
    path = os.path.join(os.getcwd(), 'Data', 'human_walking_dataset')
    max_val = float('-inf')
    
    for sub in os.listdir(path):
        print(sub, 'normal')
        sub_path = os.path.join(path, sub, 'Kinematics','normal')
        
        for file in os.listdir(sub_path):
            filepath = os.path.join(sub_path, file)
            print(file)
            tensor_list = get_sequenceList(filepath, sub=sub, time_norm_len=128, max_abs_scale=None)
            if len(tensor_list) == 0:
                continue
            tensor_list = torch.stack(tensor_list, dim=0)
        
            temp_max = torch.max(abs(tensor_list))
            if temp_max > max_val:
                max_val = temp_max
        
        print(sub, 'orth')
        sub_path = os.path.join(path, sub, 'Kinematics','orthosis')
        for file in os.listdir(sub_path):
            filepath = os.path.join(sub_path, file)
            print(file)
            tensor_list = get_sequenceList(filepath, sub=sub, time_norm_len=128, max_abs_scale=None)
            if len(tensor_list) == 0:
                continue
            tensor_list = torch.stack(tensor_list, dim=0)
        
            temp_max = torch.max(abs(tensor_list))
            if temp_max > max_val:
                max_val = temp_max
            
    max_abs_scale = float(max_val)
    np.save(os.path.join(os.getcwd(), 'Data', 'max_abs_scale_hwd'), max_abs_scale)
    return 


# Sequenzen für DataLoader finden (Output: Tensorliste)
def get_sequenceList(path, sub, time_norm_len=128, max_abs_scale=None):
    tensor_list = []
    
    # Einlesen der Daten
    col = get_column(path)
    data = pd.read_csv(path, sep=',', names=col, skiprows=5)
    data = data.drop(columns=DEL_MARKER, errors='ignore')
    
    assert list(data.columns) == check_col
  
    data_smooth = zero_lag_butter(data)
    
    # suchen nach dem Peaks, mit min. Distanz von 80 -> (80 hat Lars gewählt)
    steps,_ = scipy.signal.find_peaks(-1*data_smooth['LLM_Y'], distance=80)
    
    
    
    # Unterteilen der Daten in Schritte - Output Liste aus Tensoren [Frames(100), Marker(50), xyz(3)] (einen pro Schritt)
    # letzter TD wird nicht mehr gewertet
    for i in range(len(steps)-2):
        step = data_smooth[steps[i]:steps[i+1]]
        
        # Ausschließen von Schritten in denen Marker nicht erkannt wurden
        if step.isna().values.any():
            continue
        
        # Zeitliche normalisierung des Schrittes 
        norm_step = time_norm(step, time_norm_len)
        # finden der Raum-Normalisierungs-Koordinate (Mittelpunkt aus LASI, RASI, LPSI, RPSI)
        X_norm = norm_step[['LASI_X','RASI_X','RPSI_X','LPSI_X']].mean(axis=1).to_numpy()
        Y_norm = norm_step[['LASI_Y','RASI_Y','RPSI_Y','LPSI_Y']].mean(axis=1).to_numpy()
        Z_norm = norm_step[['LASI_Z','RASI_Z','RPSI_Z','LPSI_Z']].mean(axis=1).to_numpy()
        
        #print(X_norm, Y_norm, Z_norm)
        
        # Zentralisieren der Koordinaten um Zentralen Hüftpunkt (Mittelpunkt aus LASI, RASI, LPSI, RPSI)
        x_step = df_2_np(norm_step, 'X')
        x_step_cent = x_step - X_norm[:, np.newaxis]
        # x_step = torch.Tensor(x_step)
        # x_step_norm = torch.Tensor(x_step - X_norm[:, np.newaxis])
        
        y_step = df_2_np(norm_step, 'Y')
        y_step_cent = y_step - Y_norm[:, np.newaxis]
        # y_step = torch.Tensor(y_step)
        # y_step_norm = torch.Tensor(y_step - Y_norm[:, np.newaxis])
        
        z_step = df_2_np(norm_step, 'Z')
        z_step_cent = z_step - Z_norm[:, np.newaxis]
        # z_step = torch.Tensor(z_step)
        # z_step_norm = torch.Tensor(z_step - Z_norm[:, np.newaxis])
        
        body_scale = np.full((time_norm_len,1), Leg_len[sub])
        
        x_step_cent /= body_scale
        y_step_cent /= body_scale
        z_step_cent /= body_scale
        
        if max_abs_scale is not None:
            x_step_cent /= max_abs_scale
            y_step_cent /= max_abs_scale
            z_step_cent /= max_abs_scale
        else:
            print('Tensor nicht normalisiert!')
        
        # Umwandeln  in Tensor
        x_step = torch.Tensor(x_step_cent)
        y_step = torch.Tensor(y_step_cent)
        z_step = torch.Tensor(z_step_cent)
        
        # [Frames, Marker, xyz]
        tensor_step = torch.stack((x_step, y_step, z_step), dim=2)
        
        assert tensor_step.size(dim=1) == 32
        
        
        #tensor_step = normalise(tensor_step)
        
        tensor_list.append(tensor_step)
    
    return tensor_list

# Histogramm für Plot von Trainingsergebnissen 
def gen_histogramm(error_normal, error_patho):
    plt.ioff()
    
    plt.figure(figsize=(8, 5))

    plt.hist(error_normal, bins=20, alpha=0.5, label='Normal', color='blue')
    plt.hist(error_patho, bins=20, alpha=0.5, label='Gestört', color='red')

    plt.xlabel('Rekonstruktionsfehler')
    plt.ylabel('Häufigkeit')
    plt.title('Verteilung der Rekonstruktionsfehler (Normal vs. Gestört)')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    
    return buf


# Sichern des trainierten Models
def save_model(epoch, model, optimizer, SAVE_PATH):
    torch.save({'epoch': epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()}, SAVE_PATH)
    print('Model saved!')
    return 


# Reset der Werte des Models
def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


# Auflisten von x,y,z für Schritt (z.B. vor Plotten)
def list_xyz(frame):
    x=[]
    y=[]
    z=[]
    
    for marker in frame:
        x.append(float(marker[0]))
        y.append(float(marker[1]))
        z.append(float(marker[2]))
        
    return x,y,z

# Berechnet Winkel zwischen Vektoren
def calc_joint_angle(v1,v2):
    dot_product = (v1 * v2).sum(dim=1)
    
    norm_v1 = v1.norm(dim=1)
    norm_v2 = v2.norm(dim=1)

    
    cos_angle = dot_product / (norm_v1 * norm_v2 + 1e-8)  
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    
    angle_rad = torch.acos(cos_angle)
    angle_deg = torch.rad2deg(angle_rad)

    return angle_deg

# Berechnet das Physikalische Loss
def phy_loss(recon_batch):
    COB_r = recon_batch[:, :, [4,5,6,7,8,9,10,11,12], :].mean(dim=2)
    COB_l = recon_batch[:, :, [13,14,15,16,17,18,19,20,21], :].mean(dim=2)
    
    G = torch.tensor([0.0, 0.0, -1.0], device=recon_batch.device).view(1,3,1)
    
    angle_r = calc_joint_angle(COB_r, G) 
    angle_l = calc_joint_angle(COB_l, G)
    
    angle_r_shifted = torch.roll(angle_r, shifts=angle_r.size(1)//2, dims=1)
    
    dif = (angle_l - angle_r_shifted)**2 
    
    return dif.mean()

# Input (B, C, N)
def close_loop(Mitte):
    N = Mitte.shape[2]
    T = N - 1 
    device = Mitte.device
    
    D = Mitte[:,:,-1] - Mitte[:,:,0]
    
    t = torch.arange(N, dtype=torch.float32, device=device)
    
    ratio = t / T
    
    ratio = ratio.unsqueeze(dim=0)
    
    D_3d = D.unsqueeze(dim=-1)
    shift = D_3d * ratio
    
    return Mitte - shift

def gen_histogramm(error_normal, error_patho):
    plt.ioff()
    
    plt.figure(figsize=(8, 5))

    plt.hist(error_normal, bins=20, alpha=0.5, label='Normal', color='blue')
    plt.hist(error_patho, bins=20, alpha=0.5, label='Gestört', color='red')

    plt.xlabel('Rekonstruktionsfehler')
    plt.ylabel('Häufigkeit')
    plt.title('Verteilung der Rekonstruktionsfehler (Normal vs. Gestört)')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    
    return buf

def gen_Liss_Kurve(data, liss, recon):
    plt.ioff()
    
    fig = plt.figure(figsize=(6,10))
    ax = plt.subplot(projection='3d')
    plt.plot(data[0].cpu().numpy(), data[1].cpu().numpy(), data[2].cpu().numpy(), c='green', label='Mitte Daten')
    plt.plot(liss[0].cpu().numpy(), liss[1].cpu().numpy(), liss[2].cpu().numpy(), c='red', ls='--', label='Lissajous-Kurve')
    plt.plot(recon[0].cpu().numpy(), recon[1].cpu().numpy(), recon[2].cpu().numpy(), c='blue', ls='--', label='Mitte Recon.')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()

    return buf

# Animation der Koordinaten als gif 
def model_animation(input_step, fig, ax, color=[['model','blue','green']], COB=[]):
    if type(input_step) is not list:
        input_step = [input_step]
        

    x_all = []
    y_all = []
    z_all = []
    
    for tensor in input_step:
        for frame in tensor:
            x,y,z = list_xyz(frame)
            x_all += x
            y_all += y
            z_all += z

    x_dif = abs(np.max(x_all)) + abs(np.min(x_all))
    y_dif = abs(np.max(y_all)) + abs(np.min(y_all))
    z_dif = abs(np.max(z_all)) + abs(np.min(z_all))
    
    ax.set_box_aspect([x_dif, y_dif, z_dif])

    def animate(i):
        ax.clear()
        
        for idx,tensor in enumerate(input_step):    
            disc = color[idx]
            frame = tensor[i]
            
            x,y,z = list_xyz(frame)
            
            ax.scatter(x,y,z, label=disc[0], c=disc[1])
            
            if len(COB) == 2:
                ax.scatter(COB[0][i][0], COB[0][i][1], COB[0][i][2], c='red')
                ax.scatter(COB[1][i][0], COB[1][i][1], COB[1][i][2], c='red')
                
                ax.plot([COB[0][i][0], 0.0], [COB[0][i][1],0.0], [COB[0][i][2],0.0], c='red')
                ax.plot([COB[1][i][0], 0.0], [COB[1][i][1],0.0], [COB[1][i][2],0.0], c='red')
                
                ax.plot([0.0,0.0],[0.0,0.0],[0.0,-1.0], c='red')
                ax.scatter(0.0,0.0,0.0, c='red')
            
            
            ax.set_xlim(np.min(x_all), np.max(x_all))
            ax.set_ylim(np.min(y_all), np.max(y_all))
            ax.set_zlim(np.min(z_all), np.max(z_all))

            for _,part in MARKER_LOC.items():
                for con in part:
                    lx = [x[con[1]], x[con[0]]]
                    ly = [y[con[1]], y[con[0]]]
                    lz = [z[con[1]], z[con[0]]]
                    ax.plot(lx,ly,lz, color=disc[2], linewidth=1)
        ax.legend()
        
    
    ani = FuncAnimation(fig, animate, frames=100, interval=20)
    
    return ani


def plot_pseudo(input_tensor:list, label:list, color:list, save=None, fig_num=1):
    
    plt.figure(num=fig_num, figsize=(25,10), dpi=100)
    for idx,tensor in enumerate(input_tensor):
        plt.subplot(1,len(input_tensor),idx+1)
        if color[idx]:
            plt.imshow(tensor, aspect='auto')
        else:
            plt.imshow(tensor, cmap='gray_r', vmin=0, vmax=0.2, aspect='auto')
        
        plt.colorbar()
        plt.yticks(ticks=range(len(Marker_Label)), labels=Marker_Label)
        plt.title(label[idx])
        
        if idx == 0:
            plt.xlabel('Frames')
            plt.ylabel('Marker')
    
    if save is not None:
        plt.tight_layout()
        plt.savefig(save, dpi=500)
    plt.show()