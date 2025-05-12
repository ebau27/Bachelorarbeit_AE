# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:24:56 2024

@author: Eva Bauer
"""
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm

import utils as ult

"""
Für Labeled Data
0 : normal 
1 : patho
"""

class Loader(Dataset):
    def __init__(self, path, condition='normal', split='train', unknown=False, time_norm_len=128):
        
        self.data = []
        self.label = []
        
        
        max_abs_scale = np.load(os.path.join(os.getcwd(), 'Data', 'max_abs_scale_hwd.npy'))
        
        
        for sub in os.listdir(path):
            print(sub)
            
            # Sub8 für Test Netzwerk auf unbekannte Person (später)
            if unknown: 
                if sub != 'Sub8':
                    print('Testperson 8')
                    continue
            else:
                if sub == 'Sub8':
                    continue
            
            sub_path = os.path.join(path, sub, 'Kinematics', condition)
            progress_bar = tqdm(os.listdir(sub_path), total=len(os.listdir(sub_path)))
            for file in progress_bar:
                
                filepath = os.path.join(sub_path, file)
                sequenceList = ult.get_sequenceList(filepath, sub=sub, time_norm_len=time_norm_len, max_abs_scale=max_abs_scale)
                
                
                # erste zwei Schritte nicht nutzen für Test split
                if len(sequenceList) > 0:
                    if split == 'train':
                        sequenceList = sequenceList[2:]
                    elif split == 'test':
                        sequenceList = sequenceList[:2]
                    
                if condition == 'orthosis':
                    con_label = 1
                else:
                    con_label = 0
                    
                self.data += sequenceList
                self.label += [torch.tensor(con_label) for _ in range(len(sequenceList))]
                

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    


