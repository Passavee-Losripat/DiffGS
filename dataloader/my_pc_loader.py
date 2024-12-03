#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from . import base 
from tqdm import tqdm
import numpy as np
import pandas as pd 
import csv

class MyPCloader(base.Dataset):

    def __init__(
        self,
        data_path
    ):

        self.gaussian_np = np.load(data_path)
        self.gaussian_np = self.gaussian_np[:,:,:59]
        self.gaussian_torch = []
        for i in range(self.gaussian_np.shape[0]):
            gaussian_idx = self.gaussian_np[i]
            gaussian_idx[:,51] = np.exp(gaussian_idx[:,51]) / (1 + np.exp(gaussian_idx[:,51]))
            gaussian_idx[:,52:55] = np.exp(gaussian_idx[:,52:55])
            self.gaussian_torch.append(torch.from_numpy(gaussian_idx))


    
    def __getitem__(self, idx): 
        gaussian = self.gaussian_torch[idx]
        pc = gaussian[:,:3]
        return pc


    def __len__(self):
        return len(self.gaussian_torch)



    
