# there is the sample to constrcu data generator

from torch.utils.data import Dataset,Sampler
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch
import os
import scipy.io as scio
import random
import numpy as np
import math

class addRandom_gnoise(object):
    """add Gaussian noise """
    def __init__(self,low_snr,high_snr):
        self.low_snr=low_snr
        self.high_snr=high_snr

    def __call__(self, singal):
        dim = len(singal.shape)
        if dim==1:
            singal = np.expand_dims(np.array(singal), axis=0) #(500,)->(1,500)

        if dim == 2:
            self.snr = np.random.choice(15, 1, p=[0.1, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
            self.snr += self.low_snr
            for i in range(singal.shape[0]):  # loop channel 
                noise = np.random.normal(size=singal.shape[-1])  #add Gaussian noise
                s_p = self.get_rms(singal[i])  # Signal power
                n_p = self.get_rms(noise)      # Noise power
                n_v = s_p / (10**(self.snr/10))  
                noise = noise * np.sqrt(n_v/n_p) 
                singal[i] = noise+singal[i]  
                singal[i] = (singal[i] - np.mean(singal[i])) / np.std(singal[i]) # normalize
        return singal

    def get_rms(self, x):
        return np.dot(x, x.T) / len(x)

class Signalset(Dataset):
    def __init__(self,data_dir,transform=None,flag="training"):
        """make sure only .mat file in data_dir document """
        self.dir=data_dir
        self.transform = transform
        self.flag = flag
        if not os.path.exists(self.dir):
            self.dir = None
            raise FileNotFoundError
        assert flag in ["training","inference"],"make sure 'flag'='training' or 'inference' ."
        if flag=="inference":
            self.transform = None

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("idx:{}",idx)
        try:
            data_name = os.path.join(self.dir, "dat" + str(idx + 1) + ".mat")  # idx start with zero
            dat = scio.loadmat(data_name)
        except FileNotFoundError:
            data_name = os.path.join(self.dir, str(idx + 1) + ".mat")  # idx start with zero
            dat = scio.loadmat(data_name)
        dat = dat['dat']
        singal = dat[0,0]['singal']#（6,6）
        if self.transform is not None:
            singal = self.transform(singal)
        if self.flag == "training":
            label = dat[0,0]['As']
            index = np.where(label>0.995)
            label[index]=0.995
            label = label.T 
            return torch.FloatTensor(singal),torch.FloatTensor(label)
        return torch.FloatTensor(singal)

    def __len__(self):
        count = 0
        if self.dir is not None:
            file_names = os.listdir(self.dir)
            for name in file_names:
                if name.endswith("mat"):
                    count += 1
        return count

