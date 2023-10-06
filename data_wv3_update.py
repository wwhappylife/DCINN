import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np
import scipy.io as sio

class Dataset_Pro_h5(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro_h5, self).__init__()
        self.ms, self.pan, self.lms, self.gt = load_setmat(file_path)
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()
    def __len__(self):
        return self.gt.shape[0]

def load_setmat(file_path):
    data = h5py.File(file_path)
    lms1 = data['lms'][...] 
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)

    pan1 = data['pan'][...] 
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)

    ms1 = data['ms'][...]
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)

    gt1 = data['gt'][...]
    gt1 = np.array(gt1, dtype=np.float32) / 2047.0
    gt = torch.from_numpy(gt1)
    Nn, Wn, Hn, Cn = gt.shape
    return ms, pan, lms, gt

class Dataset_Pro_Eval_Full(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro_Eval_Full, self).__init__()
        self.ms, self.pan, self.lms = load_setmat_full(file_path)
    def __getitem__(self, index):
        return self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()
    def __len__(self):
        return self.lms.shape[0]

def load_setmat_full(file_path):
    data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  
    lms1 = np.array(lms1, dtype=np.float32)
    lms = torch.from_numpy(lms1).unsqueeze(0)
    pan1 = data['pan'][...]  
    pan1 = np.array(pan1, dtype=np.float32)
    pan = torch.from_numpy(pan1)
    pan = pan.unsqueeze(0).unsqueeze(0)
    ms1 = data['ms'][...]  
    ms1 = np.array(ms1, dtype=np.float32)
    ms = torch.from_numpy(ms1).unsqueeze(0)

    ms = ms.permute(0,3,1,2)
    lms = lms.permute(0,3,1,2)
    return ms, pan, lms
