import torch.utils.data as data
import torch
import h5py
import numpy as np
import scipy.io as sio

class Dataset_Pro_h5(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro_h5, self).__init__()
        self.ms, self.hs, self.gt = load_setmat(file_path)
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.hs[index, :, :, :].float()
    def __len__(self):
        return self.gt.shape[0]

def load_setmat(file_path):
    data = sio.loadmat(file_path)
    hs1 = data['hs'][...] 
    hs1 = np.array(hs1, dtype=np.float32)
    hs = torch.from_numpy(hs1)

    ms1 = data['ms'][...]
    ms1 = np.array(ms1, dtype=np.float32)
    ms = torch.from_numpy(ms1)

    gt1 = data['gt'][...]
    gt1 = np.array(gt1, dtype=np.float32)
    gt = torch.from_numpy(gt1)

    hs = hs.permute(2,0,1).unsqueeze(0)
    ms = ms.permute(2,0,1).unsqueeze(0)
    gt= gt.permute(2,0,1).unsqueeze(0)
    return ms, hs, gt

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path, 'r')
        print(dataset.keys())
        self.GT = dataset.get("GT")
        self.UP = dataset.get("HSI_up")
        self.LRHSI = dataset.get("LRHSI")
        self.RGB = dataset.get("RGB")

    def __getitem__(self, index):
        return {'gt': torch.from_numpy(self.GT[index, :, :, :]).float(),
                'up': torch.from_numpy(self.UP[index, :, :, :]).float(),
                'lrhsi': torch.from_numpy(self.LRHSI[index, :, :, :]).float(),
                'rgb': torch.from_numpy(self.RGB[index, :, :, :]).float()}

    def __len__(self):
        return self.GT.shape[0]
