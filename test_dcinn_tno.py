from PIL import Image
import numpy as np
import os
import torch

import glob

import time
import imageio

import torchvision.transforms as transforms
from thop import clever_format
from thop import profile
from torch.utils.data import DataLoader, Dataset

from model.dcinn_ivf import DCINN

from tqdm import tqdm

device = torch.device('cuda:0')


class GetDataset(Dataset):
    def __init__(self, ir_name_list, vi_name_list, transform=None):
        #ir_name_list.sort()
        #vi_name_list.sort()
        self.ir_name_list = ir_name_list
        self.vi_name_list = vi_name_list
        self.transform = transform

    def __getitem__(self, index):

        ir = self.ir_name_list[index]
        vi = self.vi_name_list[index]

        ir = Image.open(ir).convert('L')
        vi = Image.open(vi).convert('L')

        
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            ir = tran(ir)

            vi = tran(vi)


            return ir,vi

    def __len__(self):
        return len(self.ir_name_list)

training_dir_ir = "/home/wangwu/Test_TNO/ir1/*.bmp"
folder_dataset_train_ir = glob.glob(training_dir_ir)
    
training_dir_vi = "/home/wangwu/Test_TNO/vi1/*.bmp"
    
folder_dataset_train_vi = glob.glob(training_dir_vi)

transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])

dataset_test_dir = GetDataset(folder_dataset_train_ir,folder_dataset_train_vi,
                                                  transform=transform_train)
test_loader = DataLoader(dataset_test_dir,
                              shuffle=False,
                              batch_size=1)
# test inn

model = DCINN().to(device)
model_path = "./models/model_inn/model_tno.pth"
model.load_state_dict(torch.load(model_path))

def fusion():
    for i, (ir,vi)  in tqdm(enumerate(test_loader), total=len(test_loader)):
        

        ir = ir.to(device)
        vi = vi.to(device)

        
        model.eval()
        
        if ir.shape[-2]%8 != 0:
            new_h = ir.shape[-2] - ir.shape[-2]%8
            
            ir = ir[:,:,:new_h,:]
            vi = vi[:,:,:new_h,:]
        if ir.shape[-1]%8 != 0:
            new_w = ir.shape[-1] - ir.shape[-1]%8
            
            ir = ir[:,:,:,:new_w]
            vi = vi[:,:,:,:new_w]

        
            
        fused_detail, fused_base, ir_detail, vi_detail,m1,m2,m3  = model.forward(ir,vi)
        out = fused_detail+fused_base
        out = torch.clamp(out,0,1)
        d = np.squeeze(out.detach().cpu().numpy())

        result = (d* 255).astype(np.uint8)
        
        d = np.squeeze(ir.detach().cpu().numpy())
        ir = (d* 255).astype(np.uint8)

        d = np.squeeze(vi.detach().cpu().numpy())
        vi = (d* 255).astype(np.uint8)

        imageio.imwrite('./tno_result/{}.bmp'.format( i), result)
    
        d = np.squeeze(out.detach().cpu().numpy())

        result = (d* 255).astype(np.uint8)
        
        d = np.squeeze(ir.detach().cpu().numpy())
        ir = (d* 255).astype(np.uint8)

        d = np.squeeze(vi.detach().cpu().numpy())
        vi = (d* 255).astype(np.uint8)

        imageio.imwrite('./tno_result/{}.bmp'.format( i), result)


if __name__ == '__main__':

    fusion()
