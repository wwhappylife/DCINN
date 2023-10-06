from PIL import Image
import numpy as np
import os
import torch
import time
import imageio
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from model.dcinn_ivf import DCINN
from tqdm import tqdm

device = torch.device('cuda:0')


class GetDataset(Dataset):
    def __init__(self, train_dir_ir, ir_name_list, train_dir_vi, vi_name_list, transform=None):
        self.ir_name_list = ir_name_list
        self.vi_name_list = vi_name_list
        self.ir_dir = train_dir_ir
        self.vi_dir = train_dir_vi
        self.transform = transform

    def __getitem__(self, index):

        ir_name = self.ir_name_list[index]
        vi_name = self.vi_name_list[index]
        ir = Image.open(self.ir_dir+ir_name).convert('L')
        vi = Image.open(self.vi_dir+vi_name).convert('L')

        
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            ir = tran(ir)
            vi = tran(vi)

            return ir,vi,ir_name

    def __len__(self):
        return len(self.ir_name_list)

testing_dir_ir = "./testing_dataset/ivf/tno/ir/"
ir_name_list = os.listdir(testing_dir_ir)
    
testing_dir_vi = "./testing_dataset/ivf/tno/vi/"
vi_name_list = os.listdir(testing_dir_vi)

transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])

dataset_test_dir = GetDataset(testing_dir_ir, ir_name_list,testing_dir_vi, vi_name_list,
                                                  transform=transform_train)
test_loader = DataLoader(dataset_test_dir,
                              shuffle=False,
                              batch_size=1)
# test inn

model = DCINN().to(device)
model_path = "./pretrained/model_tno.pth"
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

def fusion():
    for i, (ir,vi,name)  in tqdm(enumerate(test_loader), total=len(test_loader)):
        

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
 
        out = model.forward(vi,ir,'Max')
        out = torch.clamp(out,0,1)
        d = np.squeeze(out.detach().cpu().numpy())
        result = (d* 255).astype(np.uint8)
        imageio.imwrite('./tno_result/'+name[0], result)

if __name__ == '__main__':

    fusion()
