import os
import argparse

from tqdm import tqdm
import pandas as pd

import glob

from collections import OrderedDict
import torch
import joblib
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model.dcinn_ivf import DCINN


import random
from random import randrange
from vgg import Vgg16
from loss import vgg_loss
from pytorch_ssim import ssim,tv_loss
from losses import ssim_loss_ir,ssim_loss_vi , sf_loss_ir, sf_loss_vi, z_loss

device = torch.device('cuda:1')


l1_loss = torch.nn.L1Loss().to(device)
vgg = Vgg16(requires_grad = False).to(device)
pc_loss = vgg_loss().to(device)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model_inn', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--gamma', default=0.5, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', '--learning-rate', default=3e-5, type=float)
    parser.add_argument('--weight', default=[1,0.05,0.0006, 0.00025], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)

    args = parser.parse_args()

    return args


def get_patch(img_in, img_tar, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))

    return img_in, img_tar

class GetDataset(Dataset):
    def __init__(self, ir_name_list, vi_name_list, transform=None):
        ir_name_list.sort()
        vi_name_list.sort()
        self.ir_name_list = ir_name_list
        self.vi_name_list = vi_name_list
        self.transform = transform

    def __getitem__(self, index):

        ir = self.ir_name_list[index]
        vi = self.vi_name_list[index]

        ir = Image.open(ir).convert('L')
        vi = Image.open(vi).convert('L')

        ir, vi = get_patch(ir, vi, patch_size=152)
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            ir = tran(ir)

            vi = tran(vi)


            return ir,vi

    def __len__(self):
        return len(self.ir_name_list)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader_ir,train_loader_vi, model, criterion_ssim_ir,  criterion_ssim_vi, criterion_sf_ir,criterion_sf_vi,optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    losses_ssim_ir = AverageMeter()
    losses_ssim_vi = AverageMeter()
    losses_sf_ir = AverageMeter()
    losses_sf_vi = AverageMeter()
    losses_vi_back = AverageMeter()
    losses_ir_back = AverageMeter()
    weight = args.weight
    model.train()

    for i, (ir,vi)  in tqdm(enumerate(train_loader_ir), total=len(train_loader_ir)):

        ir = ir.to(device)
        vi = vi.to(device)

        
        fused_detail, fused_base, ir_detail, vi_detail,m1,m2,m3 = model.forward(ir,vi)
        recon_detail = model.reverse(fused_detail,m1,m2,m3)
        out = fused_detail+fused_base
        l_r =  0.1*criterion_sf_vi(fused_detail, vi_detail)# + 2*criterion_ssim_ir(fused_detail, ir_detail)# + 1e1*criterion_ssim_ir(fused_detail, ir_detail)
        
        loss_vi_back = criterion_ssim_ir(recon_detail,fused_detail)
        
        loss_ir_back = loss_vi_back
        
        loss_ssim_ir = 0.6*criterion_ssim_ir(out,ir)
        loss_ssim_vi=  criterion_ssim_vi(out,vi)
        loss_sf_ir= 0.0002* criterion_sf_ir(out, ir) 
        loss_sf_vi= 0.0002* criterion_sf_vi(out, vi) 
        loss = loss_ssim_ir + loss_ssim_vi + loss_sf_ir + loss_sf_vi# + 0.1*loss_vi_back#  + l_r
        
        losses.update(loss.item(), ir.size(0))
        losses_ssim_ir.update(loss_ssim_ir.item(), ir.size(0))
        losses_ssim_vi.update(loss_ssim_vi.item(), ir.size(0))
        losses_sf_ir.update(loss_sf_ir.item(), ir.size(0))
        losses_sf_vi.update(loss_sf_vi.item(), ir.size(0))
        losses_vi_back.update(loss_vi_back.item(), ir.size(0))
        losses_ir_back.update(loss_ir_back.item(), ir.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2e-4, norm_type=2)
        optimizer.step()
        print('IR Back Loss: {:.2e} || VI Back Loss: {:.2e}'.format(loss_ir_back.item(),loss_vi_back.item()))

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_ssim_ir', losses_ssim_ir.avg),
        ('loss_ssim_vi', losses_ssim_vi.avg),
        ('loss_sf_ir', losses_sf_ir.avg),
        ('loss_sf_vi', losses_sf_vi.avg),
        ('loss_vi_back', losses_vi_back.avg),
        ('loss_ir_back', losses_ir_back.avg),
    ])
    return log



def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True


    training_dir_ir = "/home/wangwu/vin_fusion/RoadScene-master/train/infrared/*.jpg"
    folder_dataset_train_ir = glob.glob(training_dir_ir)
    
    training_dir_vi = "/home/wangwu/vin_fusion/RoadScene-master/train/visible/*.jpg"
    
    folder_dataset_train_vi = glob.glob(training_dir_vi)
    
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])

    dataset_train_ir = GetDataset(folder_dataset_train_ir,folder_dataset_train_vi,
                                                  transform=transform_train)
    dataset_train_vi = GetDataset(folder_dataset_train_vi,folder_dataset_train_vi,
                                  transform=transform_train)

    train_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              batch_size=args.batch_size)
    train_loader_vi = DataLoader(dataset_train_vi,
                                 shuffle=True,
                                 batch_size=args.batch_size)
    model = DCINN().to(device)
    
    
    criterion_ssim_ir = ssim_loss_ir
    criterion_ssim_vi = ssim_loss_vi
    criterion_sf_ir = sf_loss_ir
    criterion_sf_vi= sf_loss_vi

    milestones = []
    for i in range(1, args.epochs+1):
        if i == 50:
            milestones.append(i)
        if i == 100:
            milestones.append(i)
        if i == 200:
            milestones.append(i)
        if i == 300:
            milestones.append(i)
    

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    scheduler_f = lrs.MultiStepLR(optimizer, milestones, args.gamma)
    log = pd.DataFrame(index=[],
                       columns=['epoch',

                                'loss',
                                'loss_ssim_ir',
                                'loss_ssim_vi',
                                'loss_sf_ir',
                                'loss_sf_vi',
                                'loss_vi_back',
                                'loss_ir_back',
                                ])

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))

        train_log = train(args, train_loader_ir,train_loader_vi, model, criterion_ssim_ir,  criterion_ssim_vi, criterion_sf_ir,  criterion_sf_vi, optimizer, epoch)     # 训练集

        print('loss: %.4f - loss_ssim_ir: %.4f - loss_ssim_vi: %.4f - loss_sf_ir: %.4f- loss_sf_vi: %.4f loss_vi_back: %.4f loss_ir_back: %.4f'
              % (train_log['loss'],
                 train_log['loss_ssim_ir'],
                 train_log['loss_ssim_vi'],
                 train_log['loss_sf_ir'],
                 train_log['loss_sf_vi'],
                 train_log['loss_vi_back'],
                 train_log['loss_ir_back'],
                 ))

        tmp = pd.Series([
            epoch + 1,

            train_log['loss'],
            train_log['loss_ssim_ir'],
            train_log['loss_ssim_vi'],
            train_log['loss_sf_ir'],
            train_log['loss_sf_vi'],
            train_log['loss_vi_back'],
            train_log['loss_ir_back'],
        ], index=['epoch', 'loss', 'loss_ssim_ir', 'loss_ssim_vi', 'loss_sf_ir', 'loss_sf_vi', 'loss_vi_back', 'loss_ir_back'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        scheduler_f.step()
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)


if __name__ == '__main__':
    main()


