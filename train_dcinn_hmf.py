

import os
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.dcinn_hmf import DCINN,read_r, read_R
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim.lr_scheduler as lrs
from dataUPHSI_D700 import DatasetFromHdf5
import socket
from utils import cal_psnr, print_network, compute_ssim, compute_ergas, compute_sam
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--Warm_Epochs', type=int, default=1200, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--D_kernel_size', type=int, default=14, help='Starting Epoch')
parser.add_argument('--num_spectral', type=int, default=93, help='Starting Epoch')
parser.add_argument('--num_channel', type=int, default=4, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=4e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--warm-lr', type=float, default=1.25e-5, help='Learning Rate. Default=0.01')
parser.add_argument('--skip_threshold', type=float, default=1e6, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='50', help='learning rate decay type')
parser.add_argument('--decay1', type=int, default='100', help='learning rate decay type')
parser.add_argument('--decay2', type=int, default='200', help='learning rate decay type')
parser.add_argument('--decay3', type=int, default='300', help='learning rate decay type')
parser.add_argument('--warm_decay', type=int, default='300', help='learning rate decay type')
parser.add_argument('--warm_gamma', type=float, default=2, help='learning rate decay factor for step decay')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--train_dataset', type=str, default='/home/wangwu/HIF/data/train_harvard(with_up)x4_rgb.h5')
parser.add_argument('--test_dataset', type=str, default='/home/wangwu/HIF/data/test_harvard(with_up)x4_rgb.h5')
parser.add_argument('--model_type', type=str, default='IBP2')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=64, help='Size of cropped HR image')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')


opt = parser.parse_args()
device = torch.device(opt.device)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)



def train(epoch):
    epoch_loss = 0
    for _, batch in enumerate(training_data_loader, 1):
        HSI,MS,HS,_ = Variable(batch['gt']), Variable(batch['lrhsi']), Variable(batch['rgb']), Variable(batch['up'])
        if cuda:
            MS = MS.to(device)
            HS = HS.to(device)
            HSI = HSI.to(device)

        f_model.train()
        optimizer_f.zero_grad()
        MS0 = MS
        HS0 = HS
        
        MS = torch.nn.functional.interpolate(MS, scale_factor=4, mode='bilinear') # MS
        r = read_r().to(device)
        B,C,H,W = HS.shape
        HS1 = rearrange(HS, 'b c h w -> b (h w) c')
        HS1 = torch.matmul(HS1, r)
        HS1 = rearrange(HS1, 'b (h w) c -> b c h w', h=H,w=W)
        out_HSI = f_model(HS1-MS, MS0, HS0) + MS
        
        HS_l1 = l1(out_HSI, HSI)
        loss = HS_l1
        loss.backward()
        nn.utils.clip_grad_norm_(f_model.parameters(), max_norm=2e-4, norm_type=2)
        optimizer_f.step()
        epoch_loss += loss.item()
        print("===> Epoch{}: Loss: {:.2e} || Learning rate: lr={}.".format(epoch, 
              loss.item(), optimizer_f.param_groups[0]['lr']))

def test():
    avg_psnr = 0.0
    avg_ergas = 0.0
    avg_sam = 0.0
    avg_ssim = 0.0
    torch.set_grad_enabled(False)

    epoch = scheduler_f.last_epoch
    
    f_model.eval()
    
    
    print('\nEvaluation:')
     
    for batch in test_data_loader:
        with torch.no_grad():
            HSI,MS,HS,_ = Variable(batch['gt']), Variable(batch['lrhsi']), Variable(batch['rgb']), Variable(batch['up'])
        if cuda:
            MS = MS.to(device)
            HS = HS.to(device)
            HSI = HSI.to(device)
            
            with torch.no_grad():
                MS0 = MS
                HS0 = HS
                MS = torch.nn.functional.interpolate(MS, scale_factor=4, mode='bilinear') # MS: 32x32x31
                r = read_r().to(device)
            
                B,C,H,W = HS.shape
                HS1 = rearrange(HS, 'b c h w -> b (h w) c')
                HS1 = torch.matmul(HS1, r)
                HS1 = rearrange(HS1, 'b (h w) c -> b c h w', h=H,w=W)
                
                out_HSI = f_model(HS1-MS,MS0,HS0) + MS
                
                out_HSI = out_HSI.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
                HSI = HSI.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
                
                avg_psnr += cal_psnr(out_HSI, HSI)
                avg_ergas += compute_ergas(out_HSI, HSI)
                avg_ssim += compute_ssim(out_HSI, HSI)
                avg_sam += compute_sam(out_HSI, HSI)
            
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    avg_sam = avg_sam / len(test_data_loader)
    avg_ergas = avg_ergas / len(test_data_loader)
    if avg_psnr >= ckt['psnr']:
        ckt['epoch'] = epoch
        ckt['psnr'] = avg_psnr
    print("===> Avg.PSNR: {:.4f} dB || ssim: {:.4f} || ergas: {:.4f} || sam: {:.4f} || Best.PSNR: {:.4f} dB || Epoch: {}"
          .format(avg_psnr, avg_ssim, avg_ergas, avg_sam, ckt['psnr'], ckt['epoch']))
    torch.set_grad_enabled(True)


def checkpoint(epoch):
    model_out_path = opt.save_folder+hostname+opt.model_type+"_epoch_{}.pth".format(epoch)
    torch.save({
        'f_model_state_dict': f_model.state_dict()},
        model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

train_set = DatasetFromHdf5(opt.train_dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, pin_memory=True, shuffle=True)
test_set = DatasetFromHdf5(opt.test_dataset)
test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, pin_memory=True, shuffle=True)
print('===> Building model ', opt.model_type)


f_model = DCINN(channel_in=31, channel_out=31, block_num=4).to(device)
l1 = torch.nn.L1Loss().to(device)

print('---------- Networks architecture -------------')
print_network(f_model)

optimizer_f = optim.Adam(f_model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i == opt.decay:
        milestones.append(i)
    if i == opt.decay1:
        milestones.append(i)
    if i == opt.decay2:
        milestones.append(i)
    if i == opt.decay3:
        milestones.append(i)
        
    
scheduler_f = lrs.MultiStepLR(optimizer_f, milestones, opt.gamma)
ckt = {'epoch':0, 'psnr':0.0} 

for epoch in range(opt.start_iter, opt.nEpochs + 1):
            
            train(epoch)
            scheduler_f.step()
            if (epoch+1) % (opt.snapshots) == 0:
                checkpoint(epoch)
                test()
