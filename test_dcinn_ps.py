#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:07:41 2021

@author: wangwu
"""


from __future__ import print_function
import argparse
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.dcinn_ps import DCINN
from data_wv3_update import Dataset_Pro_Eval_Full
import scipy.io as sio
from utils import cal_psnr, compute_ssim, compute_ergas, compute_sam

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=64, help='Size of cropped HR image')
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--test_dataset', type=str, default='./testing_dataset/pansharpening/test.mat')
parser.add_argument('--output', default='result/', help='Location to save checkpoint models')
parser.add_argument('--model', default='./pretrained/model_pansharpening.pth', help='sr pretrained base model')


opt = parser.parse_args()
gpus_list=range(opt.gpus)
print(opt)
device = torch.device(opt.device)
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = Dataset_Pro_Eval_Full(opt.test_dataset)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)


print('===> Building model')
checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
f_model = DCINN(channel_in=8, channel_out=8, block_num=3).to(device)
f_model.load_state_dict(checkpoint['f_model_state_dict'])
print('Pre-trained model is loaded.')


if cuda:
    f_model = f_model.cuda(gpus_list[0])

def eval():
    i=1
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_ergas = 0.0
    avg_sam = 0.0
    for batch in testing_data_loader:
        with torch.no_grad():
            MS, HSI, HS = Variable(batch[0]), Variable(batch[1]),  Variable(batch[2])
        if cuda:
            MS = MS.to(device)
            HS = HS.to(device)
            HSI = HSI.to(device)

            with torch.no_grad():
                print(MS.shape,HS.shape,HSI.shape)
                    
                MS = HSI
                MS0 = HSI
                HS0 = HS
                HS1 = torch.repeat_interleave(HS, 8, dim=1)
                out_HSI = f_model.forward(HS1-MS,MS0,HS0)+MS
                    
                out_HSI = out_HSI.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
                HSI = HSI.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
                HS = HS.cpu().data.squeeze().clamp(0, 1).numpy()
                MS = MS.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

                psnr = cal_psnr(out_HSI, HSI)
                avg_psnr = avg_psnr + psnr
                ergas = compute_ergas(HSI, out_HSI)
                avg_ergas += ergas
                ssim = compute_ssim(out_HSI, HSI)
                avg_ssim = avg_ssim + ssim
                sam = compute_sam(out_HSI, HSI)
                avg_sam = avg_sam + sam
                print("===> PSNR: {:.4f} dB || ssim: {:.4f} || ergas: {:.4f}, ||sam: {:.4f}".format(psnr, ssim, ergas, sam))
                save_dir = './ps_result/out'+str(i)+'.mat'
                sio.savemat(save_dir, {'out':out_HSI})
                i = i+1
    print("Avg.ERGAS: {:.4f} || Avg.SSIM: {:.4f} || Avg.ERGAS: {:.4f}, || Avg.SAM: {:.4f}".format(avg_psnr/len(testing_data_loader),avg_ssim/len(testing_data_loader),avg_ergas/len(testing_data_loader),avg_sam/len(testing_data_loader)))

eval()
