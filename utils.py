import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import mm
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
#from skimage.measure import compare_ssim
import random
import numpy
#import xlrd
import os

def determine_conv_functional(n_dim, transposed=False):
    if n_dim is 1:
        if not transposed:
            return nn.functional.conv1d
        else:
            return nn.functional.conv_transposed1d
    elif n_dim is 2:
        if not transposed:
            return nn.functional.conv2d
        else:
            return nn.functional.conv_transposed2d
    elif n_dim is 3:
        if not transposed:
            return nn.functional.conv3d
        else:
            return nn.functional.conv_transposed3d
    else:
        NotImplementedError("No convolution of this dimensionality implemented")


def compute_ergas(out, gt):
    num_spectral = out.shape[-1]
    out = np.reshape(out, (-1, num_spectral)) 
    gt = np.reshape(gt, (-1, num_spectral))
    diff = gt - out
    mse = np.mean(np.square(diff), axis=0)
    gt_mean = np.mean(gt, axis=0)
    mse = np.reshape(mse, (num_spectral,1))
    gt_mean = np.reshape(gt_mean, (num_spectral,1))
    ergas = 100/4*np.sqrt(np.mean(mse/(gt_mean**2+1e-6)))
    return ergas

def compute_sam(im1, im2):
    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral))
    im2 = np.reshape(im2, (-1, num_spectral))
    mole = np.sum(np.multiply(im1, im2), axis=1)
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
    deno = np.multiply(im1_norm, im2_norm)
    
    sam = np.rad2deg(np.arccos((mole)/(deno+1e-7)))
    sam = np.mean(sam)
    return sam

def compute_ssim(im1, im2):
    n = im1.shape[2]
    ms_ssim=0.0
    for i in range(n):
        single_ssim = compare_ssim(im1[:,:,i], im2[:,:,i])
        ms_ssim += single_ssim
    return ms_ssim/n

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    

def cal_psnr(im1, im2):
    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral ))
    im2 = np.reshape(im2, (-1, num_spectral ))
    diff = im1 - im2

    mse = np.mean(np.square(diff), axis=0)
    

    return np.mean(10 * np.log10(1/mse))
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()
