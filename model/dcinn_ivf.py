
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from haar_down import HaarDownsampling
from math import exp
from torch.autograd import Variable


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class GN(nn.Module):
    def __init__(self, num_spectral, r=2):
        super(GN, self).__init__()
        
        self.num_spectral = num_spectral
        self.g = nn.Conv2d(num_spectral, num_spectral*r, 1, 1, 0, bias=False)
        self.q = nn.Conv2d(num_spectral, num_spectral*r, 1, 1, 0, bias=False)
        self.o = nn.Conv2d(num_spectral*r, num_spectral, 1, 1, 0, bias=False)
        self.gelu = nn.GELU()
       
    def forward(self, x):
        x0 = x
        
        g = self.g(x)
        q = self.gelu(g)
        q = self.q(x)
        o = g*q
        o = self.o(o)
        x = x0+o
        return x
        

class SELayer(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, norm=None):
        super(SELayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_du = torch.nn.Sequential(
                torch.nn.Conv2d(num_filter, num_filter//8, 1, 1, 0, bias=bias),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(num_filter//8, num_filter, 1, 1, 0, bias=bias),
                torch.nn.Sigmoid())
    
    def forward(self, x):
        mask = self.global_pool(x)
        mask = self.conv_du(mask)
        x = x * mask
        return x

class Res(nn.Module):
    def __init__(self, n_feat, groups=1):
        super(Res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, bias=True, groups=groups),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, bias=True, groups=groups))
        self.se = SELayer(n_feat)
        
    def forward(self, x):
        res = self.conv1(x)
        res = self.se(res)
        return res+x

class Condition_Net(torch.nn.Module):
    def __init__(self, num_channel=1, base_filter=16, num_spectral=1, num_block=2):
        super(Condition_Net, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.in_conv = nn.Conv2d(num_channel+num_spectral, base_filter, 3, 1, 0, bias=True)
        self.down2 = nn.Upsample(scale_factor=1/2, mode='bilinear')
        body = [Res(base_filter) \
                for _ in range(num_block)]
        self.df = nn.Sequential(*body)

        self.tail1 = nn.Conv2d(base_filter, 8, 3, 1, 0, bias=True)
        self.tail2 = nn.Conv2d(base_filter, 16, 3, 1, 0, bias=True)
        self.tail3 = nn.Conv2d(base_filter, 32, 3, 1, 0, bias=True)

    def forward(self, Y, Z):

        res = torch.cat((Y,Z), dim=1)
        res = self.in_conv(self.pad(res))
        res = self.df(res)
        res = self.down2(res)
        m1 = self.tail1(self.pad(res))
        res = self.down2(res)
        m2 = self.tail2(self.pad(res))
        m3 = self.tail3(self.pad(res))
        m3 = self.down2(m3)
        
        return m1,m2,m3

class My_Block(nn.Module):
    def __init__(self, channel_in, channel_out, gc=48, mshape=32, bias=True):
        super(My_Block, self).__init__()
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(channel_in+mshape, gc, 3, 1, 0, bias=bias))
        self.conv2 = Res(gc)
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(gc, channel_out, 3, 1, 0, bias=bias))
        self.ca = GN(gc)
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)
    
        initialize_weights(self.conv3, 0)

    def forward(self, x,m):
        x = torch.cat((x,m),dim=1)
        x = self.conv1(x)
        x = self.ca(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

class InvBlockC(nn.Module):
    def __init__(self, channel_num, channel_split_num, gc=64, mshape=16):
        super(InvBlockC, self).__init__()

        self.split_len1 = channel_split_num 
        self.split_len2 = channel_num - channel_split_num 
        self.F = My_Block(self.split_len2, self.split_len1, gc,mshape)
        self.G = My_Block(self.split_len1, self.split_len2, gc,mshape)
        self.H = My_Block(self.split_len1, self.split_len2, gc,mshape)

        self.kernel1 = nn.Parameter(torch.rand(channel_num,channel_num,5,5)*1e-6)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x,m, rev=False):
        if not rev:     
            x = conv_exp(x,self.kernel1)
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) 
            y1 = x1 + self.F(x2,m)
            self.s = (torch.sigmoid(self.H(y1,m)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1,m) 
            out = torch.cat((y1, y2), 1)
        else:
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) 
            self.s = (torch.sigmoid(self.H(x1,m)) * 2 - 1)
            y2 = (x2 - self.G(x1,m)).div(torch.exp(self.s)) 
            y1 = x1 - self.F(y2,m) 
            x = torch.cat((y1, y2), 1) 
            out = inv_conv_exp(x, self.kernel1)
        return out

def conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    B,C,H,W = input.size()
    
    assert kernel.size(0) == kernel.size(1)
    assert kernel.size(0) == C, '{} !={}'.format(kernel.size(0), C)
    
    padding = (kernel.size(2)-1)//2, (kernel.size(3)-1)//2
    
    result = input
    product = input
    
    for i in range(1,terms+1):
        product = F.conv2d(product, kernel, padding=padding, stride=(1,1))/i
        result = result + product
        
        if dynamic_truncation != 0 and i>5:
            if product.abs().max().item() < dynamic_truncation:
                break
    if verbose:
        print('Maximum element size in term:{}'.format(torch.max(torch.abs(product))))
    return result
    
def inv_conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    return conv_exp(input, -kernel, terms, dynamic_truncation, verbose)


class Encoder(nn.Module):
    def __init__(self, block_num=4):
        super(Encoder, self).__init__()
        operations = []

        self.inv_in1 = InvBlockC(4, 4//2, gc=8,mshape=8)
        self.inv_in2 = InvBlockC(16, 16//2, gc=32,mshape=16)
        self.inv_out2 = InvBlockC(16, 16//2, gc=32,mshape=16)
        self.inv_out1 = InvBlockC(4, 4//2, gc=8,mshape=8)
        for j in range(block_num): 
            b = InvBlockC(64, 64//2, gc=64,mshape=32)
            operations.append(b)
        self.operations = nn.ModuleList(operations)
        self.har_in1 = HaarDownsampling(1)
        self.har_in2 = HaarDownsampling(4)
        self.har_in3 = HaarDownsampling(16)
        self.har_out3 = HaarDownsampling(16)
        self.har_out2 = HaarDownsampling(4)
        self.har_out1 = HaarDownsampling(1)
    def forward(self, x,m1,m2,m3): 
        
        x = self.har_in1(x) 
        x = self.inv_in1(x,m1)
        x = self.har_in2(x) 
        x = self.inv_in2(x,m2)
        x = self.har_in3(x)
        for op in self.operations:
            x = op.forward(x,m3)
        x = self.har_out3(x,True)
        x = self.inv_out2(x,m2)
        x = self.har_out2(x, rev=True)
        x = self.inv_out1(x,m1)
        x = self.har_out1(x,True)
        return x
    def reverse(self, x,m1,m2,m3):
        x = self.har_out1(x)
        x = self.inv_out1(x,m1,True)
        x = self.har_out2(x)
        x = self.inv_out2(x,m2, True)
        x = self.har_out3(x)
        for op in reversed(self.operations):
            x = op.forward(x,m3, True)
        x = self.har_in3(x, True)
        x = self.inv_in2(x,m2)
        x = self.har_in2(x, True) # up
        x = self.inv_in1(x,m1, True)
        x = self.har_in1(x, True) # up
        return x

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def avg_filter(img1, window_size, sigma):
    channel=1
    window = create_window(window_size, channel, sigma)
    window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    return mu1
        
class DCINN(nn.Module):
    def __init__(self):
        super(DCINN, self).__init__()
        self.inn = Encoder()
        self.c_net = Condition_Net()

    def forward(self, ir, vi, base_rule): 
        m1,m2,m3 = self.c_net(ir,vi)


        ir_base = avg_filter(ir,11, 1)
        vi_base = avg_filter(vi,11, 1) 
        if base_rule == 'Max':
            fused_base,_ = torch.max(torch.cat((ir_base,vi_base),dim=1),dim=1)
            fused_base = fused_base.unsqueeze(1)
        else:
            fused_base = ir_base*1/2 + vi_base*1/2

        ir_detail = ir - ir_base
        vi_detail = vi - vi_base
        
        fused_detail = self.inn(ir_detail+vi_detail, m1, m2, m3)
        out = fused_detail + fused_base
        return out
