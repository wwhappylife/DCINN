
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from utils import determine_conv_functional
from haar_down import HaarDownsampling
from scipy import linalg as la
from einops import rearrange


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

class LN(nn.Module):
    def __init__(self, dim):
        super(LN, self).__init__()
        
        self.weights = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.ones(dim))
        
    def forward(self, X):
        
        h,w = X.shape[-2:]
        X = rearrange(X, 'b c h w -> b (h w) c')
        mu = X.mean(-1, keepdim=True)
        sigma = X.var(-1, keepdim=True, unbiased=False)
        X = (X-mu)/torch.sqrt(sigma+1e-5)*self.weights + self.bias
        X = rearrange(X, 'b (h w) c -> b c h w', h=h, w=w)
        return X

class GN(nn.Module):
    def __init__(self, num_spectral, r=2):
        super(GN, self).__init__()
        
        self.num_spectral = num_spectral
        self.g = nn.Conv2d(num_spectral, int(num_spectral*r), 1, 1, 0, bias=False)
        self.q = nn.Conv2d(num_spectral, int(num_spectral*r), 1, 1, 0, bias=False)
        self.o = nn.Conv2d(int(num_spectral*r), num_spectral, 1, 1, 0, bias=False)
        self.gelu = nn.GELU()
        self.ln = LN(num_spectral)
    
    def forward(self, x):
        x0 = x
        x = self.ln(x)
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
                torch.nn.Conv2d(num_filter, num_filter//16, 1, 1, 0, bias=bias),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(num_filter//16, num_filter, 1, 1, 0, bias=bias),
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

class CNet(torch.nn.Module):
    def __init__(self, scale_factor=4, num_channel=1, base_filter=32, num_spectral=8, num_block=2):
        super(CNet, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.in_conv = nn.Conv2d(num_channel+num_spectral, base_filter, 3, 1, 0, bias=True)
        self.tailh = nn.Conv2d(base_filter, 16, 3, 1, 0, bias=True)
        self.tail = nn.Conv2d(base_filter, 32, 3, 1, 0, bias=True)
        self.down = nn.Upsample(scale_factor=1/2, mode='bilinear')
        body = [Res(base_filter) \
                for _ in range(num_block)]
        self.df = nn.Sequential(*body)

    def forward(self, Y, Z):

        res = torch.cat((Y,Z), dim=1)
        res = self.in_conv(self.pad(res))
        res = self.df(res)
        resh = self.tailh(self.pad(res))
        res1 = self.pad(self.tail(res))
        reslr = self.down(res1)
    
        return resh,reslr

class MyBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=48, mshape=32, bias=True):
        super(MyBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(channel_in+mshape, gc, 3, 1, 0, bias=bias))
        self.conv2 = Res(gc)
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(gc, channel_out, 3, 1, 0, bias=bias))            
        self.ca = GN(gc)
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)
    
        initialize_weights(self.conv3, 0)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.ca(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class CInvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, gc=64, mshape=32):
        super(CInvBlock, self).__init__()

        self.channel_num=channel_num
        self.channel_split_num = channel_split_num
        self.gc=gc
        self.mshape=mshape
        self.split_len1 = channel_split_num 
        self.split_len2 = channel_num - channel_split_num 
        self.clamp_s = nn.Parameter(torch.rand(1,self.split_len2,1,1))
        self.F = MyBlock(self.split_len2, self.split_len1, gc, mshape)
        self.G = MyBlock(self.split_len1, self.split_len2, gc, mshape)
        self.H = MyBlock(self.split_len1, self.split_len2, gc, mshape)
        

        self.norm = ActNorm(channel_num)
        self.kernel1 = nn.Parameter(torch.rand(channel_num,channel_num,5,5)*1e-6)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, m, rev=False):
        if not rev:         
            
            x = self.norm.forward(x)
            x = conv_exp(x,self.kernel1)
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) 
            
            y1 = x1 + self.F(torch.cat((x2,m),dim=1))
            self.s = self.clamp_s * torch.sigmoid(self.H(torch.cat((y1,m),dim=1) * 2 - 1))
            y2 = x2.mul(torch.exp(self.s)) + self.G(torch.cat((y1,m),dim=1)) 
            out = torch.cat((y1, y2), 1)
        else:
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) 
            self.s = self.clamp_s * torch.sigmoid(self.H(torch.cat((x1,m),dim=1) * 2 - 1))
            y2 = (x2 - self.G(torch.cat((x1,m),dim=1))).div(torch.exp(self.s)) 
            y1 = x1 - self.F(torch.cat((y2,m),dim=1))
            x = torch.cat((y1, y2), 1) 
            out = inv_conv_exp(x, self.kernel1)
            out = self.norm.reverse(out) 

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



class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

class DCINN(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, block_num=8):
        super(DCINN, self).__init__()
        operations = []

        channel_num = channel_in
        channel_split_num = channel_in//2
        self.channel_in = channel_in
        self.inv_in = CInvBlock(channel_in, channel_in//2, gc=16, mshape=16)
        for j in range(block_num): 
            b = CInvBlock(channel_num*4, channel_split_num*4, gc=32, mshape=32)
            operations.append(b)
        self.operations = nn.ModuleList(operations)
        self.inv_out = CInvBlock(channel_in, channel_in//2, gc=16, mshape=16)
        self.har_in = HaarDownsampling(channel_in)
        self.har_out = HaarDownsampling(channel_in)
        self.c_net = CNet()
    
    def forward(self, x, y,z):
        m1,m = self.c_net(y,z)
       
        x = self.inv_in(x,m1)
        x = self.har_in(x)
        
        for op in self.operations:
            x = op.forward(x,m)
        x = self.har_out(x, rev=True)
        x = self.inv_out(x,m1)
        return x
    def reverse(self, x,y,z):
        m1,m = self.c_net(y,z)
        x = self.inv_out(x, m1, True)
        x = self.har_out(x, False)
        for op in reversed(self.operations):
            x = op.forward(x, m, True)
        x = self.har_in(x, True) 
        x = self.inv_in(x,m1, True)
        return x
