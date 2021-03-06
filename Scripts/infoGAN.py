from __future__ import print_function
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(64, 512, kernel_size = 4, stride = 1, bias=False),
            nn.BatchNorm2d(512, eps = 1e-05, momentum = 0.1, affine = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(256, eps = 1e-05, momentum = 0.1, affine = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(128, eps = 1e-05, momentum = 0.1, affine = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 1, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class CommonHead(nn.Module):
    def __init__(self):
        super(CommonHead, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, 64, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(128, eps = 1e-05, momentum = 0.1, affine = True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(256, eps = 1e-05, momentum = 0.1, affine = True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(512, eps = 1e-05, momentum = 0.1, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size = 4, stride = 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size = 4, stride = 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input).view(-1, 1)

class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features = 8192, out_features = 100, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = 100, out_features = 10, bias = True)
        )
    def forward(self, input):
        return self.main(input.view(-1, 8192))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

#generate noise
def noise_sample(bs, nz, nc, device):

    idx = np.random.randint(nc, size=bs)
    c = np.zeros((bs, nc))
    c[range(bs),idx] = 1.0

    noise = torch.randn(bs, nz - nc, device=device)
    c_tensor = torch.FloatTensor(bs, nc).cuda()
    c_tensor.data.copy_(torch.Tensor(c))
    z = torch.cat([noise, c_tensor], 1).view(-1, nz, 1, 1)

    return z, idx

#generate noise
def fixedNoise_sample(nz, nc, device):
    idx = np.arange(nc)
    c = np.zeros((nc, nc))
    c[range(nc),idx] = 1.0
    noise = torch.randn(1, nz - nc, device=device)
    noise = noise.expand(nc, -1)
    c_tensor = torch.FloatTensor(nc, nc).cuda()
    c_tensor.data.copy_(torch.Tensor(c))
    z = torch.cat([noise, c_tensor], 1).view(-1, nz, 1, 1)
    return z

def fixedNoise_single_sample(nz, nc, num, device):
    idx = np.arange(num, num + 1)
    c = np.zeros((1, nc))
    c[0,idx] = 1.0
    noise = torch.randn(1, nz - nc, device=device)
    c_tensor = torch.FloatTensor(1, nc).cuda()
    c_tensor.data.copy_(torch.Tensor(c))
    z = torch.cat([noise, c_tensor], 1).view(-1, nz, 1, 1)
    return z
