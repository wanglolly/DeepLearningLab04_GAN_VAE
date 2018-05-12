from __future__ import print_function
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(64, 512, kernal_size = 4, stride = 1, bias=False),
            nn.BatchNorm2d(512, eps = 1e-05, momentum = 0.1, affine = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(512, 256, kernal_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(256, eps = 1e-05, momentum = 0.1, affine = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(256, 128, kernal_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(128, eps = 1e-05, momentum = 0.1, affine = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 64, kernal_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 1, kernal_size = 4, stride = 2, padding = 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Q = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size = 4, stride = 1, bias=False),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features = 8192, out_features = 100, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = 100, out_features = 10, bias = True)
        )

    def forward(self, input):
        output = self.main(input)
        output = self.Q(output)
        output = self.fc(output)

        return output.view(-1, 1).squeeze(1)
