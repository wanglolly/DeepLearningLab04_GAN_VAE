from __future__ import print_function
import os
import csv
import argparse
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import to_var, idx2onehot
from infoGAN import Generator

#input parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='',
                    help='path to model to evaluate')
parser.add_argument('--sets', type=int, default=10,
                    help='number of set to plot (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

#setup models
netG = Generator()
netG.load_state_dict(torch.load(args.model))
netG.cuda()
netG.eval()

#generate noise
def noise_sample(nz, nc, device):
    idx = np.arange(nc)
    c = np.zeros((nc, nc))
    c[range(nc),idx] = 1.0
    noise = torch.randn(1, nz - nc, device=device)
    noise = noise.expand(nc, -1)
    c_tensor = torch.FloatTensor(nc, nc).cuda()
    c_tensor.data.copy_(torch.Tensor(c))
    z = torch.cat([noise, c_tensor], 1).view(-1, nz, 1, 1)
    return z


plt.clf()
plotImageCount = 0
plt.subplots_adjust(wspace = 0.01, hspace = 0.01)
for i in range(args.sets):
    z = noise_sample(64, 10, device = device)
    x = netG(z)

    #plot image  
    for p in range(10):
        plt.subplot(args.sets, 10, plotImageCount + 1)
        plt.imshow(x[p].view(28,28).cpu().data.numpy(), norm = matplotlib.colors.Normalize(vmin = 0, vmax = 1))
        plt.axis('off')
        plotImageCount = plotImageCount + 1

plt.savefig('infoGAN_Results/GeneratedImage.png', dpi=500)
plt.clf()
plt.close()
