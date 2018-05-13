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
from infoGAN import Generator, fixedNoise_sample

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

#Start to plot
plt.clf()
plotImageCount = 0
plt.subplots_adjust(wspace = 0.01, hspace = 0.01)
for i in range(args.sets):
    z = fixedNoise_sample(64, 10, device = device)
    x = netG(z)
    print(x.size())

    #plot image  
    for p in range(10):
        plt.subplot(args.sets, 10, plotImageCount + 1)
        plt.imshow(x[p].view(64,64).cpu().data.numpy(), norm = matplotlib.colors.Normalize(vmin = 0, vmax = 1), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plotImageCount = plotImageCount + 1

plt.savefig('infoGAN_Results/GeneratedImage.png', dpi=500)
plt.clf()
plt.close()
