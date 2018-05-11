from __future__ import print_function
import os
import csv
import argparse
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import to_var, idx2onehot
from CVAE import CVAE

#input parser
parser = argparse.ArgumentParser(description='CVAE MNIST Example')
parser.add_argument('--model', type=str, default='',
                    help='path to model to evaluate')
parser.add_argument('--sets', type=int, default=10,
                    help='number of set to plot (default: 10)')
args = parser.parse_args()

#setup models
model = CVAE()
model.load_state_dict(torch.load(args.model))
model.cuda()
model.eval()

plt.clf()
plotImageCount = 0
plt.subplots_adjust(wspace = 0.01, hspace = 0.01)
for i in range(args.sets):
    c = to_var(torch.arange(0,10).long().view(-1,1))
    x = model.inference(n = c.size(0), c = c)

    #plot image  
    for p in range(10):
        plt.subplot(args.sets, 10, plotImageCount + 1)
        plt.imshow(x[p].view(28,28).cpu().data.numpy(), cmap='gray')
        plt.axis('off')
        plotImageCount = plotImageCount + 1

plt.savefig('CVAE_Results/PARRImage.png', dpi=500)
plt.clf()
plt.close()