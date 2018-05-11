from __future__ import print_function
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import to_var, idx2onehot

class CVAE(nn.Module):   
    def __init__(self):
        super(CVAE, self).__init__()
        #encoder
        self.conv1 = nn.Sequential(
                    nn.Conv2d(11, 3, kernel_size=3, stride = 1, padding = 1),
                    nn.ReLU(),
                    nn.Conv2d(3, 1, kernel_size=3, stride = 1, padding = 1),
                    nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(in_features = 784, out_features = 400, bias = True),
                                nn.ReLU())
        self.fc21 = nn.Linear(in_features = 400, out_features = 20, bias = True)
        self.fc22 = nn.Linear(in_features = 400, out_features = 20, bias = True)

        #decoder
        self.fc3 = nn.Sequential(nn.Linear(in_features = 30, out_features = 392, bias = True),
                                nn.ReLU())
        self.conv2 = nn.Sequential(
                    nn.Conv2d(2, 11, kernel_size=3, stride = 1, padding = 1),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.Conv2d(11, 3, kernel_size=3, stride = 1, padding = 1),
                    nn.ReLU(),
                    nn.Conv2d(3, 1, kernel_size=3, stride = 1, padding = 1),
                    nn.Sigmoid())

    def encode(self, x, c):
        c = idx2onehot(c, n=10)
        c = torch.unsqueeze(c, -1)
        c = torch.unsqueeze(c, -1)
        c = c.expand(-1, -1, x.size()[2], x.size()[3])
        x = torch.cat((x, c), dim=1)
        h1 = self.conv1(x)
        h1 = h1.view(-1, 784)
        h1 = self.fc1(h1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.rand_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c):
        c = idx2onehot(c, n=10)
        z = torch.cat((z, c), dim=-1)
        h2 = self.fc3(z)
        h2 = h2.view(-1, 2, 14, 14)
        h2 = self.conv2(h2)
        return h2

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
