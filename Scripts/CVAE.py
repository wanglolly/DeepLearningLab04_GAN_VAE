from __future__ import print_function
import os
import csv
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import to_var, idx2onehot


parser = argparse.ArgumentParser(description='CVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

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
        h2 = torch.unsqueeze(h2, -1)
        h2 = h2.view(128, 2, 196)
        h2 = torch.unsqueeze(h2, -1)
        h2 = h2.view(128, 2, 14, 14)
        h2 = self.conv2(h2)
        return h2

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar, z


model = CVAE()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch, writer):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        #get data & label
        data = to_var(data)
        label = to_var(label)
        label = label.view(-1, 1)

        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data, label)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            header = [epoch, batch_idx, loss.item() / len(data)]
            writer.writerow(header)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = to_var(data)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         '../CVAE_Results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


#Open Training loss File
trainFilename = '../CVAE_TrainingLoss.csv'
trainFile = open(trainFilename, 'w')
trainCursor = csv.writer(trainFile)

for epoch in range(1, args.epochs + 1):
    train(epoch, trainCursor)
    test(epoch)
    with torch.no_grad():
        sample = to_var(torch.randn(64, 20))
        sample = model.decode(sample, [0]).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   '../CVAE_Results/sample_' + str(epoch) + '.png')
trainFile.close()