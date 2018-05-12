from __future__ import print_function
import argparse
import os
import csv
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from infoGAN import Generator, Discriminator, weights_init, noise_sample

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=80, help='number of epochs to train for')
parser.add_argument('--glr', type=float, default=1e-3, help='learning rate for generator, default=0.001')
parser.add_argument('--dlr', type=float, default=2e-4, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

kwargs = {'num_workers': 1, 'pin_memory': True} if opt.cuda else {}
dataloader = torch.utils.data.DataLoader(
    dset.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    batch_size=opt.batchSize, shuffle=True))
device = torch.device("cuda:0" if opt.cuda else "cpu")
print(device)

nz = int(opt.nz)
nc = 10
batch_size = int(opt.batchSize)

netG = Generator().to(device)
netG.apply(weights_init)

netD = Discriminator().to(device)
netD.apply(weights_init)

criterion_D = nn.BCELoss()
criterion_Q = nn.CrossEntropyLoss()
fixed_noise = torch.randn(opt.batchSize, nz - nc, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.dlr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.glr, betas=(opt.beta1, 0.999))

#Open Training loss File
LossFilename = 'infoGAN_Results/infoGAN_TrainingLoss.csv'
LossFile = open(LossFilename, 'w')
LossCursor = csv.writer(LossFile)

ProbFilename = 'infoGAN_Results/infoGAN_TrainingProb.csv'
ProbFile = open(ProbFilename, 'w')
ProbCursor = csv.writer(ProbFile)

bestLoss = 100
bestLossEpoc = -1


for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        prob_real, _ = netD(real_cpu)
        errD_real = criterion_D(prob_real, label)
        errD_real.backward()
        D_x = prob_real.mean().item()

        # train with fake
        z, idx = noise_sample(bs = batch_size, nz = nz, nc = nc, device = device)
        fake = netG(z)
        label.fill_(fake_label)
        prob_fake, _ = netD(fake.detach())
        errD_fake = criterion_D(prob_fake, label)
        errD_fake.backward()
        D_G_z1 = prob_fake.mean().item()
        
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z))) + L(G,Q)
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        prob_fake, q_output = netD(fake)
        err_r = criterion_D(prob_fake, label)
        D_G_z2 = prob_fake.mean().item()

        class_ = torch.LongTensor(idx).cuda()
        target = Variable(class_)
        err_c = criterion_Q(q_output, target)
        
        errG = err_r + err_c
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_Q: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), err_r.item(), err_c.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    'infoGAN_Results/real_samples_%03d.png'% (epoch),
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    'infoGAN_Results/fake_samples_epoch_%03d.png' % (epoch),
                    normalize=True)
            LossHeader = [epoch, i, errD, err_r, err_c]
            LossCursor.writerow(LossHeader)
            ProbHeader = [epoch, i, D_x, D_G_z1, D_G_z2]
            ProbCursor.writerow(ProbHeader)
            

    # do checkpointing
    #torch.save(netG.state_dict(), 'Models/CheckPoint/netG_epoch_%d.tar' % (epoch))
    #torch.save(netD.state_dict(), 'Models/CheckPoint/netD_epoch_%d.tar' % (epoch))
torch.save(netG.state_dict(), 'Models/infoGAN_netG.tar')
torch.save(netD.state_dict(), 'Models/infoGAN_netD.tar')