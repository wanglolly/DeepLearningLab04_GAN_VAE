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
from torch.autograd import Variable
from infoGAN import Generator, CommonHead, Discriminator, Q, weights_init, noise_sample, fixedNoise_sample

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
dataset = dset.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor()]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, **kwargs)
device = torch.device("cuda:0" if opt.cuda else "cpu")

nz = int(opt.nz)
nc = 10
batch_size = int(opt.batchSize)

netG = Generator().to(device)
#netG.apply(weights_init)
netCH = CommonHead().to(device)
#netCH.apply(weights_init)
netD = Discriminator().to(device)
#netD.apply(weights_init)
netQ = Q().to(device)
#netQ.apply(weights_init)

criterion_D = nn.BCELoss()
criterion_Q = nn.CrossEntropyLoss()
fixed_noise = fixedNoise_sample(nz = nz, nc = nc, device = device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam([{'params':netCH.parameters()}, {'params': netD.parameters()}], lr=opt.dlr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam([{'params':netG.parameters()}, {'params': netQ.parameters()}], lr=opt.glr, betas=(opt.beta1, 0.999))

#Open Training loss File
LossFilename = 'infoGAN_Results/infoGAN_TrainingLoss.csv'
LossFile = open(LossFilename, 'w')
LossCursor = csv.writer(LossFile)

ProbFilename = 'infoGAN_Results/infoGAN_TrainingProb.csv'
ProbFile = open(ProbFilename, 'w')
ProbCursor = csv.writer(ProbFile)

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        optimizerD.zero_grad()

        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        prob_real = netD(netCH(real_cpu))
        errD_real = criterion_D(prob_real, label)
        errD_real.backward()
        D_x = prob_real.mean().item()

        # train with fake
        z, idx = noise_sample(bs = batch_size, nz = nz, nc = nc, device = device)
        fake = netG(z)
        label.fill_(fake_label)
        prob_fake = netD(netCH(fake.detach()))
        errD_fake = criterion_D(prob_fake, label)
        errD_fake.backward()
        D_G_z1 = prob_fake.mean().item()
        
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z))) + L(G,Q)
        ###########################
        optimizerG.zero_grad()

        label.fill_(real_label)  # fake labels are real for generator cost

        CH_out = netCH(fake)
        prob_fake = netD(CH_out)
        err_r = criterion_D(prob_fake, label)
        D_G_z2 = prob_fake.mean().item()

        q_output = netQ(CH_out)
        target = Variable(torch.LongTensor(idx).cuda())
        err_c = criterion_Q(q_output.squeeze(), target)

        errG = err_r + err_c
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_Q: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), err_r.item(), err_c.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    'infoGAN_Results/real_samples.png',
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    'infoGAN_Results/fake_samples_epoch_%03d.png' % (epoch),
                    normalize=True)
            LossHeader = [epoch, i, errD.item(), err_r.item(), err_c.item()]
            LossCursor.writerow(LossHeader)
            ProbHeader = [epoch, i, D_x, D_G_z1, D_G_z2]
            ProbCursor.writerow(ProbHeader)
            

    # do checkpointing
    if epoch % 5 == 0: 
        torch.save(netG.state_dict(), 'Models/CheckPoint/netG_epoch_%d.tar' % (epoch))
        torch.save(netD.state_dict(), 'Models/CheckPoint/netD_epoch_%d.tar' % (epoch))
        torch.save(netCH.state_dict(), 'Models/CheckPoint/netCH_epoch_%d.tar' % (epoch))
        torch.save(netQ.state_dict(), 'Models/CheckPoint/netQ_epoch_%d.tar' % (epoch))
torch.save(netG.state_dict(), 'Models/infoGAN_netG.tar')
torch.save(netD.state_dict(), 'Models/infoGAN_netD.tar')
torch.save(netCH.state_dict(), 'Models/infoGAN_netCH.tar')
torch.save(netQ.state_dict(), 'Models/infoGAN_netQ.tar')