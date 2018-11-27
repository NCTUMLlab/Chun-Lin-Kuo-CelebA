from __future__ import print_function
import os, pickle
import numpy as np
import random, math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from statsutil import AverageMeter, accuracy

# Default Parameters
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=64)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', type=int, default=1, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='modelfiles/celebA_1w', help='folder to output images and model checkpoints')
parser.add_argument('--numz', type=int, default=1, help='The number of set of z to marginalize over.')
parser.add_argument('--num_mcmc', type=int, default=10, help='The number of MCMC chains to run in parallel')
parser.add_argument('--gnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--dnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--d_optim', type=str, default='adam', choices=['adam', 'sgd'], help='')
parser.add_argument('--g_optim', type=str, default='adam', choices=['adam', 'sgd'], help='')
parser.add_argument('--bayes', type=int, default=1, help='Do Bayesian GAN or normal GAN')
import sys;
save_root = 'modelfiles'

sys.argv = [''];
del sys
opt = parser.parse_args()
try:
    os.makedirs(opt.outf)
except OSError:
    print("Error Making Directory", opt.outf)
    pass

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

data_dir = '../data/resized_celebA_64_10w/'  # this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)
dataloader = torch.utils.data.DataLoader(dset, batch_size= opt.batchSize, shuffle=True)

from models.discriminators import _netD64
from models.generators import _netG64
from statsutil import weights_init

netGs = []
for _idxz in range(opt.numz):
    for _idxm in range(opt.num_mcmc):
        netG = _netG64(opt.ngpu, nz=opt.nz)
        netG.apply(weights_init)
        netGs.append(netG)

num_classes = 1
netD = _netD64(opt.ngpu, num_classes=num_classes)

from models.distributions import Normal
from models.bayes import NoiseLoss, PriorLoss

# Finally, initialize the ``optimizers''
# Since we keep track of a set of parameters, we also need a set of
# ``optimizers''
if opt.d_optim == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
elif opt.d_optim == 'sgd':
    optimizerD = torch.optim.SGD(netD.parameters(), lr=opt.lr,
                                 momentum=0.9,
                                 nesterov=True,
                                 weight_decay=1e-4)
optimizerGs = []
for netG in netGs:
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerGs.append(optimizerG)

# since the log posterior is the average per sample, we also scale down the prior and the noise
gprior_criterion = PriorLoss(prior_std=1., observed=1000.)
gnoise_criterion = NoiseLoss(params=netGs[0].parameters(), scale=math.sqrt(2 * opt.gnoise_alpha / opt.lr),
                             observed=1000.)
dprior_criterion = PriorLoss(prior_std=1., observed=50000.)
dnoise_criterion = NoiseLoss(params=netD.parameters(), scale=math.sqrt(2 * opt.dnoise_alpha * opt.lr), observed=50000.)

# Fixed noise for data generation
fixed_noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).normal_(0, 1).cuda()
fixed_noise = Variable(fixed_noise)

# initialize input variables and use CUDA (optional)
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
bce = nn.BCELoss()

if opt.cuda:
    netD.cuda()
    for netG in netGs:
        netG.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()

iteration = 0
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader):
        iteration += 1
        #######
        # 1. real input
        netD.zero_grad()
        _input, _ = data
        batch_size = _input.size(0)
        if opt.cuda:
            _input = _input.cuda()
        input.resize_as_(_input).copy_(_input)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        #print(output.size(), labelv.size())
        errD_real = bce(output, label)
        errD_real.backward()

        #######
        # 2. Generated input
        fakes = []
        for _idxz in range(opt.numz):
            noise.resize_(batch_size, opt.nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            for _idxm in range(opt.num_mcmc):
                idx = _idxz * opt.num_mcmc + _idxm
                netG = netGs[idx]
                _fake = netG(noisev)
                fakes.append(_fake)
        fake = torch.cat(fakes)
        output = netD(fake.detach())
        labelv = torch.Tensor(fake.data.shape[0]).cuda().fill_(fake_label)
        #print(output.size(), labelv.size())
        errD_fake = bce(output, labelv)
        errD_fake.backward()

        if opt.bayes:
            errD_prior = dprior_criterion(netD.parameters())
            errD_prior.backward()
            errD_noise = dnoise_criterion(netD.parameters())
            errD_noise.backward()
            errD = errD_real + errD_fake + errD_prior + errD_noise
        else:
            errD = errD_real + errD_fake
        optimizerD.step()

        # 4. Generator
        for netG in netGs:
            netG.zero_grad()
        labelv = Variable(torch.FloatTensor(fake.data.shape[0]).cuda().fill_(real_label))
        output = netD(fake)
        #print(output.size(), labelv.size())
        errG = bce(output, labelv)
        if opt.bayes:
            for netG in netGs:
                errG += gprior_criterion(netG.parameters())
                errG += gnoise_criterion(netG.parameters())
        errG.backward()
        for optimizerG in optimizerGs:
            optimizerG.step()


        # 6. get test accuracy after every interval
        if iteration % len(dataloader) == 0:
            print('[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f ' % (epoch, opt.niter, i, len(dataloader), errD.data.item(), errG.data.item()))
    # after each epoch, save images
    # vutils.save_image(_input,'%s/real_samples.png' % opt.outf, normalize=True)
    # for _zid in range(opt.numz):
    #     for _mid in range(opt.num_mcmc):
    #         idx = _zid * opt.num_mcmc + _mid
    #         netG = netGs[idx]
    #
    #         fake = netG(fixed_noise)
    #         vutils.save_image(fake.data,
    #                           '%s/fake_samples_epoch_%03d_G_z%02d_m%02d.png' % (opt.outf, epoch, _zid, _mid),
    #                           normalize=True)
    for ii, netG in enumerate(netGs):
        torch.save(netG.state_dict(), '%s/netG%d_epoch_%d.pth' % (save_root, ii, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (save_root, epoch))
    #torch.save(netD_fullsup.state_dict(), '%s/netD_fullsup_epoch_%d.pth' % (opt.outf, epoch))
for e in range(1000):
    for _zid in range(opt.numz):
        for _mid in range(opt.num_mcmc):
            idx = _zid * opt.num_mcmc + _mid
            netG = netGs[idx]
            fixed_noise = torch.FloatTensor(1, opt.nz, 1, 1).normal_(0, 1).cuda()
            fixed_noise = Variable(fixed_noise)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                              '%s/%d.png' % (opt.outf, e*10 + idx),
                              normalize=True)
