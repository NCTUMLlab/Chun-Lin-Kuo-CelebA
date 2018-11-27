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
parser.add_argument('--outf', default='modelfiles/bgan_celebA_1k', help='folder to output images and model checkpoints')
parser.add_argument('--numz', type=int, default=1, help='The number of set of z to marginalize over.')
parser.add_argument('--num_mcmc', type=int, default=10, help='The number of MCMC chains to run in parallel')
parser.add_argument('--gnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--dnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--d_optim', type=str, default='adam', choices=['adam', 'sgd'], help='')
parser.add_argument('--g_optim', type=str, default='adam', choices=['adam', 'sgd'], help='')
parser.add_argument('--bayes', type=int, default=1, help='Do Bayesian GAN or normal GAN')
import sys;
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

save_root = 'modelfiles/'

sys.argv = [''];
del sys
opt = parser.parse_args()
try:
    os.makedirs(opt.outf)
except OSError:
    print("Error Making Directory", opt.outf)
    pass

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

for ii, netG in enumerate(netGs):
    netG.load_state_dict(torch.load(save_root + 'netG%d_epoch_14.pth' % ii))
netD.load_state_dict(torch.load(save_root + 'netD_epoch_14.pth'))

for e in range(100):
    for _zid in range(opt.numz):
        for _mid in range(opt.num_mcmc):
            idx = _zid * opt.num_mcmc + _mid
            netG = netGs[idx]
            fixed_noise = torch.randn(1, opt.nz, 1, 1)

            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                              '%s/%d.png' % (opt.outf, e * 10 + idx),
                              normalize=True)
