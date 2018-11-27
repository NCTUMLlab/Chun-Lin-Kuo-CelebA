import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer, GaussianVariationalInference, BBBTranspose2d, Reshape
import numpy as np

# torch.manual_seed(123)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch CelebA WAE-GAN')
parser.add_argument('-batch_size', type=int, default=10000, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=2, help='number of epochs to generate (default: 100)')
parser.add_argument('-dim_h', type=int, default=128, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default=64, help='hidden dimension of z (default: 8)')
parser.add_argument('-n_channel', type=int, default=3, help='input channels (default: 1)')
parser.add_argument('-sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')
args = parser.parse_args()

# #transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# #data_dir = './resized_celebA_64_10w/'  # this path depends on your computer
# dset = datasets.ImageFolder(data_dir, transform)
# train_loader = torch.utils.data.DataLoader(dset, batch_size= args.batch_size, shuffle=True)


result_dir = '../FID/smaple_1w'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.conv1 = BBBConv2d(self.n_channel, self.dim_h, kernel_size=5, stride=2, padding=2)
        self.conv1_bn = nn.BatchNorm2d(self.dim_h)
        self.soft1 = nn.Softplus()
        self.conv2 = BBBConv2d(self.dim_h, self.dim_h * 2, 5, 2, 2)
        self.conv2_bn = nn.BatchNorm2d(self.dim_h * 2)
        self.soft2 = nn.Softplus()
        self.conv3 = BBBConv2d(self.dim_h * 2, self.dim_h * 4, 5, 2, 2)
        self.conv3_bn = nn.BatchNorm2d(self.dim_h * 4)
        self.soft3 = nn.Softplus()
        self.conv4 = BBBConv2d(self.dim_h * 4, self.dim_h * 8, 5, 2, 2)
        self.conv4_bn = nn.BatchNorm2d(self.dim_h * 8)
        self.soft4 = nn.Softplus()
        self.flatten = FlattenLayer(self.dim_h * 8 * (4 ** 2))
        self.fc = BBBLinearFactorial(self.dim_h * 8 * (4 ** 2), self.n_z)

        layers = [self.conv1, self.conv1_bn, self.soft1,
                  self.conv2, self.conv2_bn, self.soft2,
                  self.conv3, self.conv3_bn, self.soft3,
                  self.conv4, self.conv4_bn, self.soft4,
                  self.flatten, self.fc]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        # print('logits', logits)
        return logits, kl


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.proj = BBBLinearFactorial(self.n_z, self.dim_h * 8 * (8 ** 2))
        self.reshape = Reshape(self.dim_h * 8, 8)
        self.relu0 = nn.ReLU()
        self.convt1 = BBBTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1)
        self.convt1_bn = nn.BatchNorm2d(self.dim_h * 4)
        self.soft1 = nn.Softplus()
        self.convt2 = BBBTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1)
        self.convt2_bn = nn.BatchNorm2d(self.dim_h * 2)
        self.soft2 = nn.Softplus()
        self.convt3 = BBBTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1)
        self.convt3_bn = nn.BatchNorm2d(self.dim_h)
        self.soft3 = nn.Softplus()
        self.convt4 = BBBTranspose2d(self.dim_h, self.n_channel, 5, 1, 2)
        self.tanh = nn.Tanh()

        layers = [self.proj, self.reshape, self.relu0,
                  self.convt1, self.convt1_bn, self.soft1,
                  self.convt2, self.convt2_bn, self.soft2,
                  self.convt3, self.convt3_bn, self.soft3,
                  self.convt4, self.tanh]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        # print('logits', logits)
        return logits, kl


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            # nn.BatchNorm1d(self.dim_h * 4),
            nn.ReLU(),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            # nn.BatchNorm1d(self.dim_h * 4),
            nn.ReLU(),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            # nn.BatchNorm1d(self.dim_h * 4),
            nn.ReLU(),
            nn.Linear(self.dim_h * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        hi = self.main(x)
        # sigma2_p = 1.**2
        # normsq = np.sum((x**2).cpu().data.numpy(), axis= 1)
        # normsq = torch.from_numpy(normsq).view(args.batch_size,1).to(device)
        #
        # hi = hi - normsq/ 2. / sigma2_p \
        #      - 0.5 * np.log(2. * np.pi) \
        #      - 0.5 * args.n_z * np.log(sigma2_p)
        return hi


encoder, decoder, discriminator = Encoder(args).to(device), Decoder(args).to(device), Discriminator(args).to(device)
encoder.load_state_dict(torch.load('encdoer_20.pkl'))
decoder.load_state_dict(torch.load('decoder_20.pkl'))
discriminator.load_state_dict(torch.load('discriminator_20.pkl'))

def denorm(x):
    out = (x + 1) / 2
    return out
for epoch in range(args.epochs):
    decoder.eval()
    z = (torch.randn(1, args.n_z) * args.sigma).to(device)
    x_sam, _ = decoder.probforward(z)
    save_image(denorm(x_sam.view(-1, args.n_channel, 64, 64)),
               result_dir + '%d.png' % (epoch + 1))

