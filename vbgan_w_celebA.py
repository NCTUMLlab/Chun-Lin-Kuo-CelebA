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

#torch.manual_seed(123)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch CelebA WAE-GAN')
parser.add_argument('-batch_size', type=int, default=10, metavar='N',help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=20, help='number of epochs to train (default: 100)')
parser.add_argument('-lr', type=float, default= 1e-3, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=128, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default=64, help='hidden dimension of z (default: 8)')
parser.add_argument('-LAMBDA', type=float, default=10, help='regularization coef MMD term (default: 10)')
parser.add_argument('-n_channel', type=int, default=3, help='input channels (default: 1)')
parser.add_argument('-sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')
args = parser.parse_args()

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
data_dir = './resized_celebA_64_10w/'  # this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size= args.batch_size, shuffle=True)


result_dir = 'CelebA_WAE_results_sig1_eval/'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
if not os.path.isdir(result_dir + 'Random_results'):
    os.mkdir(result_dir + 'Random_results')
if not os.path.isdir(result_dir + 'Fixed_results'):
    os.mkdir(result_dir + 'Fixed_results')


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

        self.conv1 = BBBConv2d(self.n_channel, self.dim_h, kernel_size=5, stride=2, padding =2)
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
            #nn.BatchNorm1d(self.dim_h * 4),
            nn.ReLU(),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            #nn.BatchNorm1d(self.dim_h * 4),
            nn.ReLU(),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            #nn.BatchNorm1d(self.dim_h * 4),
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
encoder.train(mode = True)
decoder.train(mode = True)
discriminator.train(mode = True)
mse = nn.MSELoss()
mse_sum = nn.MSELoss(reduction= 'sum')
bcewl = nn.BCEWithLogitsLoss()
bcewl_sum =nn.BCEWithLogitsLoss(reduction= 'sum')
sig = nn.Sigmoid()

# Optimizers
enc_optim = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0.5, 0.999)) #betas=(0.5, 0.999)
dec_optim = optim.Adam(decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
dis_optim = optim.Adam(discriminator.parameters(), lr= 1e-4, betas=(0.5, 0.999))

# enc_scheduler = StepLR(enc_optim, step_size=5, gamma=0.5)
# dec_scheduler = StepLR(dec_optim, step_size=5, gamma=0.5)
# dis_scheduler = StepLR(dis_optim, step_size=5, gamma=0.5)


def denorm(x):
    out = (x + 1) / 2
    return out

def vi(kl, log_likelihood):
    # print("kl",(kl / len(train_loader)).item())
    # print("likelihood", log_likelihood.item())
    return   kl / len(train_loader) + log_likelihood

fixed_z = torch.randn(64, args.n_z) * args.sigma  # fixed noise
fixed_z = fixed_z.to(device)


for epoch in range(args.epochs):
    step = 0
    # enc_scheduler.step(epoch = 5)
    # dec_scheduler.step(epoch = 5)
    # dis_scheduler.step(epoch = 5)
    # if (epoch) == 5:
    #     enc_optim.param_groups[0]['lr'] /= 10
    #     dec_optim.param_groups[0]['lr'] /= 10
    #     dis_optim.param_groups[0]['lr'] /= 10
    #     print("learning rate change!")
    for images, _ in tqdm(train_loader):
        images = images.to(device)
        real_labels = torch.ones(images.size()[0], 1).to(device)
        fake_labels = torch.zeros(images.size()[0], 1).to(device)

        # ======== Train Generator ======== #

        free_params(decoder)
        free_params(encoder)
        frozen_params(discriminator)

        z_enc, kl_enc = encoder.probforward(images)
        d_enc = discriminator(z_enc)
        x_recon, kl_dec = decoder.probforward(z_enc)

        recon_loss = mse_sum(x_recon, images)
        d_loss = bcewl_sum(d_enc, real_labels)
        likelihood = recon_loss + args.LAMBDA * d_loss
        # print("d_loss", (args.LAMBDA * d_loss).item())
        # print("rec",recon_loss.item())
        enc_loss = vi(kl_enc, likelihood)
        dec_loss = vi(kl_dec, likelihood)

        enc_optim.zero_grad()
        enc_loss.backward(retain_graph=True)
        enc_optim.step()

        dec_optim.zero_grad()
        dec_loss.backward()  # retain_graph reservered?
        dec_optim.step()


        # ======== Train Discriminator ======== #

        frozen_params(decoder)
        frozen_params(encoder)
        free_params(discriminator)

        z_real = (torch.randn(images.size()[0], args.n_z) * args.sigma).to(device)  # images.size()[0] -> 100
        d_real = discriminator(z_real)

        z_enc, kl_enc = encoder.probforward(images)
        d_enc = discriminator(z_enc.detach())
        #print(sig(d_enc).data)
        d_real_loss = bcewl(d_real, real_labels)
        d_enc_loss = bcewl(d_enc, fake_labels)
        dis_loss = d_real_loss + d_enc_loss

        discriminator.zero_grad()
        dis_loss.backward()
        dis_optim.step()


        step += 1

        if (step + 1) % (len(train_loader) // 4 )== 0:
            print("Epoch: [%d/%d], Step: [%d/%d], dis_loss : %.4f, encoder_Loss : %.4f, decoder_Loss: %.4f , d_loss : %.4f" %
                  (epoch + 1, args.epochs, step + 1, len(train_loader), dis_loss.item(), enc_loss.item(), dec_loss.item(), (args.LAMBDA * d_loss).item()))

    print("Training finish!... save training results")
    torch.save(encoder.state_dict(), result_dir + 'encdoer_%d.pkl' %(epoch + 1))
    torch.save(decoder.state_dict(), result_dir + 'decoder_%d.pkl' %(epoch + 1))
    torch.save(discriminator.state_dict(), result_dir + 'discriminator_%d.pkl' %(epoch + 1))


    with torch.no_grad():
        if (epoch + 1) % 1 == 0:

            # test_iter = iter(test_loader)
            # test_data = next(test_iter)
            decoder.eval()
            z = (torch.randn(64, args.n_z)* args.sigma).to(device)
            x_sam_fixed, _ = decoder.probforward(fixed_z)
            x_sam, _ = decoder.probforward(z)
            save_image(denorm(x_sam.view(-1, args.n_channel, 64, 64)), result_dir + 'Random_results/sam_%d.png' % (epoch + 1))
            save_image(denorm(x_sam_fixed.view(-1, args.n_channel, 64, 64)), result_dir + 'Fixed_results/sam_fixed_%d.png' % (epoch + 1))

            # z_real = encoder(Variable(test_data[0]).to(device))
            # reconst = decoder(z_real).to(device).view(args.batch_size, 1, 28, 28)
            #
            # if not os.path.isdir('./data/reconst_images'):
            #     os.makedirs('data/reconst_images')
            #
            # save_image(test_data[0].view(args.batch_size, 1, 28, 28),
            #            './data/reconst_images/wae_gan_input_%d.png' % (epoch + 1))
            # save_image(reconst.data, './data/reconst_images/wae_gan_images_%d.png' % (epoch + 1))





images = []
for e in range(args.epochs):
    img_name = result_dir + 'Fixed_results/sam_fixed_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(result_dir + 'generation_animation.gif', images, fps=5)