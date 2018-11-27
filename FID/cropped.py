import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils as vutils
import torchvision.transforms as transforms

# root path depends on your computer
root = './celebA_1w'
save_root = './dataset_cropped_1w/'

transform = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),

])

dset = datasets.ImageFolder(root, transform)
dataloader = torch.utils.data.DataLoader(dset, batch_size= 1, shuffle=True)
if not os.path.isdir(save_root):
    os.mkdir(save_root)


for i, data in enumerate(dataloader):
    #print(data[0])
    vutils.save_image(data[0], '%s%d.png' %(save_root, i), normalize=False)
    if (i % 1000) == 0:
        print('%d images complete' % i)