import argparse
import statsmodels.stats.api as sms
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from torch.utils.data import random_split
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import saliency.core as saliency

from skimage.metrics import structural_similarity as ssim

import wandb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

x_range = (1-0.406)/0.225+0.485/0.229

def calcSimpleGrad(net, images, labels):
    images1 = images.clone().requires_grad_()
    logits1 = net(images1)
    logits1 = logits1.gather(1, labels.view(-1, 1)).squeeze().sum()
    net.zero_grad()
    logits1.backward()
    return images1.grad

def evaluation_smoothed_saliency(net1, dataLoader, device, sigma=0.15):
    total = 0
    net1.eval()
    sampleNum = 100
    sigma = sigma*x_range
    diff_L2 = 0
    for data in dataLoader:
        images, labels = data[0].to(device), data[1].to(device)
        saliency1 = torch.zeros_like(images)
        for t in range(sampleNum):
            saliency1 += calcSimpleGrad(net1, images+torch.normal(0,sigma,size=images.shape).to(device), labels)
        saliency1 = saliency1 / sampleNum
        ground_truth = calcSimpleGrad(net1, images, labels)
        diff_L2 += float(torch.norm((saliency1-ground_truth).reshape((images.shape[0],-1)),dim=1).sum())
        total += images.shape[0]
    return diff_L2/total

def calcIntegratedGrad(net, images, labels):
    sampleNum = 20
    x_baseline = torch.zeros_like(images)
    saliency1 = torch.zeros_like(images)
    for alpha in np.linspace(0, 1, sampleNum):
        images1 = (x_baseline + alpha * (images.clone()-x_baseline)).requires_grad_()
        logits1 = net(images1)
        logits1 = logits1.gather(1, labels.view(-1, 1)).squeeze().sum()
        net.zero_grad()
        logits1.backward()
        saliency1 += images1.grad
    return saliency1 * (images.clone()-x_baseline) / sampleNum

def evaluation_smoothed_integrated_saliency(net1, dataLoader, device, sigma=0.15):
    total = 0
    net1.eval()
    sampleNum = 100
    sigma = sigma*x_range
    diff_L2 = 0
    for data in dataLoader:
        images, labels = data[0].to(device), data[1].to(device)
        saliency1 = torch.zeros_like(images)
        for t in range(sampleNum):
            saliency1 += calcIntegratedGrad(net1, images+torch.normal(0,sigma,size=images.shape).to(device), labels)
        saliency1 = saliency1 / sampleNum
        ground_truth = calcIntegratedGrad(net1, images, labels)
        diff_L2 += float(torch.norm((saliency1-ground_truth).reshape((images.shape[0],-1)),dim=1).sum())
        total += images.shape[0]
    return diff_L2/total

import matplotlib.pyplot as plt
valdir = os.path.join('/share/datasets/ImageNet/Data/CLS-LOC/', 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
# val_dataset, no_use = random_split(val_dataset, [32, len(val_dataset)-32])
s = [i for i in range(len(val_dataset))]
random.seed(10007)
random.shuffle(s)
val_dataset = Subset(val_dataset, s[:32])
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, shuffle=False,
    num_workers=16, pin_memory=True)
plt.figure(1)

def main(exp, cnt, type: int):
    def getDirectoryStr(exp, i):
        if exp.find('_')==-1:
            return exp + '_p' + str(i)
        return exp[:exp.find('_')] + '_p' + str(i) + exp[exp.find('_'):]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if exp != '':
        print('exp =',exp,getDirectoryStr(exp,1)+'/model_best.pth.tar')
        nets = []
        for i in range(cnt):
            checkpoint = torch.load(getDirectoryStr(exp,i)+'/model_best.pth.tar')
            net = models.__dict__['resnet50']()
            net = torch.nn.DataParallel(net).cuda()
            net.load_state_dict(checkpoint['state_dict'])
            nets.append(net)
    else:
        from torchvision.models import resnet50, ResNet50_Weights
        net1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        net1 = torch.nn.DataParallel(net1).cuda()
        net2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        net2 = torch.nn.DataParallel(net2).cuda()
    print('model loaded')
    x = []
    y = []
    yl = []
    yr = []
    for sigma in [x**2 for x in np.arange(0, np.sqrt(0.4)+0.001, np.sqrt(0.4)/10)]:
        vals = []
        for i in range(len(nets)):
            if type==0:
                vals.append(evaluation_smoothed_saliency(nets[i], val_loader, device, sigma=sigma))
            else:
                vals.append(evaluation_smoothed_integrated_saliency(nets[i], val_loader, device, sigma=sigma))
        diff = sum(vals)/len(vals)
        lo = sms.DescrStatsW(vals).tconfint_mean()[0]
        hi = sms.DescrStatsW(vals).tconfint_mean()[1]
        x.append(sigma)
        y.append(diff)
        yl.append(lo)
        yr.append(hi)
        print(exp, sigma, diff, hi-diff)
    plt.plot(x, y, label=exp[:exp.find('_')]+' samples', marker='o', markersize=3)
    plt.fill_between(x, yl, yr, alpha=.3)

if __name__ == '__main__':
    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    main('50000_wd0.0001_RN50', 4, type=0)
    main('150000_wd0.0001_RN50', 4, type=0)
    main('500000_wd0.0001_RN50', 4, type=0)
    plt.xlabel("$\sigma/(x_{max}-x_{min})$", fontsize=12)
    plt.ylabel("saliency map fidelity", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Simple-Grad", fontsize=13)
    plt.legend()
    
    plt.subplot(122)
    main('500000_wd0.0001_RN50', 4, type=1)
    main('50000_wd0.0001_RN50', 4, type=1)
    main('150000_wd0.0001_RN50', 4, type=1)
    plt.xlabel("$\sigma/(x_{max}-x_{min})$", fontsize=12)
    plt.ylabel("saliency map fidelity", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Integrated-Grad", fontsize=13)
    plt.legend()
    plt.savefig("fidelity_ImageNet_plot_sigma.pdf", bbox_inches='tight')