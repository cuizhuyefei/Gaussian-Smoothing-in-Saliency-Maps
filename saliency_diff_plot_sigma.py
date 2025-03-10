import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import argparse
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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', nargs='?', default='/share/datasets/ImageNet/Data/CLS-LOC/',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--part', default=None, type=int,
                    help='part 0 or 1.')
parser.add_argument('--train-size', default=None, type=int,
                    help='The number of data points each part contains.')
parser.add_argument('--model-path', default=None, type=str,
                    help='The number of data points each part contains.')
best_acc1 = 0
x_range = (1-0.406)/0.225+0.485/0.229

def evaluation_smooth_saliency(net1, net2, dataLoader, device, sigma=0.15):
    total = 0
    net1.eval()
    net2.eval()
    sampleNum = 100
    original_sigma = sigma
    sigma = sigma*x_range
    diff_L2 = 0
    sal_L2 = 0
    for data in dataLoader:
        images, labels = data[0].to(device), data[1].to(device)
        
        saliency1 = torch.zeros_like(images)
        for t in range(sampleNum):
            images1 = (images.clone()+torch.normal(0,sigma,size=images.shape).to(device)).requires_grad_()
            logits1 = net1(images1)
            logits1 = logits1.gather(1, labels.view(-1, 1)).squeeze().sum()
            net1.zero_grad()
            logits1.backward()
            saliency1 += images1.grad
        saliency1 /= sampleNum

        saliency2 = torch.zeros_like(images)
        for t in range(sampleNum):
            images2 = (images.clone()+torch.normal(0,sigma,size=images.shape).to(device)).requires_grad_()
            logits2 = net2(images2)
            logits2 = logits2.gather(1, labels.view(-1, 1)).squeeze().sum()
            net2.zero_grad()
            logits2.backward()
            saliency2 += images2.grad
        saliency2 /= sampleNum

        mask = torch.ones_like(labels,dtype=torch.bool)
        cnt = (images1.grad-images2.grad)[mask].shape[0]
        
        sal_L2 += float(torch.norm(saliency1[mask].reshape((cnt,-1)),dim=1).sum())
        diff_L2 += float(torch.norm(((saliency1-saliency2)[mask]).reshape((cnt,-1)),dim=1).sum())
        total += cnt
    # print('Mean saliency difference & saliency norm & num-acc', round(diff_L2/total,3), round(sal_L2/total,3), total)
    # print('saliency diff & normalized saliency diff & ssim & topk intersection', round(diff_L2/total,3), round(norm_diff_L2 / total, 3), round(ssim_value / total, 3), round(topk_intersection / total, 3))
    return diff_L2/total

def evaluation_smoothed_integrated_saliency(net1, net2, dataLoader, device, sigma=0.15):
    total = 0
    net1.eval()
    net2.eval()
    sampleNum = 100
    sigma = sigma*x_range
    diff_L2 = 0
    for data in dataLoader:
        images, labels = data[0].to(device), data[1].to(device)
        saliency1 = torch.zeros_like(images)
        saliency2 = torch.zeros_like(images)
        for t in range(sampleNum):
            def calcIntegratedGrad(net, images):
                sampleNum = 20
                x_baseline = torch.zeros_like(images)
                saliency1 = torch.zeros_like(images)
                for alpha in np.linspace(0, 1, sampleNum):
                    images1 = (x_baseline + alpha * (images.clone()-x_baseline)).requires_grad_()
                    logits1 = net(images1)
                    logits1 = logits1.gather(1, labels.view(-1, 1)).squeeze().sum()
                    # print('qwq', logits1, images1.shape)
                    net.zero_grad()
                    logits1.backward()
                    saliency1 += images1.grad
                return saliency1 * (images.clone()-x_baseline) / sampleNum
            saliency1 += calcIntegratedGrad(net1, images+torch.normal(0,sigma,size=images.shape).to(device))
            saliency2 += calcIntegratedGrad(net2, images+torch.normal(0,sigma,size=images.shape).to(device))
        saliency1 = saliency1 / sampleNum
        saliency2 = saliency2 / sampleNum
        diff_L2 += float(torch.norm((saliency1-saliency2).reshape((images.shape[0],-1)),dim=1).sum())
        total += images.shape[0]
    return diff_L2/total

def calculate_smoothed_saliency(net1, dataLoader, device, sigma=0.15):
    net1.eval()
    sampleNum = 100
    sigma = sigma*x_range
    list_res = []
    for data in dataLoader:
        images, labels = data[0].to(device), data[1].to(device)
        saliency1 = torch.zeros_like(images)
        for t in range(sampleNum):
            images1 = (images.clone()+torch.normal(0,sigma,size=images.shape).to(device)).requires_grad_()
            logits1 = net1(images1)
            logits1 = logits1.gather(1, labels.view(-1, 1)).squeeze().sum()
            net1.zero_grad()
            logits1.backward()
            saliency1 += images1.grad
        saliency1 = saliency1 / sampleNum
        list_res.append(saliency1.reshape((images.shape[0],-1)))
    return torch.cat(list_res, dim=0)

def calculate_smoothed_integrated_saliency(net1, dataLoader, device, sigma=0.15):
    net1.eval()
    sampleNum = 100
    sigma = sigma*x_range
    list_res = []
    for data in dataLoader:
        images, labels = data[0].to(device), data[1].to(device)
        saliency1 = torch.zeros_like(images)
        for t in range(sampleNum):
            def calcIntegratedGrad(net, images):
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
            saliency1 += calcIntegratedGrad(net1, images+torch.normal(0,sigma,size=images.shape).to(device))
        saliency1 = saliency1 / sampleNum
        list_res.append(saliency1.reshape((images.shape[0],-1)))
    return torch.cat(list_res, dim=0)

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

    for sigma in np.arange(0, 0.41, 0.05):
        vals = []
        for i in range(len(nets)):
            for j in range(i+1, len(nets)):
                if i==0 and j==1 or i==2 and j==3 or i==0 and j==2 or i==1 and j==3:
                    if type==0:
                        vals.append(evaluation_smooth_saliency(nets[i], nets[j], val_loader, device, sigma=sigma))
                    else:
                        vals.append(evaluation_smoothed_integrated_saliency(nets[i], nets[j], val_loader, device, sigma=sigma))
        diff = sum(vals)/len(vals)
        lo = sms.DescrStatsW(vals).tconfint_mean()[0]
        hi = sms.DescrStatsW(vals).tconfint_mean()[1]

        x.append(sigma)
        y.append(diff)
        yl.append(lo)
        yr.append(hi)
        print(exp, sigma, diff, hi-diff)
    plt.plot(x, y, label=exp[:exp.find('_')]+' samples', marker='o', linestyle='-', markersize=3)
    plt.fill_between(x, yl, yr, alpha=.3)

if __name__ == '__main__':
    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    main('50000_wd0.0001_RN50', 4, type=0)
    main('150000_wd0.0001_RN50', 4, type=0)
    main('500000_wd0.0001_RN50', 4, type=0)
    plt.xlabel("$\sigma/(x_{max}-x_{min})$", fontsize=12)
    plt.ylabel("saliency map difference", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Simple-Grad", fontsize=13)
    plt.legend()
    
    plt.subplot(122)
    main('50000_wd0.0001_RN50', 4, type=1)
    main('150000_wd0.0001_RN50', 4, type=1)
    main('500000_wd0.0001_RN50', 4, type=1)
    plt.xlabel("$\sigma/(x_{max}-x_{min})$", fontsize=12)
    plt.ylabel("saliency map difference", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Integrated-Grad", fontsize=13)
    plt.legend()
    plt.savefig("plot_sigma.pdf", bbox_inches='tight')