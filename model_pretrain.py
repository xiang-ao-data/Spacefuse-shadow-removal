"""
This script is used to pretrain auto-encoder network in COCO dataset, and fine-tune in spacecraft dataset

the transmef net is limited to the image size which only process 256x256
"""

from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import numpy as np
from network import TransMEF_model, Densenet
from torchvision import transforms
from dataloader import Fusionset,dataset_weight
from loss import *
import time
import argparse

import copy
from tqdm import tqdm
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NWORKERS = 4

parser = argparse.ArgumentParser(description='auto encoder model pretrain')
parser.add_argument('--exp_name', type=str, default='TransMEF', help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--root', type=str, default='./coco', help='data path')
parser.add_argument('--save_path', type=str, default='./train_result', help=' pics save path')
parser.add_argument('--model_path', type=str, default='./checkpoint', help='model save path')
parser.add_argument('--ssl_transformations', type=bool, default=True, help='use ssl_transformations or not')
parser.add_argument('--miniset', type=bool, default=False, help='to choose a mini dataset')
parser.add_argument('--minirate', type=float, default=0.2, help='to detemine the size of a mini dataset')
parser.add_argument('--seed', type=int, default=3, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--epoch', type=int, default=100, help='training epoch')
parser.add_argument('--batch_size', type=int, default=4, help='batchsize')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lamda_ssim', type=float, default=20, help='weight of the SSIM loss')
parser.add_argument('--lamda_tv', type=float, default=20, help='weight of the tv loss')
parser.add_argument('--lamda_mse', type=float, default=1, help='weight of the mse loss') # or use perceptual loss
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--summary_name', type=str, default='TransMEF_alldata_ssl_transformations_',
                    help='Name of the tensorboard summmary')
parser.add_argument("--encoder_ckpt", type=str, default='./checkpoint/Transmef_epoch_30_tensor.pth',help='auto-encoder checkpoint path')

args = parser.parse_args()
writer = SummaryWriter(comment=args.summary_name)

# initial

toPIL = transforms.ToPILImage()
np.random.seed(1)  # to get the same images and leave it fixed
torch.manual_seed(args.seed)
device = torch.device("cuda:" + str(args.gpus[0]) if torch.cuda.is_available() else "cpu")

train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                     torchvision.transforms.RandomCrop(256),
                                                     torchvision.transforms.RandomHorizontalFlip()
                                                     ])

dataset = Fusionset(args, transform=train_augmentation, gray=True, partition='train',
                    ssl_transformations=args.ssl_transformations)

# Creating data indices for training and validation splits:
train_indices = dataset.train_ind
val_indices = dataset.val_ind

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)  # sampler will assign the whole data according to batchsize.
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size,
                          sampler=train_sampler, drop_last=True)
val_loader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size,
                        sampler=valid_sampler)

# model
model = TransMEF_model().to(device)
# model = Densenet().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd)
scheduler = CosineAnnealingLR(optimizer, args.epoch)

MEF_LOSS = SSIM()
MSE_LOSS = nn.MSELoss()
Tv_loss = TV_Loss()

if torch.cuda.is_available():
    MEF_LOSS = MEF_LOSS.cuda()
    MSE_LOSS = MEF_LOSS.cuda()
    Tv_loss = Tv_loss.cuda()

loss_train = []
loss_val = []
model.train()

def train():
    for epoch in tqdm(range(args.epoch)):
        total_epoch_loss =0.

        for index, image in enumerate(train_loader):
            img_orig = image[0].to(device)
            optimizer.zero_grad()
            img_recon = model(img_orig.float())
            img_recon_save = toPIL(img_recon[0].squeeze(0).detach().cpu())

            if index % 1000 == 0:
                img_recon_save.save(os.path.join(args.save_path, 'epoch' + str(epoch) + '_' + str(
                    index) + '_coco_train.png'))

            mef_loss = (1-MEF_LOSS(img_orig,img_recon))
            tv_loss = Tv_loss(img_orig,img_recon)
            mse_loss = MSE_LOSS(img_orig,img_recon)

            total_loss = args.lamda_ssim*mef_loss+args.lamda_tv*tv_loss+args.lamda_mse*mse_loss
            total_epoch_loss += total_loss
            total_loss.backward()
            optimizer.step()

        print('Epoch:[%d/%d]-----Train------ LOSS:%.4f' % (
            epoch, args.epoch, total_epoch_loss / (len(train_loader))))

        # ==================
        # Model Validation
        # ==================
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for index, image in enumerate(val_loader):
                img_orig = image[0].to(device)
                img_recon = model(img_orig.float())
                bright_mef_loss = 1 - MEF_LOSS(img_orig, img_recon)
                bright_tv_loss = Tv_loss(img_orig, img_recon)
                bright_mse_loss = MSE_LOSS(img_orig, img_recon)
                all_bright_loss = args.lamda_ssim * bright_mef_loss + args.lamda_tv * bright_tv_loss + args.lamda_mse * bright_mse_loss
                total_loss =  all_bright_loss
                total_val_loss+= total_loss
            print('Epoch:[%d/%d]-----Val------ LOSS:%.4f' % (
                epoch, args.epoch, total_val_loss / (len(val_loader))))

            loss_val.append(total_val_loss / (len(val_loader)))

        if epoch % 10 ==0:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
            }
            torch.save(state, os.path.join(args.model_path,
                                          'epoch_' + str(epoch) + '_' + str(loss_val[epoch]) + '.pth'))


def fine_tune():
    Transmef = TransMEF_model().to(device)
    state_dict = torch.load(args.encoder_ckpt, map_location='cuda:0')['model']
    Transmef.load_state_dict(state_dict)
    model = Transmef
    model.train()
    dataset = dataset_weight()
    dataloader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size,
                           drop_last=True)
    for epoch in tqdm(range(1000)):
        total_epoch_loss = 0.

        for index, image in enumerate(dataloader):
            img1 = image[0].to(device)
            img2 = image[1].to(device)
            gt = image[2].to(device)
            optimizer.zero_grad()
            img_recon = model(img1.float())
            img2_recon = model(img2.float())
            gt_recon = model(gt.float())

            img_recon_save = toPIL(img_recon[0].squeeze(0).detach().cpu())

            if index % 10 == 0:
                img_recon_save.save(os.path.join(args.save_path, 'epoch' + str(epoch) + '_' + str(
                    index) + '_fine_tune.png'))


            mef_loss = (1 - MEF_LOSS(img1, img_recon)) + (1 - MEF_LOSS(img2, img2_recon)) +(1 - MEF_LOSS(gt, gt_recon))
            tv_loss = Tv_loss(img1, img_recon) + Tv_loss(img2, img2_recon) +Tv_loss(gt, gt_recon)
            mse_loss = MSE_LOSS(img1, img_recon) + MSE_LOSS(img2, img2_recon) + MSE_LOSS(gt, gt_recon)

            total_loss = args.lamda_ssim * mef_loss + args.lamda_tv * tv_loss + args.lamda_mse * mse_loss
            total_epoch_loss += total_loss
            total_loss.backward()
            optimizer.step()

        print('Epoch:[%d/%d]-----Train------ LOSS:%.4f' % (
            epoch, 1000, total_epoch_loss / (len(train_loader))))

        if epoch % 100 ==0:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
            }
            torch.save(state, os.path.join(args.model_path,
                                          'fine_epoch_' + str(epoch) + '.pth'))

if __name__ == "__main__":
    fine_tune()