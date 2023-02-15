"""
This script is used to train full network on MIAS dataset, we provide three train option.
1.training the full pipeline
2.training without self-supervised loss
3.training by directly fusing two input images
author : xiangao
time: 2023-2-14
"""

from torch.optim.lr_scheduler import CosineAnnealingLR
from network import Sub_net, TransMEF_model, Fusion_net
from dataloader import dataset_weight, val_weight
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from visualizer import Visualizer
import os
from loss import shadow_loss
from torchvision.transforms.functional import normalize
import numpy as np
from PIL import Image
from loss import SSIM,TV_Loss
import argparse


# train set
parser = argparse.ArgumentParser(description='Sub-net training')
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--epoch', type=int, default=1500, help='training epochs')
parser.add_argument('--val_epoch', type=int, default=50, help='validate epochs')
parser.add_argument('--save_epoch', type=int, default=100, help='save epoch')
parser.add_argument('--niter', type=int, default=20000, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=20000, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--option', type=str, default="full", help='three train options: full, no_uns,d_fuse')


# path option
parser.add_argument("--encoder_ckpt", type=str, default='./checkpoint/Transmef_epoch_30_tensor.pth',help='auto-encoder checkpoint path')
parser.add_argument("--model_path", type=str, default='./ckpt/',help='subnet model path')
parser.add_argument("--val_path", type=str, default='./val_temp/',help='use to generata temp validate img')

# visualiza
parser.add_argument("--enable_vis", action='store_true', default=True,help="use visdom for visualization")
parser.add_argument("--vis_port", type=str, default='13570',help='port for visdom')
parser.add_argument("--vis_env", type=str, default='pytorch', help='env for visdom')
parser.add_argument("--vis_num_samples", type=int, default=8, help='number of samples for visualization (default: 8)')


opts = parser.parse_args()

def frozen(model):
    for name, parameter in model.named_parameters():
        parameter.required_grad = False

    return model

# load and initial network, sub-net, auto-encoder
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Transmef = TransMEF_model().to(device)
state_dict = torch.load(opts.encoder_ckpt, map_location='cuda:0')['model']
Transmef.load_state_dict(state_dict)
Transmef.eval()
Subnet = Sub_net().to(device)

# load spacefuse dataset
transform = transforms.Compose([
     transforms.ToTensor(),
 ])
dataset = dataset_weight(transform=transform)
dataload = DataLoader(dataset,batch_size=opts.batch_size,num_workers=4,shuffle=True)
optimizer = optim.Adam(Subnet.parameters(), lr=opts.lr, weight_decay = opts.wd)
scheduler = CosineAnnealingLR(optimizer, opts.epoch)
_pil_gray = transforms.ToPILImage()

# loss
mse_loss = nn.MSELoss().to(device)
ssim_loss = SSIM().to(device)
tv_loss = TV_Loss().to(device)


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x

def encode(img1,img2,model):
    model.eval()
    with torch.no_grad():
        f_1 = model.encoder(img1)
        f_2 = model.encoder(img2)
    return f_1, f_2

def decode(img,model):
    model.eval()
    with torch.no_grad():
        out = model.decoder(img)
    return out


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def train():
    Subnet.train()
    cur_itrs = 0
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    loss_val =[]
    best_score = 100.
    for epoch in tqdm(range(opts.epoch)):
        total_epoch_loss = 0.
        for index, image in enumerate(dataload):
            cur_itrs +=1
            img = image[0].to(device)
            relate_img = image[1].to(device)
            gt = image[2].to(device)
            s_1 = image[3].to(device) * relate_img
            s_2 = image[4].to(device) *img
            optimizer.zero_grad()
            feature1,feature2 = encode(img,relate_img,Transmef)
            shadow_inf_1, shadow_inf_2 = encode(s_1,s_2,Transmef)

            # produce fusion weight by subnet
            if opts.option == "full" or "no_uns":
                weight1 = Subnet(img, s_1)
                weight2 = Subnet(relate_img, s_2)
                feature_fuse_1 = feature1 * weight1 + shadow_inf_1 * (1 - weight1)
                feature_fuse_2 = feature2 * weight2 + shadow_inf_2 * (1 - weight2)
                feature_fuse = (feature_fuse_2 + feature_fuse_1)
                out1 = Transmef.decoder(feature_fuse_1)
                out2 = Transmef.decoder(feature_fuse_2)
                out = Transmef.decoder(feature_fuse)

            else:
                weight = Subnet(img, relate_img)
                feature_fuse = feature1*(weight)+feature2*(1-weight)
                out = Transmef.decoder(feature_fuse)

            if opts.option == "full":
                s_loss_ssim = (1 - ssim_loss(out,gt))
                uns_loss_ssim = (1 - ssim_loss(out1,out2))
                s_loss_mse = (mse_loss(out,gt))*20
                uns_loss_mse = (mse_loss(out1,out2))*20
                loss = s_loss_ssim+s_loss_mse+uns_loss_ssim+uns_loss_mse
            else:
                mse = (mse_loss(out,gt))*20
                ssim = (1 - ssim_loss(out,gt))
                loss = mse+ssim

            loss.backward()
            optimizer.step()
            np_loss = loss.detach().cpu().numpy()
            #sloss = loss_ssim_1.detach().cpu().numpy()
            if vis is not None:
                vis.vis_scalar('supervised_Loss', cur_itrs, np_loss)
                #vis.vis_scalar('unsupervised_Loss', cur_itrs, s_loss)
            total_epoch_loss += (np_loss)


            if epoch % 50 ==0: # save train example
                out_img = out[0]
                out_img = out_img.squeeze(0).detach().cpu()
                out_img = transforms.ToPILImage()(out_img)
                out_img.save(os.path.join("./train_result", "epoch_" + str(epoch) + "_" + str(index) + ".png"))

        loss_val.append(total_epoch_loss)
        print("Epoch %d/%d, Loss=%f" %
              (epoch,  opts.epoch, loss.item()))

        if epoch % opts.save_epoch == 0: #save checkpoint
            state_s = {
                'epoch': epoch,
                'model': Subnet.state_dict(),
            }
            torch.save(state_s, os.path.join(opts.model_path,
                                      'Subnet'+opts.option + str(epoch) + '_' + str(loss_val[epoch]) + '.pth'))


        if epoch % opts.val_epoch == 0:
            Subnet.eval()
            val_set = val_weight(transform=transform)
            dataloader = DataLoader(val_set, batch_size=opts.val_batch_size, num_workers=4)
            score = 0
            iter = 0
            for index, image in enumerate(dataloader):
                cur_itrs += 1
                img = image[0].to(device)
                relate_img = image[1].to(device)
                gt = image[2].to(device)
                s_1 = image[3].to(device) * relate_img
                s_2 = image[4].to(device) * img
                feature1, feature2 = encode(img, relate_img, Transmef)
                shadow_inf_1, shadow_inf_2 = encode(s_1, s_2, Transmef)

                shape = feature2.shape
                if opts.option == "full" or "no_uns":
                    weight1 = Subnet(img, s_1)
                    weight2 = Subnet(relate_img, s_2)
                    feature_fuse_1 = feature1 * weight1 + shadow_inf_1 * (1 - weight1)
                    feature_fuse_2 = feature2 * weight2 + shadow_inf_2 * (1 - weight2)
                    feature_fuse = (feature_fuse_2 + feature_fuse_1)
                    out1 = Transmef.decoder(feature_fuse_1)
                    out2 = Transmef.decoder(feature_fuse_2)
                    out = Transmef.decoder(feature_fuse)

                else:
                    weight = Subnet(img, relate_img)
                    feature_fuse = feature1 * (weight) + feature2 * (1 - weight)
                    out = Transmef.decoder(feature_fuse)

                if opts.option == "full":
                    s_loss_ssim = (1 - ssim_loss(out, gt))
                    uns_loss_ssim = (1 - ssim_loss(out1, out2))
                    s_loss_mse = (mse_loss(out, gt)) * 20
                    uns_loss_mse = (mse_loss(out1, out2)) * 20
                    loss = s_loss_ssim + s_loss_mse + uns_loss_ssim + uns_loss_mse
                else:
                    mse = (mse_loss(out, gt)) * 20
                    ssim = (1 - ssim_loss(out, gt))
                    loss = mse + ssim

                np_loss = loss.detach().cpu().numpy()
                score+= np_loss

                for i in range(shape[0]): # save validation results
                    out_img =out[i]
                    out_img = out_img.squeeze(0).detach().cpu()
                    out_img = transforms.ToPILImage()(out_img)
                    out_img.save(os.path.join("./val_temp","epoch_"+str(epoch)+"_"+str(iter*4+i)+".png"))
                iter+=1

            vis.vis_scalar('Val_Loss', epoch, score/len(dataloader))


            if score < best_score:
                best_score = score
                state = {
                    'epoch': epoch,
                    'model': Subnet.state_dict(),
                }
                torch.save(state, os.path.join(opts.model_path,
                                               'best'  + '.pth'))

            Subnet.train()
        scheduler.step()


if __name__ == "__main__":
    train()
















