
from math import exp
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import os

""""
 This script include four type of loss, MEF-SSIM loss ,perceptual loss ,gradient loss, TV_loss

 the MEF-SSIM loss is borrowed from deepfuse
"""

"""
MEF-SSIM LOSS
"""
L2_NORM = lambda b: torch.sqrt(torch.sum((b + 1e-8) ** 2))

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """
            Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])

        # result tensor([9.1594e-01, 7.9480e-02, 4.4221e-03, 1.5775e-04, 3.6084e-06, 5.2921e-08,
        #         4.9764e-10, 3.0005e-12, 1.1600e-14, 2.8752e-17, 4.5697e-20])
        return gauss / gauss.sum()

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
            Compute the SSIM for the given two image
            The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def create_window(self, window_size, channel):
        """
            Create the gaussian window
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)  # size [11,1]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # .t() tansform [11,1] to [1,11]
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class MEF_SSIM_Loss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        """
            Constructor
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """
            Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])

        # result tensor([9.1594e-01, 7.9480e-02, 4.4221e-03, 1.5775e-04, 3.6084e-06, 5.2921e-08,
        #         4.9764e-10, 3.0005e-12, 1.1600e-14, 2.8752e-17, 4.5697e-20])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        """
            Create the gaussian window
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)  # size [11,1]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # .t() tansform [11,1] to [1,11]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
            Compute the SSIM for the given two image
            The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def w_fn(self, y):
        """
            Return the weighting function that MEF-SSIM defines
            We use the power engery function as the paper describe: https://ece.uwaterloo.ca/~k29ma/papers/15_TIP_MEF.pdf

            Arg:    y   (torch.Tensor)  - The structure tensor
            Ret:    The weight of the given structure
        """
        out = torch.sqrt(torch.sum(y ** 2))
        return out

    def forward(self, y_1, y_2, y_f):
        """
            Compute the MEF-SSIM for the given image pair and output image
            The y_1 and y_2 can exchange

            Arg:    y_1     (torch.Tensor)  - The LDR image
                    y_2     (torch.Tensor)  - Another LDR image in the same stack
                    y_f     (torch.Tensor)  - The fused HDR image
            Ret:    The loss value
        """
        miu_y = (y_1 + y_2) / 2  # due to the special issue for spacefuse , this should change

        # Get the c_hat
        c_1 = L2_NORM(y_1 - miu_y)
        c_2 = L2_NORM(y_2 - miu_y)
        c_hat = torch.max(torch.stack([c_1, c_2]))

        # Get the s_hat
        s_1 = (y_1 - miu_y) / L2_NORM(y_1 - miu_y)
        s_2 = (y_2 - miu_y) / L2_NORM(y_2 - miu_y)
        s_bar = (self.w_fn(y_1) * s_1 + self.w_fn(y_2) * s_2) / (self.w_fn(y_1) + self.w_fn(y_2))
        s_hat = s_bar / L2_NORM(s_bar)

        # =============================================================================================
        # < Get the y_hat >
        #
        # Rather to output y_hat, we shift it with the mean of the over-exposure image and mean image
        # The result will much better than the original formula
        # =============================================================================================
        y_hat = c_hat * s_hat
        y_hat += (y_2 + miu_y) / 2

        # Check if need to create the gaussian window
        (_, channel, _, _) = y_hat.size()
        if channel == self.channel and self.window.data.type() == y_hat.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(y_f.get_device())
            window = window.type_as(y_hat)
            self.window = window
            self.channel = channel

        # Compute SSIM between y_hat and y_f
        score = self._ssim(y_hat, y_f, window, self.window_size, channel, self.size_average)
        return 1 - score


"""
Perceptual loss
"""


class perceputual_loss(nn.Module):
    def __init__(self, opt):
        super(perceputual_loss, self).__init__()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.opt = opt

    def compute_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)

        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    mean = mean.cuda()
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        if opt.vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h

        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)

        conv5_3 = self.conv5_3(relu5_2)
        relu5_3 = F.relu(conv5_3, inplace=True)

        if opt.vgg_choose == "conv4_3":
            return conv4_3
        elif opt.vgg_choose == "relu4_2":
            return relu4_2
        elif opt.vgg_choose == "relu4_1":
            return relu4_1
        elif opt.vgg_choose == "relu4_3":
            return relu4_3
        elif opt.vgg_choose == "conv5_3":
            return conv5_3
        elif opt.vgg_choose == "relu5_1":
            return relu5_1
        elif opt.vgg_choose == "relu5_2":
            return relu5_2
        elif opt.vgg_choose == "relu5_3" or "maxpool":
            return relu5_3


def load_vgg16(model_dir):
    vgg = Vgg16()
    # vgg.cuda()
    vgg.cuda()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    # vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg


"""
Gradient loss
"""


class Gradient_net(nn.Module):
    def __init__(self):
        super(Gradient_net, self).__init__()
        kenel_x = [[-1., 0, 1.], [-2., 0, 2.], [-1., 0, 1.]]
        kenel_x = torch.FloatTensor(kenel_x).unsqueeze(0).unsqueeze(0).to("cuda")
        kenel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kenel_y = torch.FloatTensor(kenel_y).unsqueeze(0).unsqueeze(0).to("cuda")
        self.weight_x = nn.Parameter(data=kenel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kenel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)  # get the gradient map ,but not loss
        return gradient


def gradient(x):
    model = Gradient_net().cuda()
    gra = model.forward(x)
    gra = torch.mean(gra)
    return gra

"""
TV loss
"""

class TV_Loss(torch.nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IF):
        r = IA - IF
        h = r.shape[2]
        w = r.shape[3]
        tv1 = torch.pow((r[:, :, 1:, :] - r[:, :, :h - 1, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:] - r[:, :, :, :w - 1]), 2).mean()
        return tv1 + tv2




