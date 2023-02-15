from __future__ import print_function
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
from glob import glob
import os
import copy
from PIL import Image
import torch
import random
import argparse
from imgaug import augmenters as iaa
from os.path import join
import cv2
from tqdm import tqdm
from shadow_detection import shadow_detection

root = '/home/xiangao/PycharmProjects/spacefuse/data/train2014/'

sometimes = lambda aug: iaa.Sometimes(0.8, aug)
np.random.seed(2)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # error: image file is truncated (9 bytes not processed)


def local_pixel_shuffling(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 10
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        window = orig_image[noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x,
                                 block_noise_size_y))
        image_temp[noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = window
    local_shuffling_x = image_temp

    return local_shuffling_x


def global_patch_shuffling(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)

    img_rows, img_cols = x.shape
    num_block = 10
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)

        noise_x1 = random.randint(0, img_rows - block_noise_size_x)
        noise_y1 = random.randint(0, img_cols - block_noise_size_y)

        noise_x2 = random.randint(0, img_rows - block_noise_size_x)
        noise_y2 = random.randint(0, img_cols - block_noise_size_y)

        window1 = orig_image[noise_x1:noise_x1 + block_noise_size_x,
                  noise_y1:noise_y1 + block_noise_size_y]
        window2 = orig_image[noise_x2:noise_x2 + block_noise_size_x,
                  noise_y2:noise_y2 + block_noise_size_y]

        window_tmp = window1
        window1 = window2
        window2 = window_tmp

        image_temp[noise_x1:noise_x1 + block_noise_size_x,
        noise_y1:noise_y1 + block_noise_size_y] = window1
        image_temp[noise_x2:noise_x2 + block_noise_size_x,
        noise_y2:noise_y2 + block_noise_size_y] = window2

    local_shuffling_x = image_temp

    return local_shuffling_x


def brightness_aug(x, gamma):
    aug_brightness = iaa.Sequential(sometimes(iaa.GammaContrast(gamma=gamma)))
    aug_image = aug_brightness(images=x)
    return aug_image


def bright_transform(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 10
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        window = orig_image[noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y]
        window = brightness_aug(window, 3 * np.random.random_sample())

        image_temp[noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = window
    bright_transform_x = image_temp

    return bright_transform_x


def fourier_broken(x, nb_rows, nb_cols):
    aug_a = iaa.GaussianBlur(sigma=0.5)
    aug_p = iaa.Jigsaw(nb_rows=nb_rows, nb_cols=nb_cols, max_steps=(1, 5))
    fre = np.fft.fft2(x)
    fre_a = np.abs(fre)
    fre_p = np.angle(fre)
    fre_a_normalize = fre_a / (np.max(fre_a) + 0.0001)
    fre_p_normalize = fre_p
    fre_a_trans = aug_a(image=fre_a_normalize)
    fre_p_trans = aug_p(image=fre_p_normalize)
    fre_a_trans = fre_a_trans * (np.max(fre_a) + 0.0001)
    fre_p_trans = fre_p_trans
    fre_recon = fre_a_trans * np.e ** (1j * (fre_p_trans))
    img_recon = np.abs(np.fft.ifft2(fre_recon))
    return img_recon


def fourier_transform(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 10
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        window = orig_image[noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y]
        window = fourier_broken(window, block_noise_size_x, block_noise_size_y)
        image_temp[noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = window
    bright_transform_x = image_temp

    return bright_transform_x


class Fusionset(Data.Dataset):
    def __init__(self, args, root = root, transform=None, gray=True, partition='train', ssl_transformations=None):
        self.files = glob(os.path.join(root, '*.*'))
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transform
        self.ssl_transformations = ssl_transformations
        self.args = args

        if args.miniset == True:
            self.files = random.sample(self.files, int(args.minirate * len(self.files)))
        self.num_examples = len(self.files)

        if self.ssl_transformations == True:
            print('used ssl_transformations')
        else:
            print('not used ssl_transformations')

        if partition == 'train':
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.gray:
            img = img.convert('L')
        img = np.array(img)

        if self.ssl_transformations == True:
            img_bright_orig = img.copy()
            img_bright_trans = bright_transform(img_bright_orig)
            img_bright_trans = self._tensor(img_bright_trans)

            img_fourier_orig = img.copy()
            img_fourier_trans = fourier_transform(img_fourier_orig)
            img_fourier_trans = self._tensor(img_fourier_trans)

            img_shuffling_orig = img.copy()
            img_shuffling_trans = global_patch_shuffling(img_shuffling_orig)
            img_shuffling_trans = self._tensor(img_shuffling_trans)
            img = self._tensor(img)
        else:
            img = self._tensor(img)
            img_bright_trans = img
            img_fourier_trans = img
            img_shuffling_trans = img

        return img, img_bright_trans, img_fourier_trans, img_shuffling_trans


class dataset_weight(Data.Dataset):
    def __init__(self, img_dir = "/home/xiangao/PycharmProjects/spacefuse/full project/spacedataset", crop_size=256,transform =None):
        super(dataset_weight, self).__init__()
        self.img_list = glob(os.path.join(img_dir,"*"))

        if transform is not None:
            self.transform = transform
        else:
            transform_list = [transforms.ToTensor()]
            self.transform = transforms.Compose(transform_list)


    def __getitem__(self, item):
        # random_num = random.randint(0, len(self.imglist[item])-1)
        a_img_path = glob(os.path.join(self.img_list[item],"A","*"))
        b_img_path = glob(os.path.join(self.img_list[item],"B","*"))
        GT_img_path = glob(os.path.join(self.img_list[item], "GT", "*"))
        mask_path = glob(os.path.join(self.img_list[item],"Semantic","*"))
        img = a_img_path[random.randint(0, len(a_img_path)-1)]
        related_img = b_img_path[random.randint(0, len(b_img_path)-1)]
        gt = GT_img_path[0]
        mask = mask_path[0]
        image = Image.open(img).convert('L')
        image = image.resize((256, 256))
        related_img = Image.open(related_img).convert('L')
        related_img = related_img.resize((256,256))
        # res = np.abs(np.array(image).astype(float)-np.array(related_img).astype(float))
        s_1,s_2 = shadow_detection(image,related_img)
        mask = Image.open(mask)
        mask = mask.resize((256,256))
        gt = Image.open(gt).convert('L')
        gt = gt.resize((256, 256))
        img = self.transform(image)
        related_img = self.transform(related_img)
        s_1 = torch.from_numpy(s_1) # shadow area of image1
        s_1 = s_1.unsqueeze(0)
        s_2 = torch.from_numpy(s_2)
        s_2 = s_2.unsqueeze(0)
        gt = self.transform(gt)
        mask = torch.from_numpy(np.array(mask, dtype='uint8'))


        return img, related_img, gt ,s_1,s_2,mask

    def __len__(self):
        return len(self.img_list)

    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

class val_weight(Data.Dataset):
    def __init__(self, img_dir = "./val", crop_size=256,transform =None):
        super(val_weight, self).__init__()
        self.a_img = glob(os.path.join(img_dir,"A","*"))
        self.b_img = glob(os.path.join(img_dir,"B","*"))
        self.gt = glob(os.path.join(img_dir,"GT","*"))
        self.mask = glob(os.path.join(img_dir,"mask_voc","*"))
        if transform is not None:
            self.transform = transform
        else:
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5))]
            self.transform = transforms.Compose(transform_list)


    def __getitem__(self, item):
        # random_num = random.randint(0, len(self.imglist[item])-1)

        img = self.a_img[item]
        related_img = self.b_img[item]
        gt = self.gt[item]
        mask = self.mask[item]
        mask = Image.open(mask)
        mask = mask.resize((256,256))
        image = Image.open(img).convert('L')
        image = image.resize((256, 256))
        related_img = Image.open(related_img).convert('L')
        related_img = related_img.resize((256,256))
        #res = np.abs(np.array(image).astype(float)-np.array(related_img).astype(float))
        s_1, s_2 = shadow_detection(image, related_img)
        gt = Image.open(gt).convert('L')
        gt = gt.resize((256, 256))
        img = self.transform(image)
        related_img = self.transform(related_img)
        s_1 = torch.from_numpy(s_1)  # shadow area of image1
        s_1 = s_1.unsqueeze(0)
        s_2 = torch.from_numpy(s_2)
        s_2 = s_2.unsqueeze(0)
        gt = self.transform(gt)
        mask = torch.from_numpy(np.array(mask, dtype='uint8'))


        return img, related_img, gt,s_1,s_2,mask

    def __len__(self):
        return len(self.a_img)