"""
predict the test image,
"""
import os.path
from glob import glob
from network import Sub_net, TransMEF_model,Fusion_net
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image
from shadow_detection import shadow_detection
import torch.nn.functional as F

import argparse
parser = argparse.ArgumentParser(description='Full pipeline predict')
parser.add_argument("--deeplab_ckpt", type=str, default='./checkpoint/best_finetune.pth',help='deeplab checkpoint path')
parser.add_argument("--encoder_ckpt", type=str, default='./checkpoint/Transmef_epoch_30_tensor.pth',help='auto-encoder checkpoint path')
parser.add_argument("--subnet_path", type=str, default='./ckpt/epoch_600.pth',help='subnet model path')
parser.add_argument("--test_path", type=str, default='./val',help='test image path')
opts = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Transmef = TransMEF_model().to(device)
# Dense_net = Densenet().to(device)
state_dict = torch.load(opts.encoder_ckpt, map_location='cuda:0')['model']
Transmef.load_state_dict(state_dict)
# Dense_net = frozen(Dense_net)
Transmef.eval()
Subnet = Sub_net().to(device)
Subnet.eval()
_to_tensor = transforms.ToTensor()
Subnet.load_state_dict(torch.load("./ckpt/Subnet_ssim(s+uns).pth",map_location='cuda:0')['model'])
Subnet.eval()

# get test data
root = './val'
num =0

def encode(img1,img2,model):
    model.eval()
    with torch.no_grad():
        f_1 = model.encoder(img1)
        f_2 = model.encoder(img2)
    return f_1, f_2

for i in range(17):
    num = str(i+1)
    A_path = os.path.join(root,"A",str(i+1)+".png")
    img_A = Image.open(A_path).convert('L')
    B_path = os.path.join(root,"B",str(i+1)+".png")
    img_B = Image.open(B_path).convert('L')
    s_1, s_2 = shadow_detection(img_A.resize((256,256)), img_B.resize((256,256)))
    img_A = _to_tensor(img_A.resize((256,256)))
    img_B = _to_tensor(img_B.resize((256,256)))
    img_A = img_A.unsqueeze(0).to(device)
    img_B = img_B.unsqueeze(0).to(device)
    s_1 = torch.from_numpy(s_1).unsqueeze(0).unsqueeze(0).to(device)
    s_2 = torch.from_numpy(s_2).unsqueeze(0).unsqueeze(0).to(device)

    shadow_1 = img_B * s_1
    shadow_2 = img_A * s_2
    feature1, feature2 = encode(img_A, img_B, Transmef)
    shadow_inf_1, shadow_inf_2 = encode(shadow_1, shadow_2, Transmef)
    weight1 = Subnet(img_A, shadow_1)
    weight2 = Subnet(img_B, shadow_2)
    weight = Subnet(img_A,img_B)
    feature_fuse_1 = feature1 * weight1 + shadow_inf_1 * (1 - weight1)
    feature_fuse_2 = feature2 * weight2 + shadow_inf_2 * (1 - weight2)
    feature = feature_fuse_1+feature_fuse_2

    out1 = Transmef.decoder(feature)
    out_img = out1[0].squeeze(0).detach().cpu()
    out_img = transforms.ToPILImage()(out_img)
    out_img.save(os.path.join("./train_recoder/val",  num + ".png"))



