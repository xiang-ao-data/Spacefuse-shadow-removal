# The main auto-encoder network used in this project with detail explanation



import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import functools

"""
Transmef network 
"""
def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad

    return hook

class convblock(nn.Module):
    " Cnn block: conv2d+relu+conv2d+relu"
    def __init__(self,indim,outdim,middle=0):
        super(convblock, self).__init__()
        if middle ==0:
            middle=outdim
        self.block = nn.Sequential(
            nn.Conv2d(indim,outdim,kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(middle,outdim,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        x = self.block(x)
        return x

class outconv(nn.Module):
    def __init__(self,indim,outdim=1):
        super(outconv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(indim,outdim,kernel_size =1)
        )

    def forward(self,x):
        return self.layer(x)

class Encoder(nn.Module):
    "Cnn+Transform encoder"
    def __init__(self,):
        super(Encoder, self).__init__()
        self.layer1 = convblock(1,16)
        self.trans = ViT(image_size=256, patch_size=16, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1,
                               emb_dropout=0.1)
        self.layer2 = convblock(17,32)
        self.layer3 = convblock(32,48)

    def forward(self,x):
        x_c = self.layer1(x)
        x_t = self.trans(x)
        x = torch.cat((x_c,x_t),dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Decoder(nn.Module):
    "Cnn based decoder with two convblock and 1x1 conv"
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = convblock(48,32)
        self.layer2 = convblock(32,16)
        self.layer3 = outconv(16,1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class TransMEF_model(nn.Module):
    " auto encoder architechual "
    def __init__(self):
        super(TransMEF_model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1) # chunk tensor to 3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # MSA multi-scale attention
            x = ff(x) + x  # MLP multi linear projection
        return x


class ViT(nn.Module):  # Transblock 
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=1, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), # hxw means the nums of patch
            nn.Linear(patch_dim, dim)
        )
        self.dim = dim
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.convd1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, img):
        x = self.to_patch_embedding(img)  # [B,C,256,256]
        b, n, _ = x.shape

        x = self.transformer(x)
        x = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, h=16, c=1)(x)  # [B,1,256,256]

        return x


# the sub_net use to generate fuse weight
class Sub_net(nn.Module):
    def __init__(self):
        super(Sub_net, self).__init__()
        self.sub_net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32,out_channels=48,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(48)
        )

    def forward(self,img1,s_2):
        x = torch.cat((img1,s_2),dim=1)
        res = self.sub_net(x)

        return res




















