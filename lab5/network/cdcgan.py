import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, n_z,n_c,num_conditions,n_ch_g):
        super(Generator, self).__init__()
        self.n_z = n_z
        self.n_c = n_c
        n_ch = [n_ch_g*8, n_ch_g*4, n_ch_g*2, n_ch_g]

        self.embed_c= nn.Sequential(
            nn.Linear(num_conditions, n_c),
            nn.ReLU(inplace=True))

        model = [
            nn.ConvTranspose2d(
                n_z+n_c, n_ch[0], kernel_size=4, stride=2, padding=0,
                bias=False),
            nn.BatchNorm2d(n_ch[0]),
            nn.ReLU(inplace=True)
        ]
        for i in range(1, len(n_ch)):
            model += [
                nn.ConvTranspose2d(
                    n_ch[i-1], n_ch[i], kernel_size=4, stride=2, padding=1,
                    bias=False),
                nn.BatchNorm2d(n_ch[i]),
                # nn.ReLU(inplace=True)
            ]
        model += [
            nn.ConvTranspose2d(
                n_ch[-1], 3, kernel_size=4, stride=2, padding=1,
                bias=False),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, z, c):
        z = z.reshape(-1, self.n_z, 1, 1)
        c_embd = self.embed_c(c).reshape(-1, self.n_c, 1, 1)
        x = torch.cat((z, c_embd), dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, img_sz,n_ch_d,num_conditions):
        super(Discriminator, self).__init__()
        self.img_sz = img_sz
        n_ch = [n_ch_d, n_ch_d*2, n_ch_d*4, n_ch_d*8]

        self.embed_c= nn.Sequential(
            nn.Linear(num_conditions, img_sz*img_sz),
            nn.ReLU(inplace=True))

        model = [
            nn.Conv2d(
                4, n_ch[0], kernel_size=4, stride=2, padding=1,
                bias=False),
            nn.BatchNorm2d(n_ch[0]),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        act=[nn.ReLU(),nn.LeakyReLU(0.2),nn.Softplus(),nn.Tanh()]
        for i in range(1, len(n_ch)):
            model += [
                nn.Conv2d(
                    n_ch[i-1], n_ch[i], kernel_size=4, stride=2, padding=1,
                    bias=False),
                nn.BatchNorm2d(n_ch[i]),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        model += [
            nn.Conv2d(
                n_ch[-1], 1, kernel_size=4, stride=1, padding=0,
                bias=False),
            # nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, image, c):
        c_embd = self.embed_c(c).reshape(-1, 1, self.img_sz, self.img_sz)
        x = torch.cat((image, c_embd), dim=1)
        return self.model(x).reshape(-1)