from torch import nn
import torch
# from torch.legacy.nn import Identity

# Residual network.
# WGAN-GP paper defines a residual block with up & downsampling.
# See the official implementation (given in the paper).
# I use architectures described in the official implementation,
# since I find it hard to deduce the blocks given here from the text alone.
class MeanPoolConv(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(MeanPoolConv, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        self.model = nn.Sequential(conv1)
    def forward(self, x):
        out = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4.0
        out = self.model(out)
        return out

class ConvMeanPool(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(ConvMeanPool, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        self.model = nn.Sequential(conv1)
    def forward(self, x):
        out = self.model(x)
        out = (out[:,:,::2,::2] + out[:,:,1::2,::2] + out[:,:,::2,1::2] + out[:,:,1::2,1::2]) / 4.0
        return out

class UpsampleConv(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(UpsampleConv, self).__init__()

        self.model = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        )
    def forward(self, x):
        x = x.repeat((1, 4, 1, 1)) # Weird concat of WGAN-GPs upsampling process.
        out = self.model(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_output, k_size, resample='up', bn=True, spatial_dim=None):
        super(ResidualBlock, self).__init__()

        self.resample = resample

        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size)
            self.conv2 = nn.Conv2d(n_output, n_output, k_size, padding=(k_size-1)//2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size)
            self.out_dim = n_output
        elif resample == 'down':
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = ConvMeanPool(n_input, n_output, k_size)
            self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size)
            self.out_dim = n_output
            self.ln_dims = [n_input, spatial_dim, spatial_dim] # Define the dimensions for layer normalization.
        else:
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv_shortcut = None # Identity
            self.out_dim = n_input
            self.ln_dims = [n_input, spatial_dim, spatial_dim]

        self.model = nn.Sequential(
            nn.BatchNorm2d(n_input) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv1,
            nn.BatchNorm2d(self.out_dim) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv2,
        )

    def forward(self, x):
        if self.conv_shortcut is None:
            return x + self.model(x)
        else:
            return self.conv_shortcut(x) + self.model(x)

class DiscBlock1(nn.Module):
    def __init__(self, n_output):
        super(DiscBlock1, self).__init__()

        self.conv1 = nn.Conv2d(4, n_output, 3, padding=(3-1)//2)
        self.conv2 = ConvMeanPool(n_output, n_output, 1)
        self.conv_shortcut = MeanPoolConv(4, n_output, 1)

        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.conv2
        )

    def forward(self, x):
        return self.conv_shortcut(x) + self.model(x)

class Generator(nn.Module):
    def __init__(self,num_cond,c_size,z_size):
        super(Generator, self).__init__()

        self.z_size=z_size
        self.c_size=c_size

        self.embed_c=nn.Sequential(
            nn.Linear(num_cond,c_size),
            nn.ReLU(inplace=True)
        )

        self.model = nn.Sequential(                     # 128 x 1 x 1
            nn.ConvTranspose2d(128+c_size, 128, 8, 1, 0),      # 128 x 4 x 4
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 8 x 8
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 16 x 16
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=(3-1)//2),     # 3 x 32 x 32
            nn.Tanh()
        )

    def forward(self, z,c):
        z = z.reshape(-1, self.z_size, 1, 1)
        c_embd=self.embed_c(c).reshape(-1, self.c_size, 1, 1)
        x = torch.cat((z, c_embd), dim=1)
        img = self.model(x)
        return img

class Discriminator(nn.Module):
    def __init__(self,num_cond,im_size):
        super(Discriminator, self).__init__()
        n_output = 128
        '''
        This is a parameter but since we experiment with a single size
        of 3 x 32 x 32 images, it is hardcoded here.
        '''
        self.im_size=im_size

        self.embed_c=nn.Sequential(
            nn.Linear(num_cond,self.im_size*self.im_size),
            nn.ReLU(inplace=True)
        )

        self.DiscBlock1 = DiscBlock1(n_output)                      # 128 x 16 x 16

        self.model = nn.Sequential(
            ResidualBlock(n_output, n_output, 3, resample='down', bn=False, spatial_dim=32),  # 128 x 8 x 8
            ResidualBlock(n_output, n_output, 3, resample=None, bn=False, spatial_dim=16),    # 128 x 8 x 8
            ResidualBlock(n_output, n_output, 3, resample=None, bn=False, spatial_dim=16),    # 128 x 8 x 8
            nn.ReLU(inplace=True),
        )
        self.l1 = nn.Sequential(nn.Linear(128, 1))                  # 128 x 1

    def forward(self, x,c):
        c_embd=self.embed_c(c).reshape(-1,1,self.im_size,self.im_size)
        x = torch.cat((x, c_embd), dim=1)
        # x = x.view(-1, 3, 32, 32)
        y = self.DiscBlock1(x)
        y = self.model(y)
        y = y.view(x.size(0), 128, -1)
        y = y.mean(dim=2)
        out = self.l1(y).unsqueeze_(1).unsqueeze_(2) # or *.view(x.size(0), 128, 1, 1, 1)
        return out
