import torch.nn as nn

from functools import partial


class Conv2dAuto(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        """padding version of nn.Conv2d"""
        super().__init__(*args,**kwargs)
        self.padding=(self.kernel_size[0]//2,self.kernel_size[1]//2)


def act_func(act):
    """function that can choose activation function
    Args:
        act: (str) activation function name

    Returns:
        corresponding Pytorch activation function
    """
    return nn.ModuleDict([
        ['relu',nn.ReLU(inplace=True)],
        ['leaky_relu',nn.LeakyReLU(negative_slope=0.01,inplace=True)],
        ['selu',nn.SELU(inplace=True)]
    ])[act] 


class ResidualBlock(nn.Module):
    """
    Takes an input with in_channels, applies some blocks of convolutional layers to reduce it 
    to out_channels and sum it up to the original input.
    """
    def __init__(self,in_channels,out_channels,expansion=1,downsampling=1,act='relu',*args,**kwargs):
        """
        Args:
            in_channels: (int) input channel num
            out_channels: (int) output channel num
            expansion: (int) increase the out_channels if needed, default=1
            downsampling: (int) stride size in nn.Conv2d
        """
        super().__init__(*args,**kwargs)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.expansion=expansion
        self.downsampling=downsampling
        self.act=act_func(act)

        self.blocks=nn.Identity()
        self.shortcut=nn.Sequential(
            nn.Conv2d(self.in_channels,self.expanded_channels,kernel_size=1,stride=self.downsampling,bias=False),
            nn.BatchNorm2d(self.expanded_channels)
        )
    
    def forward(self,x):
        residual=x
        if self.should_apply_shortcut:
            residual=self.shortcut(x)
        x=self.blocks(x)
        x+=residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels!=self.out_channels
       
    @property
    def expanded_channels(self):
        return self.out_channels*self.expansion


def conv_bn(in_channels,out_channels,*args,**kwargs):
    """
    Stack one conv and batchnorm layer

    Args:
        in_channels: (int) input channel num
        out_channels: (int) output channel num
    
    Returns:
        3x3 conv layer + batchnorm
    """
    conv=partial(Conv2dAuto,bias=False)
    return nn.Sequential(
        conv(in_channels,out_channels,*args,**kwargs),
        nn.BatchNorm2d(out_channels)
    )


class BasicBlock(ResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion=1
    
    def __init__(self,in_channels,out_channels,*args,**kwargs):
        """
        Args:
            in_channels: (int) input channel num
            out_channels: (int) output channel num
        """
        super().__init__(in_channels,out_channels,*args,**kwargs)
        self.blocks=nn.Sequential(
            conv_bn(self.in_channels,out_channels,kernel_size=3,bias=False,stride=self.downsampling),
            self.act,
            conv_bn(self.out_channels,self.expanded_channels,kernel_size=3,bias=False)
        )


class BottleNeckBlock(ResidualBlock):
    """
    To increase the network depth while keeping the parameters size as low as possible
    1x1conv/3x3conv/1x1conv
    """
    expansion=4

    def __init__(self,in_channels,out_channels,*args,**kwargs):
        """
        Args:
            in_channels: (int) input channel num
            out_channel: (int) output channel num
        """
        super().__init__(in_channels,out_channels,expansion=4,*args,**kwargs)
        self.blocks=nn.Sequential(
            conv_bn(self.in_channels,self.out_channels,kernel_size=1),
            self.act,
            conv_bn(self.out_channels,self.out_channels,kernel_size=3,stride=self.downsampling),
            self.act,
            conv_bn(self.out_channels,self.expanded_channels,kernel_size=1)
        )

