import torchvision.models as models
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
        if self.should_apply_shortcut:
            self.shortcut=nn.Sequential(
                nn.Conv2d(self.in_channels,self.expanded_channels,kernel_size=1,stride=self.downsampling,bias=False),
                nn.BatchNorm2d(self.expanded_channels)
            )
        else:
            self.shortcut=None
    
    def forward(self,x):
        residual=x
        if self.should_apply_shortcut:
            residual=self.shortcut(x)
        x=self.blocks(x)
        x+=residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels!=self.expanded_channels
       
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


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed of `n` blocks stacked one after the other
    """
    def __init__(self,in_channels,out_channels,block=BasicBlock,n=1,*args,**kwargs):
        """
        Args:
            in_channels: (int) input channel num
            out_channels: (int) output channel num
            block: (nn.Module) BasicBlock or BottleNeckBlock
            n: (int) how many blocks
        """
        super().__init__()
        downsampling=2 if in_channels!=out_channels else 1
        block_list=[]
        block_list.append(block(in_channels,out_channels,*args,**kwargs,downsampling=downsampling))
        for _ in range(n-1):
            block_list.append(block(out_channels*block.expansion,out_channels,downsampling=1,*args,**kwargs))
        self.blocks=nn.Sequential(*block_list)
    
    def forward(self,x):
        x=self.blocks(x)
        return x


class Encoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features
    """
    def __init__(self,in_channels=3,blocks_sizes=[64,128,256,512],deepths=[2,2,2,2],act='relu',block=BasicBlock,*args,**kwargs):
        """
        Args:
            in_channels: (int) input channel, default=3 (RGB)
            blocks_size: (list(int)) feature map channel num in each block
            deepths: (list(int)) how many layer ResNet layer for each block size
            act: (str) activation function
            block: (nn.Module) BasicBlock or BottleNeckBlock
        """
        super().__init__()
        self.blocks_size=blocks_sizes
        self.act=act_func(act)

        self.gate=nn.Sequential(
            nn.Conv2d(in_channels,self.blocks_size[0],kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(self.blocks_size[0]),
            self.act,
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        block_list=[]
        block_list.append(ResNetLayer(blocks_sizes[0],blocks_sizes[0],n=deepths[0],act=act,block=block,*args,**kwargs))
        for i in range(1,len(self.blocks_size)):
            block_list.append(ResNetLayer(blocks_sizes[i-1]*block.expansion,blocks_sizes[i],n=deepths[i],act=act,block=block,*args,**kwargs))
        self.blocks=nn.Sequential(*block_list)

    def forward(self,x):
        x=self.gate(x)
        x=self.blocks(x)
        return x


class Decoder(nn.Module):
    """
    The tail of ResNet (classifier). It performs a global average pooling and use a fully
    connected layer to map the output to correct the class
    """
    def __init__(self,in_features,n_classes):
        """
        Args:
            in_features: (int) parameters in encoder
            n_classes: (int) total class num
        """
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.decoder=nn.Linear(in_features,n_classes)
    
    def forward(self,x):
        x=self.avg_pool(x)
        x=x.view(x.shape[0],-1)
        x=self.decoder(x)
        return x


class ResNet(nn.Module):
    """
    ResNet model class
    """
    def __init__(self,in_channels,n_classes,*args,**kwargs):
        """
        Args:
            in_channels: (int) input channel num
            n_classes: (int) total class
        """
        super().__init__()
        self.encoder=Encoder(in_channels,*args,**kwargs)
        self.decoder=Decoder(self.encoder.blocks[-1].blocks[-1].expanded_channels,n_classes)

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x


def resnet18(in_channels,n_classes,block=BasicBlock,*args,**kwargs):
    """
    ResNet18 model
    
    Args:
        in_channels: (int) input channel num
        n_classes: (int) total class
        block: (nn.Module) BasicBlock or BottleNeckBlock
    
    Returns:
        ReNet model with resnet18 specific layers
    """
    return ResNet(in_channels,n_classes,block=block,deepths=[2,2,2,2],*args,**kwargs)


def resnet50(in_channels,n_classes,block=BottleNeckBlock,*args,**kwargs):
    """
    ResNet50 model
    
    Args:
        in_channels: (int) input channel num
        n_classes: (int) total class
        block: (nn.Module) BasicBlock or BottleNeckBlock
    
    Returns:
        ReNet model with resnet18 specific layers
    """
    return ResNet(in_channels,n_classes,block=block,deepths=[3,4,6,3],*args,**kwargs)


def pretrained_resnet(_model):
    """
    Get the pretrained ResNet and modified the last fc layer to [num_feat,5] (5 is the output 
    class num)
    
    Args:
        _model: (str) which model [resnet18,resnet50]
    
    Return:
        return pretrained model + modification of last fc layer with initialization

    """
    if _model=="resnet18":
        model=models.resnet18(pretrained=True)
    elif _model=="resnet50":
        model=models.resnet50(pretrained=True)

    num_feat=model.fc.in_features
    model.fc=nn.Linear(num_feat,5)
    nn.init.xavier_uniform_(model.fc.weight)
    model.fc.bias.data.fill_(0.001)

    return model


def initialize_weights(m):
    """
    Initialize the model
    
    Args:
        m: (nn.Module) model
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)




