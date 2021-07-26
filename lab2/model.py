import torch.nn as nn
import torch


class EEGNet(nn.Module):
    def __init__(self,activation):
        super(EEGNet,self).__init__()
        self.firstconv=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=(1,51),stride=(1,1),padding=(0,25),bias=False),
            nn.BatchNorm2d(16,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        )
        self.depthwiseConv=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=(2,1),stride=(1,1),groups=16,bias=False),
            nn.BatchNorm2d(32,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1,4),stride=(1,4),padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(32,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8),padding=0),
            nn.Dropout(p=0.25)
        )
        self.classifier=nn.Linear(in_features=736,out_features=2,bias=True)     
    

    def forward(self,x):
        x=self.firstconv(x)
        x=self.depthwiseConv(x)
        x=self.separableConv(x)
        x=x.view(x.shape[0],-1)
        out=self.classifier(x)
        return out


class DeepConvNet(nn.Module):
    def __init__(self,activation):
        super(DeepConvNet,self).__init__()
        self.activation=activation

        self.conv1=nn.Conv2d(1,25,kernel_size=(1,5))
        self.first_conv_block=nn.Sequential(
            nn.Conv2d(25,25,kernel_size=(2,1)),
            nn.BatchNorm2d(25,eps=1e-5,momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )

        conv_layers=[]
        channel=25
        for i in range(4):
            conv_layers.append(self._make_conv_block(channel))
            channel*=2
        self.conv_blocks=nn.Sequential(*conv_layers) 

        self.classifier=nn.Linear(7600,2)  


    def _make_conv_block(self,channel):
        return nn.Sequential(
            nn.Conv2d(channel,channel*2,kernel_size=(1,5)),
            nn.BatchNorm2d(channel*2,eps=1e-5,momentum=0.1),
            self.activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        ) 
    

    def forward(self,x):
        x=self.conv1(x)
        x=self.first_conv_block(x)
        x=self.conv_blocks(x)
        x=x.view(x.shape[0],-1)
        out=self.classifier(x)
        return out


def initialize_weights(m):
    """
    Initialize the model
    :param m: model
    """
    if isinstance(m,nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data,0)
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight.data,1)
        nn.init.constant_(m.bias.data,0)
    elif isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data,0)

