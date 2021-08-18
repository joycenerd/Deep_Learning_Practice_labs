from network import Generator,Discriminator
from evaluator import evaluation_model
from dataset import ICLEVRLoader

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
import torch

import argparse
import random


def parse_option():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=64,help="batch size")
    parser.add_argument('--im-size',type=int,default=64,help="image size")
    parser.add_argument('--z-size',type=int,default=128,help="latent size")
    parser.add_argument('--g-conv-dim',type=int,default=64,help="generator convolution size")
    parser.add_argument('--d-conv-dim',type=int,default=64,help="discriminator convolution size")
    parser.add_argument('--device',type=str,default="cuda:0",help='cuda or cpu device')
    parser.add_argument('--g-lr',type=float,default=0.0001,help='initial generator learing rate')
    parser.add_argument('--d-lr',type=float,default=0.0004,help='initial discriminator learning rate')
    parser.add_argument('--beta1',type=float,default=0.0,help='Adam beta 1')
    parser.add_argument('--beta2',type=float,default=0.9,help='Adam beta 2')
    parser.add_argument('--epochs',type=int,default=500,help="total epochs")
    parser.add_argument('--eval-iter',type=int,default=50)
    parser.add_argument('--num-cond',type=int,default=24,help='number of conditions')
    parser.add_argument('--adv-loss',type=str,default='wgan-gp',help='adversarial loss method: [bce,hinge,wgan-gp]')
    parser.add_argument('--c_size',type=int,default=100)
    args=parser.parse_args()
    return args


def tensor2var(x,device,grad=False):
    x=x.to(device)
    x=Variable(x,requires_grad=grad)
    return x


def sample_z(bs, n_z, mode='normal'):
    if mode == 'normal':
        return torch.normal(torch.zeros((bs, n_z)), torch.ones((bs, n_z)))
    elif mode == 'uniform':
        return torch.randn(bs, n_z)
    else:
        raise NotImplementedError()


def train(G,D,g_optimizer,d_optimizer,criterion,adv_loss,train_loader,test_loader,epochs,z_size,eval_iter,device):
    eval_model=evaluation_model()
    best_acc=0.0
    
    for epoch in range(epochs):
        G.train()
        D.train()
        
        g_loss=0.0
        d_loss=0.0

        for idx,(inputs,conds) in enumerate(tqdm(train_loader)):
            inputs=inputs.to(device)
            conds=conds.to(device)
            bs=inputs.shape[0]

            real_label_g = torch.ones(bs).to(device)

            real_label_d = torch.normal(torch.ones(bs), torch.ones(bs)*0.01).to(device)
            torch.clamp(real_label_d, max=1)
            fake_label_d = torch.normal(torch.zeros(bs), torch.ones(bs)*0.01).to(device)
            torch.clamp(fake_label_d, min=0)
            if random.random() < 0.1:
                real_label_d, fake_label_d = fake_label_d, real_label_d

            # train discriminator
            d_optimizer.zero_grad()
            d_out_real,dr1,dr2 = D(inputs, conds)
            d_x = d_out_real.mean().item()

            if adv_loss=='bce':
                d_loss_real = criterion(d_out_real, real_label_d)
            elif adv_loss=='wgan-gp':
                d_loss_real=-torch.mean(d_out_real)
            elif adv_loss=='hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            
            # apply Gumbel softmax
            z = sample_z(bs, z_size).to(device)
            fake_images,gf1,gf2 = G(z, conds)
            """outputs = D(fake_images.detach(), conds)
            loss_fake = criterion(outputs, fake_label_d)
            d_g_z1 = outputs.mean().item()
            loss_d = loss_real + loss_fake
            loss_d.backward()
            d_optimizer.step()"""



if __name__=="__main__":
    args=parse_option()
    # dataset
    data_transform=transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train_set=ICLEVRLoader('./data',trans=data_transform,cond=True,mode='train')
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=3)
    test_set=ICLEVRLoader('./data',cond=True,mode='test')
    test_loader=DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=3)

    # model
    G=Generator(args.batch_size,args.im_size,args.z_size,args.g_conv_dim,args.num_cond,args.c_size)
    D=Discriminator(args.batch_size,args.im_size,args.d_conv_dim,args.num_cond)
    G=G.to(args.device)
    D=D.to(args.device)

    # loss
    criterion=nn.BCELoss()

    # optimizer
    g_optimizer=Adam(filter(lambda p:p.requires_grad,G.parameters()),lr=args.g_lr,betas=[args.beta1,args.beta2])
    d_optimizer=Adam(filter(lambda p:p.requires_grad,D.parameters()),lr=args.d_lr,betas=[args.beta1,args.beta2])

    train(G=G,
          D=D,
          g_optimizer=g_optimizer,
          d_optimizer=d_optimizer,
          criterion=criterion,
          adv_loss=args.adv_loss,
          train_loader=train_loader,
          test_loader=test_loader,
          epochs=args.epochs,
          z_size=args.z_size,
          eval_iter=args.eval_iter,
          device=args.device)


