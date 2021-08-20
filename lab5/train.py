from evaluator import evaluation_model
from network import sagan,cdcgan
from dataset import ICLEVRLoader

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch

from pathlib import Path
import argparse
import random
import os


def parse_option():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=128,help="batch size")
    parser.add_argument('--im-size',type=int,default=64,help="image size")
    parser.add_argument('--z-size',type=int,default=128,help="latent size")
    parser.add_argument('--g-conv-dim',type=int,default=300,help="generator convolution size")
    parser.add_argument('--d-conv-dim',type=int,default=100,help="discriminator convolution size")
    parser.add_argument('--device',type=str,default="cuda:1",help='cuda or cpu device')
    parser.add_argument('--g-lr',type=float,default=0.0002,help='initial generator learing rate')
    parser.add_argument('--d-lr',type=float,default=0.0002,help='initial discriminator learning rate')
    parser.add_argument('--beta1',type=float,default=0.5,help='Adam beta 1')
    parser.add_argument('--beta2',type=float,default=0.999,help='Adam beta 2')
    parser.add_argument('--epochs',type=int,default=300,help="total epochs")
    parser.add_argument('--eval-iter',type=int,default=50)
    parser.add_argument('--num-cond',type=int,default=24,help='number of conditions')
    parser.add_argument('--adv-loss',type=str,default='wgan-gp',help='adversarial loss method: [bce,hinge,wgan-gp]')
    parser.add_argument('--c_size',type=int,default=100)
    parser.add_argument('--lambda-gp',type=float,default=10)
    parser.add_argument('--net',type=str,default='cdcgan',help='model')
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


def grad_penalty(real_images,fake_images,labels,device,D,net):
    alpha = torch.rand(real_images.size(0), 1, 1, 1).to(device).expand_as(real_images)
    interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
    if net=='sagan':
        out,_,_ = D(interpolated,labels)
    else:
        out=D(interpolated,labels)

    grad = torch.autograd.grad(outputs=out,
                                inputs=interpolated,
                                grad_outputs=torch.ones(out.size()).to(device),
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    grad = grad.view(grad.size(0), -1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
    return d_loss_gp


def train(G,D,g_optimizer,d_optimizer,criterion,adv_loss,train_loader,test_loader,epochs,z_size,eval_iter,device,lambda_gp,task_name,net):
    eval_model=evaluation_model(device)
    best_acc=0.0
    iters=0

    writer=SummaryWriter(log_dir=f'runs/{task_name}')
    model_save_dir=f"./checkpoints/{task_name}"
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    gen_img_dir=f"./gen_img/{task_name}"
    if not os.path.isdir(gen_img_dir):
        os.makedirs(gen_img_dir)
    
    for epoch in range(1,epochs+1):
        G.train()
        D.train()
        
        train_g_loss=0.0
        train_d_loss=0.0

        pbar=tqdm(train_loader)
        for idx,(inputs,conds) in enumerate(pbar):
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

            # ========== Train Discriminator =========== #
            d_optimizer.zero_grad()
            if net=='sagan':
                d_out_real,dr1,dr2 = D(inputs, conds)
            else:
                d_out_real = D(inputs, conds)
            d_x = d_out_real.mean().item()

            if adv_loss=='bce':
                d_loss_real = criterion(d_out_real, real_label_d)
            elif adv_loss=='wgan-gp':
                d_loss_real=-torch.mean(d_out_real)
            elif adv_loss=='hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            
            # apply Gumbel softmax
            z = sample_z(bs, z_size).to(device)
            if net=='sagan':
                fake_images,gf1,gf2 = G(z, conds)
                d_out_fake,df1,df2 = D(fake_images, conds)
            else:
                fake_images = G(z, conds)
                d_out_fake = D(fake_images, conds)
            d_g_z1 = d_out_fake.mean().item()

            if adv_loss=='bce':
                d_loss_fake = criterion(d_out_fake, fake_label_d)
            elif adv_loss=='wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif adv_loss=='hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            
            # backward + optimize
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            if adv_loss == 'wgan-gp':
                # Compute gradient penalty
                d_loss_gp=grad_penalty(inputs,fake_images,conds,device,D,net)

                # Backward + Optimize
                d_loss_gp = lambda_gp * d_loss_gp

                d_optimizer.zero_grad()
                d_loss_gp.backward()
                d_optimizer.step()
            
            # ========== Train generator and gumbel ========== #
            # create random noise
            g_optimizer.zero_grad()
            z = sample_z(bs, z_size).to(device)
            if net=='sagan':
                fake_images,_,_ = G(z, conds)
            else:
                fake_images=G(z,conds)

            # compute loss with fake images
            if net=='sagan':
                g_out_fake,_,_= D(fake_images, conds)
            else:
                g_out_fake=D(fake_images,conds)
            d_g_z2 = g_out_fake.mean().item()

            if adv_loss=='bce':
                g_loss = criterion(g_out_fake, real_label_g)
            elif adv_loss=='wgan-gp':
                g_loss=-torch.mean(g_out_fake)
            elif adv_loss=='hinge':
                g_loss=-torch.mean(g_out_fake)

            g_loss.backward()
            g_optimizer.step()

            train_g_loss+=g_loss.item()
            train_d_loss+=d_loss.item()
            if adv_loss=='wgan-gp':
                train_d_loss+=d_loss_gp.item()

            iters+=1
            pbar.set_description('[{}/{}][{}/{}][LossG={:.4f}][LossD={:.4f}][D(x)={:.4f}][D(G(z))={:.4f}/{:.4f}]'
                .format(epoch, epochs, idx+1, len(train_loader), g_loss.item(), d_loss.item(), d_x, d_g_z1, d_g_z2))
            scalar_dict={
                'g_loss':g_loss.item(),
                'd_loss':d_loss.item(),
                'd_x':d_x,
                'd_g_z1':d_g_z1,
                'd_g_z2':d_g_z2
            }
            for scalar in scalar_dict:
                writer.add_scalar(f"{scalar}/iters",scalar_dict[scalar],iters)
            
            # evaluate
            if iters%eval_iter==0:
                G.eval()
                eval_acc,gen_images=eval(G,test_loader,eval_model,z_size,device,net)
                print(f"eval_acc: {eval_acc:.4f}")
                writer.add_scalar('eval_acc/iters',eval_acc,iters)
                if eval_acc>best_acc:
                    best_acc=eval_acc
                    torch.save(G.state_dict(),Path(model_save_dir).joinpath(f"ep{epoch}_it{iters}_{eval_acc:.4f}.pth"))
                
                gen_images=0.5*gen_images+0.5 # normalization
                save_image(gen_images,Path(gen_img_dir).joinpath(f"ep{epoch}_it{iters}_{eval_acc:.4f}.jpg"),nrow=8)
                
                G.train()
        
        train_d_loss/=len(train_loader)
        train_g_loss/=len(train_loader)

        G.eval()
        eval_acc,gen_images=eval(G,test_loader,eval_model,z_size,device,net)
        print(f"[{epoch}/{epochs}][AvgG: {train_g_loss:.4f}][AvgD: {train_d_loss:.4f}][Acc: {eval_acc:.4f}]")
        scalar_dict={
            'train_loss_g':train_g_loss,
            'train_loss_d':train_d_loss,
            'eval_acc':eval_acc
        }
        for scalar in scalar_dict:
            writer.add_scalar(f"{scalar}/epochs",scalar_dict[scalar],epoch)
        if eval_acc>best_acc:
            best_acc=eval_acc
            torch.save(G.state_dict(),Path(model_save_dir).joinpath(f"ep{epoch}_last_{eval_acc:.4f}.pth"))
        
        gen_images=0.5*gen_images+0.5 # normalization
        save_image(gen_images,Path(gen_img_dir).joinpath(f"ep{epoch}_last_{eval_acc:.4f}.jpg"),nrow=8)

        
def eval(G,test_loader,eval_model,z_size,device,net):
    G.eval()
    avg_acc=0.0
    gen_images=None

    with torch.no_grad():
        for idx,conds in enumerate(test_loader):
            conds=conds.to(device)
            z=sample_z(conds.shape[0],z_size).to(device)

            if net=='sagan':
                fake_images,_,_=G(z,conds)
            else:
                fake_images=G(z,conds)

            if gen_images is None:
                gen_images=fake_images
            else:
                gen_images=torch.vstack((gen_images,fake_images))
            acc=eval_model.eval(fake_images,conds)
            avg_acc+=acc*conds.shape[0]
    avg_acc/=len(test_loader.dataset)
    return avg_acc,gen_images


if __name__=="__main__":
    args=parse_option()
    task_name=f"{args.net}_glr{args.g_lr}_dlr{args.d_lr}_advloss-{args.adv_loss}_ep{args.epochs}_b1-{args.beta1}_b2-{args.beta2}_gconv{args.g_conv_dim}_dconv{args.d_conv_dim}"

    # dataset
    data_transform=transforms.Compose([
        transforms.Resize((args.im_size,args.im_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train_set=ICLEVRLoader('./data',trans=data_transform,cond=True,mode='train')
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=3)
    test_set=ICLEVRLoader('./data',cond=True,mode='test')
    test_loader=DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=3)

    # model
    if args.net=='sagan':
        G=sagan.Generator(args.batch_size,args.im_size,args.z_size,args.g_conv_dim,args.num_cond,args.c_size)
        D=sagan.Discriminator(args.batch_size,args.im_size,args.d_conv_dim,args.num_cond)
    elif args.net=='cdcgan':
        G=cdcgan.Generator(args.z_size,args.c_size,args.num_cond,args.g_conv_dim)
        D=cdcgan.Discriminator(args.im_size,args.d_conv_dim,args.num_cond)
        G.apply(cdcgan.weights_init)
        D.apply(cdcgan.weights_init)
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
          device=args.device,
          lambda_gp=args.lambda_gp,
          task_name=task_name,
          net=args.net)


