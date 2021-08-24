from dataset import ICLEVRLoader
from network.sagan import Generator
from train import sample_z
from evaluator import evaluation_model

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

import argparse


def parse_option():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--load-path',type=str,default='checkpoints/best/sagan_best.pth',help='checkpoints path')
    parser.add_argument('--im-size',type=int,default=64,help="image size")
    parser.add_argument('--z-size',type=int,default=128,help="latent size")
    parser.add_argument('--g-conv-dim',type=int,default=300,help="generator convolution size")
    parser.add_argument('--num-cond',type=int,default=24,help='number of conditions')
    parser.add_argument('--c_size',type=int,default=100)
    parser.add_argument('--device',type=str,default="cuda:1",help='cuda or cpu device')
    parser.add_argument('--filename',type=str,default='test.json')
    args=parser.parse_args()
    return args 


def eval(G,test_loader,eval_model,z_size,device):
    G.eval()
    avg_acc=0.0
    gen_images=None

    with torch.no_grad():
        for idx,conds in enumerate(tqdm(test_loader)):
            conds=conds.to(device)
            z=sample_z(conds.shape[0],z_size).to(device)
            fake_images,_,_=G(z,conds)
            if gen_images is None:
                gen_images=fake_images
            else:
                gen_images=torch.vstack((gen_images,fake_images))
            acc=eval_model.eval(fake_images,conds)
            avg_acc+=acc*conds.shape[0]
    avg_acc/=len(test_loader.dataset)
    return avg_acc,gen_images


if __name__=='__main__':
    args=parse_option()
    
    test_set=ICLEVRLoader('./data',cond=True,mode='test',filename=args.filename)
    test_loader=DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=3)
    
    G=Generator(args.batch_size,args.im_size,args.z_size,args.g_conv_dim,args.num_cond,args.c_size)
    G.load_state_dict(torch.load(args.load_path))
    G.to(args.device)

    best_acc=0.0

    eval_model=evaluation_model(args.device)
    torch.random.manual_seed(218)
    eval_acc,gen_images=eval(G,test_loader,eval_model,args.z_size,args.device)
    print(f"eval_acc: {eval_acc:.4f}")
    best_acc=eval_acc
    gen_images=0.5*gen_images+0.5 # normalization
    save_image(gen_images,"test_gen_image.jpg",nrow=8)

