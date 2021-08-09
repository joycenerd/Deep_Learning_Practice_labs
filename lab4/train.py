from network import EncoderRNN,DecoderRNN
from dataset import TextDataset

from torch.utils.data import DataLoader

import argparse


def parse_option():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch-size",type=int,default=1,help="batch size in dataloader")
    parser.add_argument("--input-size",type=int,default=28,help="vocabulary size")
    parser.add_argument("--hidden-size",type=int,default=256,help="hidden layer size")
    parser.add_argument("--c-size",type=int,default=4,help="condition size")
    parser.add_argument("--c-hidden_size",type=int,default=8,help="condition hidden layer size")
    parser.add_argument("--z-size",type=int,default=32,help="latent vector size")
    parser.add_argument("--device",type=str,default="cuda:0",help="device use for training")
    args=parser.parse_args()
    return args



if __name__=='__main__':
    opt=parse_option()
    batch_size=opt.batch_size
    input_size=opt.input_size
    hidden_size=opt.hidden_size
    c_size=opt.c_size
    c_hidden_size=opt.c_hidden_size
    z_size=opt.z_size
    device=opt.device

    # data preprocessing
    train_set=TextDataset('./data','train')
    test_set=TextDataset('./data','test')
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=3)
    test_loader=DataLoader(test_set,batch_size=batch_size,num_workers=3)
    print(f"train_set size: {len(train_set)}")
    print(f"test_set size: {len(test_set)}")

    # model
    encoder=EncoderRNN(input_size,hidden_size,c_size,c_hidden_size,z_size,device)
    decoder=DecoderRNN(hidden_size,input_size,c_size,c_hidden_size,z_size,device)
    print("Initialize encoder and decoder...")
