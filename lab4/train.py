from utils import OneHotEncoder,tf_sched,klw_sched
from network import EncoderRNN,DecoderRNN
from dataset import TextDataset
from loss import KLLoss

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn as nn
from tqdm import tqdm
import torch

import argparse
import logging


def parse_option():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch-size",type=int,default=1,help="batch size in dataloader")
    parser.add_argument("--input-size",type=int,default=28,help="vocabulary size")
    parser.add_argument("--hidden-size",type=int,default=256,help="hidden layer size")
    parser.add_argument("--c-size",type=int,default=4,help="condition size")
    parser.add_argument("--c-hidden_size",type=int,default=8,help="condition hidden layer size")
    parser.add_argument("--z-size",type=int,default=32,help="latent vector size")
    parser.add_argument("--device",type=str,default="cuda:0",help="device use for training")
    parser.add_argument("--lr",type=float,default=0.05,help="learning rate")
    parser.add_argument("--epochs",type=int,default=200,help="number of epochs to train")
    parser.add_argument("--start-epoch",type=int,default=1,help="starting epoch")
    parser.add_argument("--final-tf-ratio",type=float,default=0.8,help="teacher forcing ratio")
    parser.add_argument("--final-kl-w",type=float,default=0.2,help="kl weight")
    parser.add_argument("--kl-anneal-cyc",type=int,default=2,help="kl annealing cycle")
    parser.add_argument("--anneal-method",type=str,default="monotonic",help="KL annealing method: [monotonic,cyclic]")
    args=parser.parse_args()
    return args


def train(train_loader,test_loader,
          encoder,decoder,
          encoder_optimizer,decoder_optimizer,
          xentropy_criterion,kl_criterion,
          tokenizer,epochs,
          start_epoch,final_tf_ratio,
          final_kl_w,anneal_method,
          kl_anneal_cyc,task_name,
          device):
    writer=SummaryWriter(log_dir=f"runs/{task_name}")
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(f'./logger/{task_name}.log', 'w', 'utf-8')])
    
    encoder=encoder.to(device)
    decoder=decoder.to(device)

    for epoch in range(start_epoch,epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        print('-'*len(f"Epoch {epoch}/{epochs}"))

        tf_ratio=tf_sched(epoch,epochs,final_tf_ratio)
        kl_w=klw_sched(anneal_method,epoch,epochs,final_kl_w,kl_anneal_cyc)

        for idx,(inputs,c) in enumerate(tqdm(train_loader)):
            inputs=inputs[0]
            c=c[0]
            token=tokenizer.tokenize(inputs).to(device,dtype=torch.long)
            c=c.to(device,dtype=torch.long)

            encoder.train()
            decoder.train()

            hidden_state,cell_state=encoder.init_hidden_and_cell()
            hid_m,hid_logv,hid_z,cell_m,cell_logv,cell_z=encoder(token,hidden_state,cell_state,c)
            break
        break




if __name__=='__main__':
    opt=parse_option()
    batch_size=opt.batch_size
    input_size=opt.input_size
    hidden_size=opt.hidden_size
    c_size=opt.c_size
    c_hidden_size=opt.c_hidden_size
    z_size=opt.z_size
    device=opt.device
    lr=opt.lr
    epochs=opt.epochs
    start_epoch=opt.start_epoch
    final_tf_ratio=opt.final_tf_ratio
    final_kl_w=opt.final_kl_w
    kl_anneal_cyc=opt.kl_anneal_cyc
    anneal_method=opt.anneal_method

    task_name=f"in{input_size}_h{hidden_size}_c{c_size}_chid{c_hidden_size}_tf{final_tf_ratio}klw{final_kl_w}"
    if anneal_method=="monotonic":
        task_name+="mono"
    elif anneal_method=="cyclic":
        task_name+=f"cyc{kl_anneal_cyc}"

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

    # initialize loss
    xentropy_criterion=nn.CrossEntropyLoss()
    kl_criterion=KLLoss()
    print("Initialize loss function...")
    
    # initialize optimizer
    encoder_optimizer=SGD(encoder.parameters(),lr)
    decoder_optimizer=SGD(decoder.parameters(),lr)
    print("Initialize optimizers...")

    # initialize tokenizer
    tokenizer=OneHotEncoder()
    print("Initialize tokenizer...")

    train(train_loader,test_loader,
          encoder,decoder,
          encoder_optimizer,decoder_optimizer,
          xentropy_criterion,kl_criterion,
          tokenizer,epochs,
          start_epoch,final_tf_ratio,
          final_kl_w,anneal_method,
          kl_anneal_cyc,task_name,
          device)


