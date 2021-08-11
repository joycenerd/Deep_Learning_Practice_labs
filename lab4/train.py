from utils import *
from network import EncoderRNN,DecoderRNN
from dataset import TextDataset
from loss import KLLoss

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

import argparse
import logging
import copy
import os


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
    parser.add_argument("--max_len",type=int,default=30,help="maximum word length")
    args=parser.parse_args()
    return args


def decode(tokenizer,device,token,hid_z,cell_z,c,tf):
    in_token=torch.from_numpy(np.asarray(tokenizer.sos))
    in_token=in_token.to(device,dtype=torch.long)
    out_distribution=[]
    if token==None:
        max_len=30
    else:
        max_len=token.shape[0]-1
    for i in range(max_len):
        output,hidden_state,cell_state=decoder(in_token,hid_z,cell_z,c)
        out_distribution.append(output)
        out_token=torch.max(torch.softmax(output,dim=1),1)[1]
        if out_token.item()==tokenizer.eos:
            break
        if tf:
            in_token=token[i+1]
        else:
            in_token=out_token
    out_distribution=torch.cat(out_distribution,dim=0).to(device)
    return out_distribution


def gen_word(z_size,device,tokenizer):
    words_list=[]
    for i in range(100):
        hid_z,cell_z=gen_gauss_noise(z_size)
        hid_z=hid_z.to(device,dtype=torch.float)
        cell_z=cell_z.to(device,dtype=torch.float)
        words=[]
        for c in range(4):
            c=torch.from_numpy(np.asarray(c))
            c=c.to(device,dtype=torch.long)
            outputs=decode(tokenizer,device,None,hid_z,cell_z,c,tf=False)
            out_token=torch.max(torch.softmax(outputs,dim=1),1)[1]
            out_word=tokenizer.inv_tokenize(out_token)
            words.append(out_word)
        words_list.append(words)
    return words_list


def save_model(task_name,encoder_params,decoder_params,filename):
    if not os.path.isdir(f"checkpoints/{task_name}"):
        os.mkdir(f"checkpoints/{task_name}")
    
    save_obj={
        'encoder_state_dict':encoder_params,
        'decoder_state_dict':decoder_params
    }
    torch.save(save_obj,f"checkpoints/{task_name}/{filename}.pt")


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
    
    encoder=encoder.to(device)
    decoder=decoder.to(device)

    iters=0
    best_bleu_score=0.0
    best_gauss_score=0.0
    train_loss=0.0
    train_reconstruct_loss=0.0
    train_reg_loss=0.0

    for epoch in range(start_epoch,epochs+1):
        log_dir=f"./logger/{task_name}"
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        logger = logging.getLogger(__name__)  
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(f"{log_dir}/ep{epoch}.log")
        formatter    = logging.Formatter('%(asctime)s :%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        print(f"Epoch {epoch}/{epochs}")
        print('-'*len(f"Epoch {epoch}/{epochs}"))
        logger.info(f"Epoch {epoch}/{epochs}")

        tf_ratio=tf_sched(epoch,epochs,final_tf_ratio)
        kl_w=klw_sched(anneal_method,epoch,epochs,final_kl_w,kl_anneal_cyc)

        for idx,(inputs,c) in enumerate(tqdm(train_loader)):
            inputs=inputs[0]
            c=c[0]
            token=tokenizer.tokenize(inputs).to(device,dtype=torch.long)
            c=c.to(device,dtype=torch.long)

            encoder.train()
            decoder.train()

            iters+=1

            # encode
            hidden_state,cell_state=encoder.init_hidden_and_cell()
            hid_m,hid_logv,hid_z,cell_m,cell_logv,cell_z=encoder(token,hidden_state,cell_state,c)

            # whether to use teacher forcing
            rand_val=np.random.rand()
            tf=True if rand_val<tf_ratio else False

            # decode
            out_distribution=decode(tokenizer,device,token,hid_z,cell_z,c,tf)

            # loss
            out_len=out_distribution.shape[0]
            reconstruct_loss=xentropy_criterion(out_distribution,token[1:1+out_len])
            reg_loss=kl_criterion(hid_m,hid_logv)+kl_criterion(cell_m,cell_logv)
            total_loss=reconstruct_loss+kl_w*reg_loss
            
            # update
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            total_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            train_loss+=total_loss.item()*train_loader.batch_size
            train_reconstruct_loss+=reconstruct_loss.item()*train_loader.batch_size
            train_reg_loss+=reg_loss.item()*train_loader.batch_size

            # evaluation
            if iters%1000==0:
                is_eval=True
                test_len=len(test_loader.dataset)
                eval_loss,eval_reconstruct_loss,eval_reg_loss,bleu_score,tense_conversion_res,words_list,gauss_score=eval(encoder,decoder,tokenizer,test_loader,device,xentropy_criterion,kl_criterion,kl_w,test_len,z_size)
                print(f"Iteration {iters}:")
                logger.info(f"Iteration {iters}:")
                if bleu_score>best_bleu_score:
                    encoder_params=copy.deepcopy(encoder.state_dict())
                    decoder_params=copy.deepcopy(decoder.state_dict())
                    save_model(task_name,encoder_params,decoder_params,f"epoch_{epoch}_bleu_{bleu_score:.4f}")
                    best_bleu_score=bleu_score
                if gauss_score>best_gauss_score:
                    encoder_params=copy.deepcopy(encoder.state_dict())
                    decoder_params=copy.deepcopy(decoder.state_dict())
                    save_model(task_name,encoder_params,decoder_params,f"epoch_{epoch}_gauss_{gauss_score:.4f}")
                    best_gauss_score=gauss_score

                train_loss/=1000*train_loader.batch_size
                train_reconstruct_loss/=1000*train_loader.batch_size
                train_reg_loss/=1000*train_loader.batch_size          
                
                test_len=len(test_loader.dataset)
                eval_loss,eval_reconstruct_loss,eval_reg_loss,bleu_score,tense_conversion_res,words_list,gauss_score=eval(encoder,decoder,tokenizer,test_loader,device,xentropy_criterion,kl_criterion,kl_w,test_len,z_size)
                
                print(f"train_loss: {train_loss:.4f} train_reconstruct_loss: {train_reconstruct_loss:.4f} train_reg_loss: {train_reg_loss:.4f}")
                print(f"eval_loss: {eval_loss:.4f} eval_reconstruct_loss: {eval_reconstruct_loss:.4f} eval_reg_loss: {eval_reg_loss:.4f}")
                logger.info(f"train_loss: {train_loss:.4f} train_reconstruct_loss: {train_reconstruct_loss:.4f} train_reg_loss: {train_reg_loss:.4f}")
                logger.info(f"eval_loss: {eval_loss:.4f} eval_reconstruct_loss: {eval_reconstruct_loss:.4f} eval_reg_loss: {eval_reg_loss:.4f}")
                print_tense_conversion(tense_conversion_res,bleu_score,logger)
                print_gauss_gen(words_list,gauss_score,logger)

                scalars={
                    'train_loss':train_loss,
                    'train_xentropy_loss':train_reconstruct_loss,
                    'train_kl_loss':train_reg_loss,
                    'tf_ratio':tf_ratio,
                    'kl_w':kl_w,
                    'bleu_score':bleu_score,
                    'gauss_score':gauss_score
                }
                for scalar in scalars:
                    writer.add_scalar(scalar,scalars[scalar],iters//1000)

                train_loss=0.0
                train_reconstruct_loss=0.0
                train_reg_loss=0.0

    print(f"best_bleu_score: {best_bleu_score:.4f} best_gauss_score: {best_gauss_score}")
    logger.info(f"best_bleu_score: {best_bleu_score:.4f} best_gauss_score: {best_gauss_score}")


def eval(encoder,decoder,tokenizer,test_loader,device,xentropy_criterion,kl_criterion,kl_w,test_len,z_size):
    encoder.eval()
    decoder.eval()

    eval_reconstruct_loss=0.0
    eval_reg_loss=0.0
    eval_loss=0.0
    bleu_score=0.0
    tense_conversion_res=[]

    with torch.no_grad():
        for idx,(inputs,targets,c1,c2) in enumerate(tqdm(test_loader)):
            # load data
            inputs=inputs[0]
            targets=targets[0]
            c1=c1[0]
            c2=c2[0]
            token=tokenizer.tokenize(inputs).to(device,dtype=torch.long)
            target_token=tokenizer.tokenize(targets).to(device,dtype=torch.long)
            c1=c1.to(device,dtype=torch.long)
            c2=c2.to(device,dtype=torch.long)

            # encode
            hidden_state,cell_state=encoder.init_hidden_and_cell()
            hid_m,hid_logv,hid_z,cell_m,cell_logv,cell_z=encoder(token,hidden_state,cell_state,c1)

            # decode
            out_distribution=decode(tokenizer,device,target_token,hid_z,cell_z,c2,tf=False)
            out_token=torch.max(torch.softmax(out_distribution,dim=1),1)[1]
            out_word=tokenizer.inv_tokenize(out_token)

            # loss
            out_len=out_distribution.shape[0]
            reconstruct_loss=xentropy_criterion(out_distribution,target_token[1:1+out_len])
            reg_loss=kl_criterion(hid_m,hid_logv)+kl_criterion(cell_m,cell_logv)
            total_loss=reconstruct_loss+kl_w*reg_loss

            eval_loss+=total_loss.item()
            eval_reconstruct_loss+=reconstruct_loss.item()
            eval_reg_loss+=reg_loss.item()

            # bleu
            bleu_score+=compute_bleu(out_word,targets)
            tense_conversion_res.append([inputs,targets,out_word])
        
        eval_loss/=test_len
        eval_reconstruct_loss/=test_len
        eval_reg_loss/=test_len
        bleu_score/=test_len
        
        # generate word
        words_list=gen_word(z_size,device,tokenizer)
        gauss_score=Gaussian_score(words_list,"./data/train.txt")

        return eval_loss,eval_reconstruct_loss,eval_reg_loss,bleu_score,tense_conversion_res,words_list,gauss_score


if __name__=='__main__':
    opt=parse_option()
    print(opt)
    input("Press ENTER if no problem...")
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
    max_len=opt.max_len

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


