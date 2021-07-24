from dataloader import read_bci_data
from dataset import EEGDataset
from model import EEGNet

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch

import argparse
import copy
import os



parser=argparse.ArgumentParser()
parser.add_argument("--epochs",type=int,default=300,help="number of epochs to train")
parser.add_argument("--act",type=str,default="elu",help="which activation function to use in the network: [elu,relu,leaky_relu]")
parser.add_argument("--device",type=str,default="cuda",help="which device to use")
parser.add_argument("--lr",type=float,default=1e-2,help="inital learning rate for training")
args=parser.parse_args()

save_name="EEGNet_elu_1e-2"
writer=SummaryWriter(f"runs/{save_name}")


def train(X_train,y_train,X_test,y_test):
    epochs=args.epochs
    act=args.act
    device=args.device
    lr=args.lr

    act_dict={"elu":nn.ELU(),"relu":nn.ReLU(),"leaky_relu":nn.LeakyReLU()}

    # construct data
    train_set=EEGDataset(X_train,y_train)
    train_loader=DataLoader(train_set,batch_size=256,shuffle=True,num_workers=4)
    test_set=EEGDataset(X_test,y_test)
    test_loader=DataLoader(test_set,batch_size=256,shuffle=True,num_workers=4)

    train_size=len(train_set)
    test_size=len(test_set)

    # initialize model
    activation=act_dict[act]
    net=EEGNet(activation)
    net.to(device)

    # initialize loss function
    loss_func=nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer=torch.optim.Adam(net.parameters(),lr)

    best_train_acc=0.0
    best_acc=0.0
    best_model_params=copy.deepcopy(net.state_dict())

    for epoch in range(1,epochs+1):
        print(f"Epoch: {epoch}/{epochs}")
        print("-"*len(f"Epoch: {epoch}/{epochs}"))

        train_loss=0.0
        train_acc=0.0
        
        # training
        net.train()
        for idx,(inputs,targets) in enumerate(tqdm(train_loader)):
            inputs=inputs.to(device,dtype=torch.float)
            targets=targets.to(device,dtype=torch.long)

            # forward pass
            outputs=net(inputs)
            loss=loss_func(outputs,targets)
            train_loss+=loss.item()
            _,predicted=torch.max(outputs.data,1)
            train_acc+=(predicted==targets).sum().item()

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss=train_loss/train_size
        train_acc=train_acc/train_size  
        print(f"train_loss: {train_loss:.4f}\ttrain_acc: {train_acc:.4f}")
        
        # evaluation
        eval_loss,eval_acc=eval(net,test_loader,test_size,loss_func)
        
        # log the result
        writer.add_scalar("train_loss",train_loss,epoch)
        writer.add_scalar("train_acc",train_acc,epoch)
        writer.add_scalar("eval_loss",eval_loss,epoch)
        writer.add_scalar("eval_acc",eval_acc,epoch)
        writer.flush()

        # save model parameters if accuracy is higher
        if eval_acc>best_acc:
            best_model_params=copy.deepcopy(net.state_dict())
            best_acc=eval_acc
            best_train_acc=train_acc
    
    # save the best model
    print(f"best_acc: {best_acc:.4f}\t best_train_acc: {best_train_acc:.4f}")
    torch.save(best_model_params,f"./checkpoints/{save_name}_{best_acc:.4f}.pt")

    writer.close()


def eval(net,test_loader,test_size,loss_func):
    device=args.device

    net.eval()

    eval_loss=0.0
    eval_acc=0.0

    for idx,(inputs,targets) in enumerate(tqdm(test_loader)):
        inputs=inputs.to(device,dtype=torch.float)
        targets=targets.to(device,dtype=torch.long)

        outputs=net(inputs)
        eval_loss+=loss_func(outputs,targets).item()
        _,predicted=torch.max(outputs.data,1)
        eval_acc+=(predicted==targets).sum().item()
    
    eval_loss/=test_size
    eval_acc/=test_size
    print(f"eval_loss: {eval_loss:.4f}\teval_acc: {eval_acc:.4f}")
    
    return eval_loss,eval_acc


if __name__=="__main__":
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    
    # read data
    X_train,y_train,X_test,y_test=read_bci_data()
    train(X_train,y_train,X_test,y_test)