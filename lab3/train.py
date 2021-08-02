from resnet import resnet18,resnet50,pretrained_resnet,initialize_weights,funct
from dataloader import RetinopathyLoader

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import torch.nn as nn
import torch

import argparse
import logging
import copy
import os


parser=argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="resnet18",help="which model: [resnet18, resnet50]")
parser.add_argument("--pretrain",type=bool,default=False,help="whether to use pretrained weights")
parser.add_argument("--act",type=str,default='relu',help="which activation function to use: [relu, leaky_relu,selu]")
parser.add_argument("--device",type=str,default="cuda:1",help="use which device for training")
parser.add_argument("--batch-size",type=int,default=32,help="batch size for training")
parser.add_argument("--lr",type=float,default=1e-3,help="learning rata")
parser.add_argument("--epochs",type=int,default=10,help="num of epochs for training")
parser.add_argument("--model-path",type=str,default="./checkpoints/resnet50/relu_0.001_0.7335.pt")
parser.add_argument("--load",type=bool,default=True,help="if load the weight param from checkpoint before training")
parser.add_argument("--img-size",type=int,default=256,help="Size to resize the image")
args=parser.parse_args()


def train():
    _model=args.model
    pretrain=args.pretrain
    act=args.act
    device=args.device
    batch_size=args.batch_size
    lr=args.lr
    epochs=args.epochs
    load=args.load
    model_path=args.model_path
    img_size=args.img_size

    if pretrain:
        save_name=f"{act}_{lr}"
    else:
        save_name=f"{act}_{lr}"
    writer=SummaryWriter(f"runs/{_model}/{save_name}")
    logging.basicConfig(format='%(asctime)s - %(message)s', 
                        level=logging.INFO,
                        handlers=[logging.FileHandler(f'./record/{save_name}.log','w','utf-8')])

    # data preprocessing
    train_set=RetinopathyLoader("./Data/data","train",img_size)
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4)
    test_set=RetinopathyLoader("./Data/data","test",img_size)
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=4)

    train_size=len(train_set)
    test_size=len(test_set)

    # initialize model
    if load:
        model=load_model(model_path)
    elif pretrain:
        model=pretrained_resnet(_model)
    else:
        if _model=="resnet18":
            model=resnet18(3,5,act=act)
        elif _model=="resnet50":
            model=resnet50(3,5,act=act)
        model.apply(initialize_weights)
    model=model.to(device)

    # initialize loss function
    loss_func=nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer=torch.optim.SGD(model.parameters(),lr,momentum=0.9,weight_decay=5e-4)

    # initialize scheduler
    scheduler=StepLR(optimizer,step_size=2,gamma=0.5)

    best_train_acc=0.0
    best_acc=0.0
    best_model_params=copy.deepcopy(model.state_dict())

    for epoch in range(1,epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        print("-"*len(f"Epoch {epoch}/{epochs}"))
        logging.info(f"Epoch {epoch}/{epochs}")

        train_loss=0.0
        train_acc=0.0

        # training
        model.train()
        for idx,(inputs,targets) in enumerate(tqdm(train_loader)):
            inputs=inputs.to(device,dtype=torch.float)
            targets=targets.to(device,dtype=torch.long)

            # forward pass
            outputs=model(inputs)
            loss=loss_func(outputs,targets)
            train_loss+=loss.item()*inputs.shape[0]
            _,predicted=torch.max(outputs.data,1)
            train_acc+=(predicted==targets).sum().item()

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss/=train_size
        train_acc/=train_size
        print(f"train_loss: {train_loss:.4f}\ttrain_acc: {train_acc:.4f}")
        logging.info(f"train_loss: {train_loss:.4f}\ttrain_acc: {train_acc:.4f}")
        
        # evaluation
        eval_loss,eval_acc=eval(model,test_loader,test_size,loss_func)
        logging.info(f"eval_loss: {eval_loss:.4f}\teval_acc: {eval_acc:.4f}")

        if epoch>=4:
            scheduler.step()

        # log the result
        # log the result
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("eval_loss", eval_loss, epoch)
        writer.add_scalar("eval_acc", eval_acc, epoch)
        writer.flush()

        # save model parameters if accuracy is higher
        if eval_acc>best_acc:
            best_model_params=copy.deepcopy(model.state_dict())
            best_acc=eval_acc
            best_train_acc=train_acc
        
    # save the best model
    print(f"best_acc: {best_acc:.4f}\tbest_train_acc: {best_train_acc:.4f}")
    logging.info(f"best_acc: {best_acc:.4f}\tbest_train_acc: {best_train_acc:.4f}")
    save_model(_model,best_model_params,pretrain,act,save_name,best_acc)

    writer.close()



def eval(model, test_loader, test_size, loss_func):
    """
    model evaluation

    Args:
        model: model for evaluation
        test_loader: testing data loader
        test_size: size of the testing data
        loss_func: loss function we are using

    Returns:
        evaluation loss and accuracy
    """
    device = args.device

    model.eval()

    eval_loss = 0.0
    eval_acc = 0.0
    for idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)

        outputs = model(inputs)
        eval_loss += loss_func(outputs, targets).item()*inputs.shape[0]
        _, predicted = torch.max(outputs.data, 1)
        eval_acc += (predicted == targets).sum().item()

    eval_loss /= test_size
    eval_acc /= test_size
    print(f"eval_loss: {eval_loss:.4f}\teval_acc: {eval_acc:.4f}")

    return eval_loss, eval_acc


def test():
    """
    Test the saving model
    """
    model_path=args.model_path
    device=args.device
    batch_size=args.batch_size
    img_size=args.img_size

    # prepare data
    test_set=RetinopathyLoader("./Data/data","test",img_size)
    test_loader=DataLoader(test_set,batch_size=batch_size,num_workers=4)
    test_size=len(test_set)

    # load the model
    model=load_model(model_path)
    model.to(device)
    model.eval()

    loss_func=nn.CrossEntropyLoss()

    test_loss=0.0
    test_acc=0.0

    for idx,(inputs,targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)

        outputs = model(inputs)
        test_loss += loss_func(outputs, targets).item()*inputs.shape[0]
        _, predicted = torch.max(outputs.data, 1)
        test_acc += (predicted == targets).sum().item()

    test_loss /= test_size
    test_acc /= test_size
    print(f"test_loss: {test_loss:.4}\ttest_acc: {test_acc:.4}")
 

def save_model(_model,model_params,pretrain,act,save_name,best_acc):
    """
    Save the model as checkpoint

    Args:
        _model: (str) model name
        model_params: (state_dict) model parameters
        pretrain: (bool) whether this is a pretrained model
        act: (str) activation function use in the model
        save_name: (str) saving name of this checkpoint file
        best_acc: (float) best evaluation accuracy
    """
    save_obj={
        'model_state_dict':model_params,
        'model_name':_model,
        'pretrained':pretrain,
        'act':act
    }

    torch.save(save_obj,f"./checkpoints/{_model}/{save_name}_{best_acc:.4f}.pt")


def load_model(model_path):
    """
    Load the model from checkpoint file

    Args:
        model_path: (str) checkpoint file path
    
    Returns:
        model with saving params
    """
    checkpoint=torch.load(model_path)
    _model=checkpoint['model_name']
    pretrain=checkpoint['pretrained']
    act=checkpoint['act']
    
    if pretrain:
        model=pretrained_resnet(_model)
    else:
        if _model=="resnet18":
            model=resnet18(3,5,act=act)
        elif _model=="resnet50":
            model=resnet50(3,5,act=act)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model




if __name__=="__main__":
    print(args)
    input("Press ENTER if no problem...")

    if not os.path.isdir("./checkpoints/resnet18"):
        os.makedirs("./checkpoints/resnet18")
    if not os.path.isdir("./checkpoints/resnet50"):
        os.makedirs("./checkpoints/resnet50")

    # train()
    test()