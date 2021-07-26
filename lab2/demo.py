from model import EEGNet,DeepConvNet
from dataloader import read_bci_data
from dataset import EEGDataset

from torch.utils.data import DataLoader
import torch.nn as nn
import torch


def demo(checkpoints,X_test,y_test):
    device="cuda"

    # prepare data
    test_set=EEGDataset(X_test,y_test)
    test_loader=DataLoader(test_set,batch_size=256,shuffle=True,num_workers=4)
    test_size=len(test_set)

    for act in checkpoints:
        model_path=checkpoints[act]
    
        # model
        net=load_model(model_path)
        net.to(device)
        net.eval()

        loss_func=nn.CrossEntropyLoss()

        test_loss=0.0
        test_acc=0.0

        for idx,(inputs,targets) in enumerate(test_loader):
            inputs=inputs.to(device,dtype=torch.float)
            targets=targets.to(device,dtype=torch.long)

            outputs=net(inputs)
            test_loss+=loss_func(outputs,targets).item()
            _,predicted=torch.max(outputs.data,1)
            test_acc+=(predicted==targets).sum().item()

        test_loss/=test_size
        test_acc/=test_size
        print(f"{act} test_loss: {test_loss:.4}\ttest_acc: {test_acc:.4}")
    
    print("===========================================================\n")
        

def load_model(model_path):
    checkpoint=torch.load(model_path)
    model_name=checkpoint['model_name']
    act=checkpoint['act']
    if model_name=="EEGNet":
        net=EEGNet(act)
    elif model_name=="DeepConvNet":
        net=DeepConvNet(act)
    net.load_state_dict(checkpoint['model_state_dict'])
    return net


if __name__=='__main__':
    # read data
    X_train,y_train,X_test,y_test=read_bci_data()

    # EEGNet results
    EEGNet_checkpoints={
        'ELU':'./checkpoints/EEGNet/EEGNet_elu_5e-3_amsgrad_0.8407.pt',
        'ReLU':'./checkpoints/EEGNet/EEGNet_relu_1e-3_0.8731.pt',
        'LeakyReLU':'./checkpoints/EEGNet/EEGNet_leaky_relu_1e-2_init_amsgrad_0.8787.pt'
    }
    print("EEGNet results")
    demo(EEGNet_checkpoints,X_test,y_test)

    # DeepConvNet results
    DeepConvNet_checkpoints={
        'ELU':'./checkpoints/DeepConvNet/DeepConvNet_elu_1e-3_amsgrad_0.7454.pt',
        'ReLU':'./checkpoints/DeepConvNet/DeepConvNet_relu_1e-2_0.7102.pt',
        'LeakyReLU':'./checkpoints/DeepConvNet/DeepConvNet_leaky_relu_1e-2_init_amsgrad_0.7352.pt'
    }
    print("DeepConvNet results")
    demo(DeepConvNet_checkpoints,X_test,y_test)

