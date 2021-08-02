from dataloader import RetinopathyLoader
from train import load_model

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import torch.nn as nn
import numpy as np
import torch

import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--model-path",type=str,default="./checkpoints/best/resnet50.pt")
parser.add_argument("--device",type=str,default="cuda:1",help="use which device for training")
parser.add_argument("--batch-size",type=int,default=32,help="batch size for training")
parser.add_argument("--img-size",type=int,default=256,help="Size to resize the image")
args=parser.parse_args()


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
    gt=[]
    pred=[]

    for idx,(inputs,targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)

        outputs = model(inputs)
        test_loss += loss_func(outputs, targets).item()*inputs.shape[0]
        _, predicted = torch.max(outputs.data, 1)
        test_acc += (predicted == targets).sum().item()
        gt.extend(targets.cpu())
        pred.extend(predicted.cpu())

    test_loss /= test_size
    test_acc /= test_size
    print(f"test_loss: {test_loss:.4}\ttest_acc: {test_acc:.4}")
    
    gt=np.asarray(gt)
    pred=np.asarray(pred)
    return gt,pred


if __name__=="__main__":
    print(args)
    input("Press ENTER if no problem...")

    save_name="resnet50_confusion_mat"

    gt,pred=test()
    
    labels=[0,1,2,3,4]
    cm=confusion_matrix(gt,pred,labels,normalize='true')

    plt.rcParams["font.family"] = "serif"
    fig,ax=plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='.1f')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground truth')
    ax.xaxis.set_ticklabels(labels, rotation=45)
    ax.yaxis.set_ticklabels(labels, rotation=0)
    plt.title('ResNet50 Normalized comfusion matrix')
    plt.savefig(f"./results/{save_name}.jpg")