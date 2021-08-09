from dataloader import RetinopathyLoader
from train import load_model

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch


def demo(checkpoints):
    device="cuda:0"

    # prepare data
    test_set=RetinopathyLoader("./Data/data","test",256)
    test_loader=DataLoader(test_set,batch_size=32,num_workers=4)
    test_size=len(test_set)

    for _model in checkpoints:
        print(_model)
        model_path=checkpoints[_model]

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


if __name__=='__main__':
    checkpoints={
        'resnet50_pretrained':'./checkpoints/best/resnet50_pretrained.pt',
        'resnet50':'./checkpoints/best/resnet50.pt',
        'resnet18_pretrained':'./checkpoints/best/resnet18_pretrained.pt',
        'resnet18':'./checkpoints/best/resnet18.pt',
    }

    demo(checkpoints)

