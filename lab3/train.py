from dataloader import RetinopathyLoader
from resnet import BottleNeckBlock

from torch.utils.data import DataLoader

from pathlib import Path


def train():
    # data preprocessing
    train_set=RetinopathyLoader("./Data/data","train")
    train_loader=DataLoader(train_set,batch_size=4,shuffle=True,num_workers=4)

    # model
    print(BottleNeckBlock(32,64))




if __name__=="__main__":
    train()