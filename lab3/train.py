from dataloader import RetinopathyLoader
from resnet import ResNetLayer,BasicBlock,Encoder

from torch.utils.data import DataLoader

from pathlib import Path


def train():
    # data preprocessing
    train_set=RetinopathyLoader("./Data/data","train")
    train_loader=DataLoader(train_set,batch_size=4,shuffle=True,num_workers=4)

    # model
    print(Encoder(3,blocks_sizes=[64,128,256,512],deepths=[3,4,6,3],act='relu',block=BasicBlock))




if __name__=="__main__":
    train()