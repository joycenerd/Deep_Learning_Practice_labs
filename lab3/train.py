from dataloader import RetinopathyLoader

from pathlib import Path


def train():
    train_set=RetinopathyLoader("./Data/data","train")




if __name__=="__main__":
    train()