from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch


class EEGDataset(Dataset):
    # EEG dataset class
    def __init__(self,X,y):
        """
        :param X: data
        :param y: label
        """
        self.X=X
        self.y=y

    
    def __len__(self):
        return len(self.y)

    
    def __getitem__(self,index):
        input=np.asarray(self.X[index])
        target=np.asarray(self.y[index])
        return torch.from_numpy(input),torch.from_numpy(target) 

    