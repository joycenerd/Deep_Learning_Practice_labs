from torch.utils.data import Dataset,DataLoader
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
        return torch.from_numpy(self.X[index]),torch.from_numpy(self.y[index]) 

    