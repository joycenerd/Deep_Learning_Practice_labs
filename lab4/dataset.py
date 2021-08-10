from torch.utils.data import Dataset
import numpy as np
import torch

from pathlib import Path


class TextDataset(Dataset):
    def __init__(self,root_dir,mode):
        self.root_dir=root_dir
        self.mode=mode

        file=Path(self.root_dir).joinpath(f"{self.mode}.txt")
        self.data=np.loadtxt(file,dtype=str)

        if self.mode=='train':
            self.data=self.data.reshape(-1)
        else:
            self.tense=[[0,3],[0,2],[0,1],[0,1],[3,1],[0,2],[3,0],[2,0],[2,3],[2,1]]
            self.tense=np.asarray(self.tense,dtype=int)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.mode=='train':
            data=self.data[index]
            c=index%4
            return data,c
        else:
            data=self.data[index,0]
            target=self.data[index,1]
            c1=self.tense[index,0]
            c2=self.tense[index,1]
            return data,target,c1,c2


