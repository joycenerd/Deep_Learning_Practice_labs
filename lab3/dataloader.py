from torchvision import transforms
from torch.utils import data
from PIL import Image
import pandas as pd
import numpy as np
import torch

from pathlib import Path

from torchvision.transforms.transforms import RandomOrder, RandomVerticalFlip


def getData(mode,csv_dir):
    if mode == 'train':
        img = pd.read_csv(Path(csv_dir).joinpath('train_img.csv'))
        label = pd.read_csv(Path(csv_dir).joinpath('train_label.csv'))
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv(Path(csv_dir).joinpath('test_img.csv'))
        label = pd.read_csv(Path(csv_dir).joinpath('test_label.csv'))
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode,Path(self.root).parent.absolute())
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        # read in the actual image
        image_path=Path(self.root).joinpath(self.img_name[index]+".jpeg")
        img=Image.open(image_path).convert('RGB')

        if self.mode=="train":
            transform=[
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation((90,90)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1)
            ]
            data_transform=transforms.Compose([
                transforms.RandomOrder(transform),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        else:
            data_transform=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        
        img=data_transform(img)
        label=torch.from_numpy(np.asarray(self.label[index]))

        return img, label
