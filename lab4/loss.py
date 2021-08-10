import torch.nn as nn
import torch


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss,self).__init__()

    def forward(self,mean,log_var):
        return torch.sum(0.5 * (-log_var + (mean**2) + torch.exp(log_var) - 1))