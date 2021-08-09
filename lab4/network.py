import torch.nn.functional as F
import torch.nn as nn
import torch


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,c_size,c_hidden_size,z_size,device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size=input_size
        self.c_size=c_size
        self.c_hidden_size=c_hidden_size
        self.z_size=z_size
        self.device=device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.c_embedding=nn.Embedding(c_size,c_hidden_size)
        self.lstm=nn.LSTM(self.hidden_size,self.hidden_size)

        self.mean=nn.Linear(hidden_size,z_size)
        self.log_var=nn.Linear(hidden_size,z_size)
        self.std_normal=torch.normal(torch.zeros(self.z_size),torch.ones(self.z_size)).to(device)

    def forward(self,x,hidden,c):
        c=self.c_embedding(c).reshape(1,1,-1)
        hidden=torch.cat((hidden,c),dim=2)
        x=self.embedding(x).reshape(-1,1,self.hidden_size)
        out,hidden_out=self.lstm(x,hidden)

        mu=self.mean(hidden_out)
        log_sigma=self.log_var(hidden_out)
        z=torch.exp(log_sigma/2)*self.std_normal+mu

        return mu,log_sigma,z

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,c_size,c_hidden_size,z_size,device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size=output_size
        self.c_size=c_size
        self.c_hidden_size=c_hidden_size
        self.z_size=z_size
        self.device=device

        self.c_embedding=nn.Embedding(c_size,c_hidden_size)
        self.hidden_fc=nn.Linear(z_size+c_hidden_size,hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, z,c,hidden):
        c=self.c_embedding(c).reshape(1,1,-1)
        input=torch.cat((z,c),dim=2)
        input=self.hidden_fc(input)
        input=self.embedding(input).reshape(-1,1,self.hidden_size)
        input=F.relu(input)
        out,hidden_out=self.lstm(input,hidden)
        out=self.fc(out).reshape(-1,self.output_size)
        return out,hidden_out

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)