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
        self.hidden_fc=nn.Linear(self.hidden_size+self.c_hidden_size,self.hidden_size)
        self.cell_fc=nn.Linear(self.hidden_size+self.c_hidden_size,self.hidden_size)

        self.hidden_mean=nn.Linear(hidden_size,z_size)
        self.hidden_log_var=nn.Linear(hidden_size,z_size)
        self.cell_mean=nn.Linear(hidden_size,z_size)
        self.cell_log_var=nn.Linear(hidden_size,z_size)

        self.std_normal=torch.normal(torch.zeros(self.z_size),torch.ones(self.z_size)).to(device)

    def forward(self,x,hidden_state,cell_state,c):
        # hidden state
        c=self.c_embedding(c).reshape(1,1,-1)
        hidden_state=torch.cat((hidden_state,c),dim=2)
        hidden_state=self.hidden_fc(hidden_state)

        # cell state
        cell_state=torch.cat((cell_state,c),dim=2)
        cell_state=self.cell_fc(cell_state)
        
        x=self.embedding(x).reshape(-1,1,self.hidden_size)
        out,(hidden_state,cell_state)=self.lstm(x,(hidden_state,cell_state))

        # mean and variance
        hidden_mu=self.hidden_mean(hidden_state)
        hidden_log_sigma=self.hidden_log_var(hidden_state)
        hidden_eps=torch.normal(torch.zeros(self.z_size), torch.ones(self.z_size)).to(self.device)
        hidden_z=torch.exp(hidden_log_sigma/2)*hidden_eps+hidden_mu # reparameterization trick

        cell_mu=self.cell_mean(cell_state)
        cell_log_sigma=self.cell_log_var(cell_state)
        cell_eps=torch.normal(torch.zeros(self.z_size),torch.ones(self.z_size)).to(self.device)
        cell_z=torch.exp(cell_log_sigma/2)*self.std_normal+cell_mu # reparameterization trick

        return hidden_mu,hidden_log_sigma,hidden_z,cell_mu,cell_log_sigma,cell_z

    def init_hidden_and_cell(self):
        hidden_state=torch.zeros(1,1,self.hidden_size,device=self.device)
        cell_state=torch.zeros(1,1,self.hidden_size,device=self.device)
        return hidden_state,cell_state

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
        self.cell_fc=nn.Linear(z_size+c_hidden_size,hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self,x,hidden_z,cell_z,c):
        c=self.c_embedding(c).reshape(1,1,-1)
        hidden_state=torch.cat((hidden_z,c),dim=2)
        hidden_state=self.hidden_fc(hidden_state)
        cell_state=torch.cat((cell_z,c),dim=2)
        cell_state=self.cell_fc(cell_state)

        x=self.embedding(x).reshape(-1,1,self.hidden_size)
        x=F.relu(x)
        out,(hidden_state,cell_state)=self.lstm(x,(hidden_state,cell_state))
        out=self.fc(out).reshape(-1,self.output_size)
        return out,hidden_state,cell_state

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)