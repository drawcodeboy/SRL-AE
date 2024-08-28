import torch
from torch import nn
from .residual_lstm import ResLSTM

from typing import List

class Decoder(nn.Module):
    def __init__(self, 
                 num_layers:int=4,
                 output_dim:int=1,
                 hidden_dims:List[int]=[8, 16, 32, 64],
                 lstm_type:str='plain',
                 dropout:float=0.2):
        '''
        hidden_dims[0]: 맨 처음 input dim
        '''
        super().__init__()
    
        self.num_layers = num_layers
        self.layers_dim = hidden_dims
        self.lstm_layers = nn.ModuleList()
        
        for idx in range(self.num_layers - 1):
            if lstm_type == 'plain':
                layer = nn.LSTM(
                    input_size=self.layers_dim[idx],
                    hidden_size=self.layers_dim[idx+1],
                    batch_first=True,
                )
            elif lstm_type == 'res':
                layer = ResLSTM(
                    input_size=self.layers_dim[idx],
                    hidden_size=self.layers_dim[idx+1],
                    batch_first=True,
                )
            self.lstm_layers.append(layer)
            
        # Time Distributed Matrix (# Layer 4)
        self.tdm = nn.Parameter(
            torch.randn((self.layers_dim[-1], output_dim), dtype=torch.float32, requires_grad=True)
        )
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, seq_len):
        # Repeat Hidden State
        # (Batch Size, 1, Vector Dim) -> (Batch Size, Seq Len, Vector Dim)
        x = x.expand(x.shape[0], seq_len, x.shape[2])
        
        for idx, lstm in enumerate(self.lstm_layers):
            x, (h_n, c_n) = lstm(x)
            x = self.dropout(x)
            x = self.tanh(x)
        
        x = x @ self.tdm
        return x