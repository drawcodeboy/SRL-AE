import torch
from torch import nn
import torch.nn.functional as F
from .residual_lstm import ResLSTM
from typing import Optional, List

class Encoder(nn.Module):
    def __init__(self, 
                 num_layers:int=4,
                 input_dim:int=1,
                 hidden_dims:List=[64, 32, 16, 8],
                 lstm_type:str='plain',
                 dropout:float=0.2,
                 sparsity_param:Optional[float]=None, # tanh -> -0.95 is inactive
                 sparsity_weight:Optional[float]=None): # 1e-4
        super().__init__()
        
        self.num_layers = num_layers
        self.layers_dim = [input_dim] + hidden_dims
        self.lstm_layers = nn.ModuleList()
        
        self.sparsity_param = sparsity_param
        self.sparsity_weight = sparsity_weight
        
        for idx in range(self.num_layers):
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
            
        self.acti = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        encoder_sparsity_loss = 0
        
        for idx, lstm in enumerate(self.lstm_layers):
            x, (h_n, c_n) = lstm(x)
            x = self.dropout(x)
            x = self.acti(x)
            
            if (self.sparsity_param is not None) and (idx <= self.num_layers-3):
                # Layer 1, 2, 3에 대해서만 Sparsity parameter 적용
                encoder_sparsity_loss += self.sparsity_loss(x)
        
        # Hidden state의 dimension 순서 순서, nn.LSTM in PyTorch Docs
        # Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details.
        # batch_first=True가 cell state와 hidden state에는 적용되지 않는다는 의미이다.
        h_n = h_n.permute(1, 0, 2) # (1, batch_size, vec_dim) => (batch_size, 1, vec_dim)
            
        # Decoder에서 Sequence Length만큼 repeat시킨다.
        # 이를 위해 Encoder에서는 마지막 LSTM layer에서 Hidden State를 넘겨주도록 한다.
        return h_n, encoder_sparsity_loss
    
    def sparsity_loss(self, x):
        # 코드 레퍼런스 (https://github.com/IParraMartin/Sparse-Autoencoder/blob/main/sae.py)
        # apply for each layer
        rho_hat = torch.mean(x, dim=0)
        rho_hat = (rho_hat+1.0)/2.0 # tanh 사용시에 Overflow 방지 -> [-1, 1] => [0, 1]
        
        rho = torch.ones_like(rho_hat) * self.sparsity_param
        epsilon = 1e-8 # Divide by zero 방지
        kl_divergence = F.kl_div((rho_hat + epsilon).log(), rho + epsilon, reduction='batchmean')
        
        return self.sparsity_weight * kl_divergence