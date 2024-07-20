from typing import List

import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder

class LSTMAutoencoder(nn.Module):
    def __init__(self,
                 num_layers:int= 4,
                 in_out_dim:int= 1,
                 hidden_dims:List[int]=[64, 32, 16, 8]):
        '''
        Args:
            - num_layers: Encoder와 Decoder의 Layer depth
            - in_out_dim: Input(Output)으로 들어가는 Sequence의 Dimension
            - hidden_dims:
                - Encoder에서는 hidden_dims에 담긴 리스트 순서대로
                  element가 Output dimension의 크기가 되고,
                - Decoder에서는 hidden_dims의 element가
                  Input dimension의 크기가 된다.
        Description:
            사용되는 LSTM 자체는 Encoder에서 num_layers개,
            Decoder에서 (num_layers-1)개로 총 (2*num_layers-1)개이다.
        '''
        if num_layers != len(hidden_dims):
            raise AssertionError("Not Equal num_layers, len(hidden_dims)")
        
        super().__init__()
        
        self.encoder = Encoder(num_layers=num_layers,
                               input_dim=in_out_dim,
                               hidden_dims=hidden_dims)
        self.decoder = Decoder(num_layers=num_layers,
                               output_dim=in_out_dim,
                               hidden_dims=hidden_dims[::-1])
    
    def forward(self, x):
        seq_len = x.shape[1]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)
        return x

if __name__ == '__main__':
    model = LSTMAutoencoder(num_layers=5,
                            hidden_dims=[128, 64, 32, 16, 8])
    from torchinfo import summary
    summary(model, input_size=(1, 10, 1))