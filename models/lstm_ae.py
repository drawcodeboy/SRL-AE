from typing import List, Optional

import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder

class LSTMAutoencoder(nn.Module):
    def __init__(self,
                 num_layers:int= 5,
                 in_out_dim:int= 1,
                 hidden_dims:List[int]= [128, 64, 32, 16, 8],
                 encoder_lstm_type:str='plain',
                 decoder_lstm_type:str='plain',
                 dropout:float=0.2,
                 sparsity_param:Optional[float]=None, # tanh -> -0.95 is inactive
                 sparsity_weight:Optional[float]=None): # 1e-4
        '''
        Args:
            - num_layers: Encoder와 Decoder의 Layer depth
            - in_out_dim: Input(Output)으로 들어가는 Sequence의 Dimension
            - hidden_dims:
                - Encoder에서는 hidden_dims에 담긴 리스트 순서대로
                  element가 Output dimension의 크기가 되고,
                - Decoder에서는 hidden_dims의 element가
                  Input dimension의 크기가 된다.
            - encoder_lstm_type & decoder_lstm_type: LSTM의 종류 (plain, res)
                - 'plain': 일반적인 LSTM
                - 'res': Residual LSTM
                - 'peephole': Peephole LSTM

        Description:
            사용되는 LSTM 자체는 Encoder에서 num_layers개,
            Decoder에서 (num_layers-1)개로 총 (2*num_layers-1)개이다.
        '''
        if num_layers != len(hidden_dims):
            raise AssertionError("Not Equal num_layers, len(hidden_dims)")
        
        if encoder_lstm_type not in ['plain', 'res']:
            raise AssertionError("Check your Encoder LSTM type")
        
        if decoder_lstm_type not in ['plain', 'res']:
            raise AssertionError("Check your Encoder LSTM type")
        
        super().__init__()
        
        self.enc_sparsity_loss = None
        
        self.encoder = Encoder(num_layers=num_layers,
                               input_dim=in_out_dim,
                               hidden_dims=hidden_dims,
                               lstm_type=encoder_lstm_type,
                               dropout=dropout,
                               sparsity_param=sparsity_param,
                               sparsity_weight=sparsity_weight)
        
        self.decoder = Decoder(num_layers=num_layers,
                               output_dim=in_out_dim,
                               hidden_dims=hidden_dims[::-1],
                               lstm_type=decoder_lstm_type,
                               dropout=dropout)
    
    def forward(self, x):
        seq_len = x.shape[1]
        x, enc_sparsity_loss = self.encoder(x)
        self.enc_sparsity_loss = enc_sparsity_loss
        
        x = self.decoder(x, seq_len)
        return x