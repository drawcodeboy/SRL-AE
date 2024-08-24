import torch
from torch import nn
from .residual_lstm import ResLSTM
from typing import Optional

'''
Dimension
8 -> 16 -> 32 -> 64 -> d (Decoder)
'''

class Decoder(nn.Module):
    def __init__(self, 
                 num_layers=4,
                 output_dim=1,
                 hidden_dims=[8, 16, 32, 64],
                 lstm_type:str='plain',
                 connect:str='none',
                 num_attn_heads:Optional[int]=None,
                 dropout:float=0.2):
        '''
        hidden_dims[0]: 맨 처음 input dim
        '''
        super().__init__()
    
        self.num_layers = num_layers
        self.layers_dim = hidden_dims
        self.lstm_layers = nn.ModuleList()
        
        self.connect = connect
        
        self.attn_li = None
        if connect == 'cross-att':
            self.attn_li = nn.ModuleList()
            for idx in range(1, len(self.layers_dim)):
                self.attn_li.append(nn.MultiheadAttention(embed_dim=self.layers_dim[idx], 
                                                          num_heads=num_attn_heads,
                                                          batch_first=True))
    
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
            elif lstm_type == 'peephole':
                layer = PeepholeLSTM(
                    input_sz=self.layers_dim[idx],
                    hidden_sz=self.layers_dim[idx+1],
                    peephole=True
                )
            self.lstm_layers.append(layer)
            
        # Time Distributed Matrix (# Layer 4)
        self.tdm = nn.Parameter(
            torch.randn((self.layers_dim[-1], output_dim), dtype=torch.float32, requires_grad=True)
        )
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, seq_len, skip):
        # Repeat Hidden State
        # (Batch Size, 1, Vector Dim) -> (Batch Size, Seq Len, Vector Dim)
        x = x.expand(x.shape[0], seq_len, x.shape[2])

        for idx, lstm in enumerate(self.lstm_layers):
            x, (h_n, c_n) = lstm(x)
            x = self.dropout(x)
            
            if self.connect == 'none':
                pass
            elif self.connect == 'skip':
                x += skip[idx]
            elif self.connect == 'cross-att':
                # x: Query, skip[idx]: key, skip[idx]: value
                # Encoder의 역순으로 복원하는 구조기 때문에
                # Encoder의 output을 value로 도출한다는 의미로
                # Query, Key, Value를 다음과 같이 배치
                x, _ = self.attn_li[idx](x, skip[idx], skip[idx])
                
            x = self.tanh(x)
        
        x = x @ self.tdm
        return x
        
if __name__ == '__main__':
    model = Decoder()
    
    input_t = torch.randn(2, 1, 8)
    output_t = model(input_t, 10)
    
    print(output_t.shape)
    
    from torchinfo import summary
    # forward 시킬 게 2개라서 랜덤 텐서, seq_len 넣어줌.
    summary(model, input_data=[input_t, 100])