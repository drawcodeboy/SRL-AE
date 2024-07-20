import torch
from torch import nn

'''
Dimension
8 -> 16 -> 32 -> 64 -> d (Decoder)
'''

class Decoder(nn.Module):
    def __init__(self, 
                 num_layers=4,
                 output_dim=1,
                 hidden_dims=[8, 16, 32, 64],
                 ):
        '''
        hidden_dims[0]: 맨 처음 input dim
        '''
        super().__init__()
    
        self.num_layers = num_layers
        self.layers_dim = hidden_dims
        self.lstm_layers = nn.ModuleList()
    
        for idx in range(self.num_layers - 1):
            layer = nn.LSTM(
                input_size=self.layers_dim[idx],
                hidden_size=self.layers_dim[idx+1],
                batch_first=True,
            )
            self.lstm_layers.append(layer)
            
        # Time Distributed Matrix (# Layer 4)
        self.tdm = nn.Parameter(
            torch.empty((self.layers_dim[-1], output_dim), dtype=torch.float32, requires_grad=True)
        )
        
    def forward(self, x, seq_len):
        # Repeat Hidden State
        # (Batch Size, 1, Vector Dim) -> (Batch Size, Seq Len, Vector Dim)
        x = x.expand(x.shape[0], seq_len, x.shape[2])

        for lstm in self.lstm_layers:
            x, (h_n, c_n) = lstm(x)
        
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