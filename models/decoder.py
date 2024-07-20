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
        super().__init__()
    
        self.num_layers = num_layers
        self.layers_dim = hidden_dims + [output_dim]
        self.lstm_layers = nn.ModuleList()
    
        for idx in range(self.num_layers):
            layer = nn.LSTM(
                input_size=self.layers_dim[idx],
                hidden_size=self.layers_dim[idx+1],
                batch_first=True,
            )
            self.lstm_layers.append(layer)
            
    def forward(self, x, seq_len):
        # Repeat Hidden State
        # (Batch Size, 1, Vector Dim) -> (Batch Size, Seq Len, Vector Dim)
        x = x.expand(x.shape[0], seq_len, x.shape[2])
        return x

if __name__ == '__main__':
    model = Decoder()
    
    input_t = torch.randn(2, 1, 2)
    output_t = model(input_t, 10)
    print(output_t.shape)
    print(input_t)
    print(output_t)
    
    
    # from torchinfo import summary
    # summary(model, (2, 10, 3))