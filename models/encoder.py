import torch
from torch import nn

'''
Dimension
d -> 64 -> 32 -> 16 -> 8 (Encoder)
'''

class Encoder(nn.Module):
    def __init__(self, 
                 num_layers=4,
                 input_dim=1,
                 hidden_dims=[64, 32, 16, 8]):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers_dim = [input_dim] + hidden_dims
        self.lstm_layers = nn.ModuleList()
        
        for idx in range(self.num_layers):
            layer = nn.LSTM(
                input_size=self.layers_dim[idx],
                hidden_size=self.layers_dim[idx+1],
                batch_first=True,
            )
            self.lstm_layers.append(layer)
    
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, (h_n, c_n) = lstm(x)
            
        # Decoder에서 Sequence Length만큼 repeat시킨다.
        # 이를 위해 Encoder에서는 마지막 LSTM layer에서 Hidden State를 넘겨주도록 한다.
        return h_n
            
if __name__ == '__main__':
    model = Encoder()
    print(model)
    
    # from torchsummary import summary
    from torchinfo import summary # for LSTM
    summary(model, input_size=(32, 160, 1)) # (Batch Size, Seq Len, Input Dim)
    
    input_t = torch.randn(1, 10, 1)
    h_n = model(input_t)
    print(h_n)
    print(h_n.shape)