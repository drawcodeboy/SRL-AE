import torch
from torch import nn
from .residual_lstm import ResLSTM

'''
Dimension
d -> 64 -> 32 -> 16 -> 8 (Encoder)
'''

class Encoder(nn.Module):
    def __init__(self, 
                 num_layers=4,
                 input_dim=1,
                 hidden_dims=[64, 32, 16, 8],
                 lstm_type:str='plain',
                 connect:bool=False,
                 dropout:float=0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers_dim = [input_dim] + hidden_dims
        self.lstm_layers = nn.ModuleList()
        
        self.connect = connect
        
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
            
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        connection_li = []
        
        for idx, lstm in enumerate(self.lstm_layers):
            x, (h_n, c_n) = lstm(x)
            x = self.dropout(x)
            x = self.tanh(x)
            
            if self.connect == True and idx != len(self.lstm_layers)-1:
                connection_li.append(x)
        
        # Decoder에서 사용하기 위해 reverse
        connection_li.reverse()
        
        # Hidden state의 dimension 순서 순서, nn.LSTM in PyTorch Docs
        # Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details.
        # batch_first=True가 cell state와 hidden state에는 적용되지 않는다는 의미이다.
        h_n = h_n.permute(1, 0, 2) # (1, batch_size, vec_dim) => (batch_size, 1, vec_dim)
            
        # Decoder에서 Sequence Length만큼 repeat시킨다.
        # 이를 위해 Encoder에서는 마지막 LSTM layer에서 Hidden State를 넘겨주도록 한다.
        return h_n, connection_li
            
if __name__ == '__main__':
    model = Encoder()
    print(model)
    
    # from torchsummary import summary
    from torchinfo import summary # for LSTM
    summary(model, input_size=(32, 160, 1)) # (Batch Size, Seq Len, Input Dim)
    
    input_t = torch.randn(4, 10, 1)
    x, h_n = model(input_t)
    print(x)
    print(h_n)