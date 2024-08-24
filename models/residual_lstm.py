import torch
import torch.nn as nn
import sys
import math

class ResLSTMCell(nn.Module):
    '''
    Peephole LSTM을 Residual Connection시킨 LSTM 셀
    Args:
        - input_size: input dimension
        - hidden_size: cell state dimension
        - proj_size: hidden state dimension
    '''
    def __init__(self, input_size, hidden_size, proj_size=0):
        '''
        weights_{gate}_{input or h_state or c_state} except weights_r
        bias_{gate}
        '''
        super(ResLSTMCell, self).__init__()
        
        proj_size = hidden_size if proj_size == 0 else proj_size
                
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        
        # Input, Forget Gate의 가중치를 붙여서 한 번의 연산으로 구하도록 한 것
        # 대신 가독성은 떨어짐.
        self.weight_if_x = nn.Parameter(torch.randn(input_size, 2 * hidden_size))
        self.weight_if_h = nn.Parameter(torch.randn(proj_size, 2 * hidden_size))
        self.weight_if_c = nn.Parameter(torch.randn(hidden_size, 2 * hidden_size))
        
        self.bias_i = nn.Parameter(torch.randn(1 * hidden_size))
        self.bias_f = nn.Parameter(torch.randn(1 * hidden_size))
        
        # Cell Gate
        self.weight_c_x = nn.Parameter(torch.randn(input_size, 1 * hidden_size))
        self.weight_c_h = nn.Parameter(torch.randn(proj_size, 1 * hidden_size))
        self.bias_c = nn.Parameter(torch.randn(1 * hidden_size))
        
        # Output Gate
        self.weight_o_x = nn.Parameter(torch.randn(input_size, 1 * proj_size))
        self.weight_o_h = nn.Parameter(torch.randn(proj_size, 1 * proj_size))
        self.weight_o_c = nn.Parameter(torch.randn(hidden_size, 1 * proj_size))
        self.bias_o = nn.Parameter(torch.randn(1 * proj_size))
        
        self.weight_r_proj = nn.Parameter(torch.randn(hidden_size, proj_size))
        self.weight_r_x = nn.Parameter(torch.randn(input_size, proj_size))

        self._init_weights()
        
    def _init_weights(self):
        # *****여기서 가장 중요한 코드*****
        # 혹은 위 파라미터들을 전부 torch.empty()로 만들어주고 학습해야 함.
        stdv = 1.0 / math.sqrt(self.proj_size) # proj_size = 실제 hidden_size
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x_t, hidden=None):
        '''
        Type Hints
            - x_t: torch.Tensor
            - hidden: Tuple[torch.Tensor, torch.Tensor]
            
            Returns:
                - Output Sequence, (n-th Hidden State, n-th Cell State)
                - Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        '''
        
        # Hidden State, Cell State
        h_t, c_t = hidden[0], hidden[1]
    
        # Cell State까지 Gate를 만드는 연산에 포함시키는 것은
        # 해당 LSTM이 Peephole LSTM이라 그럼
        # 즉, 아래 코드는 forget, input, output gate를 만들기 위함인 것
        if_gates = (torch.matmul(x_t, self.weight_if_x) + 
                     torch.matmul(h_t, self.weight_if_h) + 
                     torch.matmul(c_t, self.weight_if_c))
        
        # (bz, cell_state), (bz, cell_state) = (bz, 2*cell_state)
        ingate, forgetgate = if_gates.chunk(2, -1) 
        
        ingate = ingate + self.bias_i
        forgetgate = forgetgate + self.bias_f
                
        # Cell Gate (Peephole LSTM 수식보면 여기는 이전 Cell state가 들어가지 않음.)
        cellgate = (torch.matmul(x_t, self.weight_c_x) + 
                    torch.matmul(h_t, self.weight_c_h) + self.bias_c)
        
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)

        c_t = (forgetgate * c_t) + (ingate * cellgate)
        # c_t_ = c_t.detach()
        outgate = (torch.matmul(x_t, self.weight_o_x) + 
                   torch.matmul(h_t, self.weight_o_h) + 
                   torch.matmul(c_t, self.weight_o_c) + self.bias_o)
        
        outgate = torch.sigmoid(outgate)
        
        r_t = torch.tanh(c_t) # c_t => r_t
        # r_t = r_t.detach() # Cell state에 대한 graident flow를 끊어줌
        m_t = torch.matmul(r_t, self.weight_r_proj)

        # 여기가 Residual LSTM의 핵심
        # 이전 LSTM Layer의 output과 현재 hidden state를 묶는 과정
        if self.input_size == self.proj_size:
          h_t = outgate * (m_t + x_t)
        else:
          h_t = outgate * (m_t + torch.matmul(x_t, self.weight_r_x))
          
        return h_t, c_t

class ResLSTM(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 proj_size=0, 
                 batch_first=False):
        
        super(ResLSTM, self).__init__()
        
        proj_size = (hidden_size if proj_size == 0 else proj_size)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.batch_first = batch_first
        
        self.cell = ResLSTMCell(input_size, hidden_size, proj_size)

    def forward(self, x, hidden=None):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        
        outputs = []
        
        if hidden is None:
            # nn.LSTM에서도 따로 provide 하지 않으면, h0, c0은 초기에 zero로 간다.
            hidden = (torch.zeros(1, self.proj_size).to(x.device), 
                      torch.zeros(1, self.hidden_size).to(x.device))
        
        seq_len = x.size()[1]
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            hidden = self.cell(x_t, hidden)
            outputs += [hidden[0]]
        outputs = torch.stack(outputs)
        
        if self.batch_first:
            outputs = outputs.permute(1, 0, 2) # (seq_len, bz, dim) -> (bz, seq_len, dim)
        
        # nn.LSTM처럼 맨 앞에 dim=1 추가해서 리턴
        h_n = hidden[0].unsqueeze(0)
        c_n = hidden[1].unsqueeze(0)
        
        return outputs, (h_n, c_n)
    
if __name__ == '__main__':
    # cell = ResLSTMCell(input_size=3, hidden_size=5, proj_size=4)
        
    # x = torch.randn(20, 3)
    # hidden_state = torch.randn(20, 4)
    # cell_state = torch.randn(20, 5)
    
    # cell(x, (hidden_state, cell_state))
    
    model1 = ResLSTM(input_size=4, hidden_size=8, batch_first=True)
    model2 = nn.LSTM(input_size=4, hidden_size=8, batch_first=True)
    
    p_sum = 0
    for p in model1.parameters():
        p_sum += p.numel()
    print(f"resLSTM: {p_sum}")
    
    p_sum = 0
    for p in model2.parameters():
        p_sum += p.numel()
    print(f"nn.LSTM: {p_sum}")
    # x = torch.randn(16, 20, 3)
    # output, (h_n, c_n) = model(x)
    
    # print(output.shape)
    # print(h_n.shape, c_n.shape)