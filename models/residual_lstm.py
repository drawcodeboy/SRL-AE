import torch
import torch.nn as nn

class ResLSTMCell(nn.Module):
    '''
    Peephole LSTM을 Residual Connection시킨 LSTM 셀
    Args:
        - input_size: input dimension
        - hidden_size: cell state dimension
        - proj_size: hidden state dimension
    '''
    def __init__(self, input_size, hidden_size, proj_size=0):
        super(ResLSTMCell, self).__init__()
                
        self.register_buffer('input_size', torch.Tensor([input_size]))
        self.register_buffer('hidden_size', torch.Tensor([hidden_size]))
        self.register_buffer('proj_size', torch.Tensor([hidden_size if proj_size == 0 else proj_size]))
        
        # Input, Forget Gate의 가중치를 붙여서 한 번의 연산으로 구하도록 한 것
        # 대신 가독성은 떨어짐.
        self.weight_ii = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_ih = nn.Parameter(torch.randn(2 * hidden_size, proj_size))
        self.weight_ic = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))
        
        self.bias_ii = nn.Parameter(torch.randn(2 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(2 * hidden_size))
        self.bias_ic = nn.Parameter(torch.randn(2 * hidden_size))
        
        # Cell Gate
        self.weight_ci = nn.Parameter(torch.randn(1 * hidden_size, input_size))
        self.weight_ch = nn.Parameter(torch.randn(1 * hidden_size, proj_size))
        self.bias_c = nn.Parameter(torch.randn(1 * hidden_size))
        
        # Output Gate
        self.weight_oi = nn.Parameter(torch.randn(1 * proj_size, input_size))
        self.weight_oh = nn.Parameter(torch.randn(1 * proj_size, proj_size))
        self.weight_oc = nn.Parameter(torch.randn(1 * proj_size, hidden_size))
        self.bias_o = nn.Parameter(torch.randn(1 * proj_size))
        
        self.weight_r_proj = nn.Parameter(torch.randn(proj_size, hidden_size))
        self.weight_ir = nn.Parameter(torch.randn(proj_size, input_size))

    def forward(self, input, hidden=None, pre_output=None):
        '''
        Type Hints
            - input: torch.Tensor
            - pre_output: torch.Tensor, input 혹은 이전 레이어의 output - Residual Connection
            - hidden: Tuple[torch.Tensor, torch.Tensor]
            
            Returns:
                - Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        '''
        pre_output = input if pre_output is None else pre_output
        
        # Hidden State, Cell State
        hx, cx = hidden[0], hidden[1]
    
        # Cell State까지 Gate를 만드는 연산에 포함시키는 것은
        # 해당 LSTM이 Peephole LSTM이라 그럼
        # 즉, 아래 코드는 forget, input, output gate를 만들기 위함인 것
        if_gates = (torch.matmul(input, self.weight_ii.t()) + self.bias_ii +
                     torch.matmul(hx, self.weight_ih.t()) + self.bias_ih +
                     torch.matmul(cx, self.weight_ic.t()) + self.bias_ic)
        
        # (bz, cell_state), (bz, cell_state) = (bz, 2*cell_state)
        ingate, forgetgate = if_gates.chunk(2, -1) 
        
        # Cell Gate (Peephole LSTM 수식보면 여기는 이전 Cell state가 들어가지 않음.)
        cellgate = (torch.matmul(input, self.weight_ci.t()) + 
                    torch.matmul(hx, self.weight_ch.t()) + self.bias_c)
        
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        
        outgate = (torch.matmul(input, self.weight_oi.t()) + 
                   torch.matmul(hx, self.weight_oh.t()) + 
                   torch.matmul(cy, self.weight_oc.t()) + self.bias_o)
        
        outgate = torch.sigmoid(outgate)
        
        ry = torch.tanh(cy)
        ry = torch.matmul(ry, self.weight_r_proj.t())

        # 여기가 Residual LSTM의 핵심
        # 이전 LSTM Layer의 output과 현재 hidden state를 묶는 과정
        if self.input_size == self.proj_size:
          hy = outgate * (ry + pre_output)
        else:
          hy = outgate * (ry + torch.matmul(pre_output, self.weight_ir.t()))
          
        return hy, (hy, cy)

class ResLSTM(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 proj_size=0, 
                 batch_first=False):
        
        super(ResLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = hidden_size if proj_size == 0 else proj_size
        self.batch_first = batch_first
        
        self.cell = ResLSTMCell(input_size, hidden_size, proj_size)

    def forward(self, input, hidden=None, pre_output=None):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        
        inputs = input.unbind(-2)
        print(len(inputs))
        outputs = []
        
        pre_output = input if pre_output is None else pre_output
        pre_outputs = pre_output.unbind(-2)
        
        if hidden is None:
            # nn.LSTM에서도 따로 provide 하지 않으면, h0, c0은 초기에 zero로 간다.
            hidden = (torch.zeros(1, self.proj_size), torch.zeros(1, self.hidden_size))
        
        for i in range(len(inputs)):
            out, hidden = self.cell(inputs[i], hidden, pre_outputs[i])
            outputs += [out]
        outputs = torch.stack(outputs)
        
        if self.batch_first:
            outputs = outputs.permute(1, 0, 2) # (seq_len, bz, dim) -> (bz, seq_len, dim)
        
        return outputs, hidden
    
if __name__ == '__main__':
    cell = ResLSTMCell(input_size=3, hidden_size=5, proj_size=4)
        
    x = torch.randn(1, 3)
    hidden_state = torch.randn(1, 4)
    cell_state = torch.randn(1, 5)
    
    cell(x, (hidden_state, cell_state))
    
    model = ResLSTM(input_size=3, hidden_size=5, proj_size=4, batch_first=True)
    
    x = torch.randn(2, 20, 3)
    output, h_n = model(x)
    
    print(output.shape)