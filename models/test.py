import torch
from torch import nn
from torchsummary import summary

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.li = nn.Linear(3, 2)
        a_tensor = torch.randn((3, 2))
        
        self.register_buffer('random buff', a_tensor)
        
    def forward(self, x):
        return x

if __name__ == '__main__':
    model = Model()
    summary(model, (5, 3))