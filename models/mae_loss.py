import torch
from torch import nn

class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs:torch.Tensor, targets:torch.Tensor):
        '''
        Args:
            - outputs: (bz, seq_len, dim)
            - targets: (bz, seq_len, dim)
        Returns:
            - loss: (bz, loss)
        '''
        # (1) 각 차원에 대한 abs
        error = abs(outputs - targets)
        
        # (2) 다중 차원을 하나의 스칼라 값으로 나타내기 위한 평균 계산
        # dim = dim or dims to reduce
        error = torch.mean(error, dim=(-2, -1))
        
        return error