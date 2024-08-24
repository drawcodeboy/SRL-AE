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
        
        # (2) Reconstruction에 대해 original과 큰 격차(error)를 학습하기 위해서 샘플에 대해서 평균이 아니라
        # sum을 하여 error를 크게 만들도록 하였음. 물론 배치에 대해서는 샘플 에러의 평균 값을 매겨서 학습시킴
        # dim = dim or dims to reduce
        error = torch.sum(error, dim=(-2, -1))
        
        return error