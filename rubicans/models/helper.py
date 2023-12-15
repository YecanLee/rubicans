import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self, 
                 x:torch.Tensor,
                 ):
        super().__init__()
        self.x = x

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = x * torch.sigmoid(x)
        return output