import torch
import torch.nn as nn


class ResnetConnection(nn.Module):
    """
    A residual connection begin with a normalization layer.
    """
    def __init__(self, dropout=0.0):
        super(ResnetConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        """
        Apply resnet to any sublayer whose size of input is the same as output
        """
        return x + self.dropout(sublayer(x))
