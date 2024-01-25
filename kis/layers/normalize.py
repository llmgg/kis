import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    """
    Construct a normalization layer with Layer Norm method.
    """
    def __init__(self, dim: int, eps=1.0e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (torch.sqrt(var) + self.eps)
        return self.gamma * x + self.beta
