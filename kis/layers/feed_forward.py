import torch.nn as nn
import torch


class FeedForward(nn.Module):
    """
    Feed Forward Layer.
    The forward of NN is: dim_input -> dim-hidden -> dim_input
    """
    def __init__(self, dim_input: int, dim_hidden: int, dropout=0.1):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_input)
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)
