from __future__ import annotations
import torch
import torch.nn as nn

class MLPValue(nn.Module):
    def __init__(self, input_dim: int = 1152, h1: int = 1024, h2: int = 512, h3: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
