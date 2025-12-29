from __future__ import annotations

import torch
import torch.nn as nn


class ChronosCNN(nn.Module):
    '''
    Phase 5: lightweight multi-head CNN.

    Input:  [B, 25, 8, 8] float planes (must match engine encoder).
    Outputs:
      - value       (tanh)   in [-1, 1]
      - pressure    (sigmoid) in [0, 1]
      - volatility  (relu)    in [0, +inf)   (we keep targets ~[0,1] initially)
      - complexity  (sigmoid) in [0, 1]
    '''

    def __init__(self, in_planes: int = 25, channels: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_planes, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Linear(128, 1)
        self.pressure = nn.Linear(128, 1)
        self.volatility = nn.Linear(128, 1)
        self.complexity = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(x)
        h = self.head(z)
        return {
            "value": torch.tanh(self.value(h)),
            "pressure": torch.sigmoid(self.pressure(h)),
            "volatility": torch.relu(self.volatility(h)),
            "complexity": torch.sigmoid(self.complexity(h)),
        }
