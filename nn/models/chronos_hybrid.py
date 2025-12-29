from __future__ import annotations

import torch
import torch.nn as nn

from nn.move_index import MOVE_SPACE


class ChronosHybridNet(nn.Module):
    '''
    Hybrid net with:
      - policy logits: [B, MOVE_SPACE]
      - value in [-1,1]
      - pressure in [0,1]
      - volatility in [0,+inf)
      - complexity in [0,1]
    '''

    def __init__(self, in_planes: int = 25, channels: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, MOVE_SPACE),
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
            "policy": self.policy_head(z),
            "value": torch.tanh(self.value(h)),
            "pressure": torch.sigmoid(self.pressure(h)),
            "volatility": torch.relu(self.volatility(h)),
            "complexity": torch.sigmoid(self.complexity(h)),
        }
