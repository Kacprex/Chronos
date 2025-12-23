import torch.nn as nn
import torch.nn.functional as F
from src.nn.encoding import MOVE_SPACE


class ResidualBlock(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ChessNet(nn.Module):
    def __init__(self, channels=128, num_res_blocks=5):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(18, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, MOVE_SPACE)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
