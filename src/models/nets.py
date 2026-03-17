\
from __future__ import annotations
import torch
import torch.nn as nn

class ConvBackbone(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, C, L)
        h = self.net(x)
        h = self.pool(h).squeeze(-1)  # (B, hidden)
        return h

class PhaseClassifier(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 4):
        super().__init__()
        self.backbone = ConvBackbone(in_ch, hidden=96)
        self.head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)

class TorqueRegressor(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.backbone = ConvBackbone(in_ch, hidden=96)
        self.head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h).squeeze(-1)
