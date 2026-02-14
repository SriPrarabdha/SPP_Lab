import torch.nn as nn
import torch

class NonLinearModel(nn.Module):
    def __init__(self, num_feats=257):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_feats, num_feats//2),
            nn.ReLU(),
            nn.Linear(num_feats//2, 2)
        )

    def forward(self, x):
        x = x.mean(-1)
        return self.net(x)

class ConvModel(nn.Module):
    def __init__(self, num_feats=64):  # ⭐ changed from 257 → 64
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(num_feats, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 2, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, x):
        x = self.net(x)      # (B, 2, T')
        return x.mean(-1)    # global average over time


class LSTMModel(nn.Module):
    def __init__(self, num_feats=64, hidden=32):  # ⭐ 257 → 64
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_feats,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, T, F)

        _, (h, _) = self.lstm(x)

        h = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2H)

        return self.fc(h)
