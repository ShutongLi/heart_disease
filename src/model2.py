import torch
import torch.nn as nn


class BaseMLP2(nn.Module):
    def __init__(self, feature_space=21):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_space, 4),
            nn.Sigmoid(),
            nn.Linear(4, 2),
        )

    def forward(self, X):
        return self.layers(X)

