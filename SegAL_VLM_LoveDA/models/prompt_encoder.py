import torch
import torch.nn as nn


class PromptEncoder(nn.Module):
    def __init__(self, text_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(self.text_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)
        x = self.norm(text_features)
        return self.mlp(x)
