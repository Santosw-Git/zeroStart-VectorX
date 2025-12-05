
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch    
import torch.nn as nn
from layerNormalization.geluactivation import GELU
from config.config import CONFIG_124M
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

print(CONFIG_124M["emb_dim"])

ffn = FeedForward(CONFIG_124M)
x = torch.rand(2, 3, 768) 
out = ffn(x)
print(out.shape)

# comment it
