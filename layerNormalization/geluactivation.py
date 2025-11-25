import torch
import torch.nn as nn
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

torch.manual_seed(123)
gelu = GELU()
inputs = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [0.5, -0.5, 2.0]]
)   
outputs = gelu(inputs)
print("GELU Outputs:\n", outputs)