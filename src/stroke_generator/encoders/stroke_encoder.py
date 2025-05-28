import torch.nn as nn

# Based on the PointNet architecture
# Input dimension is 19 from [2 * 7 + 12]
# 2 * 7 is (mu, std) for stroke [L, Z, B, Ax, Ay, X, Y]
# 12 is the color palette logits
class StrokeNetEncoder(nn.Module):
    def __init__(self, input_dim=26, output_dim=256):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.final = nn.Linear(256, output_dim)

    def forward(self, x):               # [B, N, 18]
        x = self.main(x)                # [B, N, 256]
        x = x.transpose(1,2)            # [B, 256, N]
        x = self.pool(x).squeeze(-1)    # [B, 256]
        return self.final(x)
