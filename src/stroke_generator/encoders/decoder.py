import torch
import torch.nn as nn


class StrokeDecoder(nn.Module):
    def __init__(self, opt, device, latent_dim=128, palette_size=12):
        super(StrokeDecoder, self).__init__()
        self.opt = opt
        self.device = device
        self.latent_dim = latent_dim
        self.palette_size = palette_size

        ### Mean decoder
        self.dec_mu = nn.Sequential(
            nn.Linear(self.latent_dim, 7),
            nn.Tanh()
        ).to(device)

        ### RGB decoder
        self.dec_rgb = nn.Sequential(
            nn.Linear(self.latent_dim, self.palette_size),
            nn.Softmax(dim=-1)
        ).to(device)
        
        ### Log_std decoders
        self.dec_log_std = nn.Linear(self.latent_dim, 7).to(device)
        self.dec_log_std.weight.data.fill_(0.2)
        self.dec_log_std.bias.data.fill_(-2.0)
        self.log_std_min = -20  # Clamping for numerical stability
        self.log_std_max = -1.0 # Clamping for numerical stability

    def forward(self, x):
        mu = self.dec_mu(x)
        log_std = self.dec_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        rgb_logits = self.dec_rgb(x)
        return mu, log_std, rgb_logits

class DeterministicStrokeDecoder(StrokeDecoder):
    def __init__(self, opt, device, latent_dim=128, palette_size=12):
        super(DeterministicStrokeDecoder, self).__init__(opt, device, latent_dim, palette_size)

        ### Mean decoder
        # Outputs [l, z, b, a, x, y]
        self.dec_mu = nn.Sequential(
            nn.Linear(self.latent_dim, 6),
            nn.Tanh()
        ).to(device)

        ### Std decoder
        # Constant 0
        self.dec_log_std = None

    def forward(self, x):
        mu = self.dec_mu(x)
        std = torch.zeros_like(mu).detach()
        rgb_logits = self.dec_rgb(x)
        return mu, std, rgb_logits