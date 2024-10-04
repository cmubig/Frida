import numpy as np
import torch
import torch.nn as nn

class MLP_VAE(nn.Module):
    def __init__(self, input_points_per_traj, latent_dim, output_points_per_traj):
        super(MLP_VAE, self).__init__()

        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_points_per_traj*3, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU()
        )
        self.mean_fc = nn.Linear(128, latent_dim)
        self.logvar_fc = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_points_per_traj*3),
            nn.Unflatten(1, (output_points_per_traj, 3)),
            nn.Tanh()
        )

        self.dummy_param = nn.Parameter(torch.empty(0))
    
    def encode(self, trajectory):
        encoded = self.encoder(trajectory)
        mean = self.mean_fc(encoded)
        logvar = self.logvar_fc(encoded)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def decode(self, z):
        if z.ndim == 1:
            z = z.unsqueeze(0)
        decoded = self.decoder(z)
        if z.shape[0] == 1:
            decoded = decoded.squeeze(0)
        decoded = decoded * torch.Tensor([0.2, 0.2, 1.0]).to(self.dummy_param.device)
        return decoded
    
    def forward(self, trajectory):
        mean, logvar = self.encode(trajectory)
        z = self.reparameterize(mean, logvar)
        decoded = self.decode(z)
        return decoded, mean, logvar
    
    def sample_trajectories(self, n):
        batch_size = 16
        res = []
        for i in range(0, n, batch_size):
            z = torch.randn(min(batch_size, n - i), self.latent_dim)
            z = z.to(self.dummy_param.device)
            res.append(self.decode(z).detach().cpu().numpy())
        res = np.concatenate(res, axis=0)
        return res
