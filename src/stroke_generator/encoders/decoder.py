import torch
import torch.nn as nn


class StrokeDecoder(nn.Module):
    def __init__(self, opt, device, latent_dim=128, pallete_size=12):
        super(StrokeDecoder, self).__init__()
        self.opt = opt
        self.device = device
        self.latent_dim = latent_dim
        self.pallete_size = pallete_size

        ### Mean decoders
        # self.dec_mu_l = LinBlock(self.latent_dim, 1, act=ScaledSigmoid(l_min, l_max), is_final_layer=True).to(device)
        # self.dec_mu_z = LinBlock(self.latent_dim, 1, act=ScaledSigmoid(z_min, z_max), is_final_layer=True).to(device)
        # self.dec_mu_b = LinBlock(self.latent_dim, 1, act=ScaledTanh(b_scale), is_final_layer=True).to(device)
        # self.dec_mu_a = LinBlock(self.latent_dim, 2, act=nn.Tanh(), is_final_layer=True).to(device)
        # self.dec_mu_xy = LinBlock(self.latent_dim, 2, act=nn.Tanh(), is_final_layer=True).to(device)
        self.dec_mu = nn.Sequential(
            nn.Linear(self.latent_dim, 7),
            nn.Tanh()
        ).to(device)

        ### RGB decoder
        self.dec_rgb = nn.Linear(self.latent_dim, self.pallete_size).to(device) # RGB logits (needs softmaxxing)
        
        ### Log_std decoders
        self.dec_log_std = nn.Linear(self.latent_dim, 7).to(device)
        self.dec_log_std.weight.data.fill_(0.2)
        self.dec_log_std.bias.data.fill_(-2.0)
        self.log_std_min = -20  # Clamping for numerical stability
        self.log_std_max = -1.0 # Clamping for numerical stability
    
    # def forward(self, x):        
    #     # Decode Mu
    #     l_mu = self.dec_mu_l(x)
    #     z_mu = self.dec_mu_z(x)
    #     b_mu = self.dec_mu_b(x)
    #     a_mu = self.dec_mu_a(x)
    #     xy_mu = self.dec_mu_xy(x)
    #     lzbaxy_mean = torch.cat([l_mu, z_mu, b_mu, a_mu, xy_mu], dim=-1)

    #     # Decode Log_std
    #     lzbaxy_log_std = self.dec_log_std(x)
    #     lzbaxy_log_std = torch.clamp(lzbaxy_log_std, self.log_std_min, self.log_std_max)

    #     # Decode RGB discrete logits
    #     rgb_logits = self.dec_rgb(x)
    #     return lzbaxy_mean, lzbaxy_log_std, rgb_logits

    def forward(self, x):
        mu = self.dec_mu(x)
        log_std = self.dec_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        rgb_logits = self.dec_rgb(x)
        return mu, log_std, rgb_logits