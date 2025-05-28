import clip
import torch
from torch import nn

from stroke_generator.utils.model_utils import ConvBlock, LinBlock, ScaledSigmoid, ScaledTanh
from stroke_generator.encoders.encoders import StrokeEncoder

class StrokeCritic(nn.Module):
    def __init__(self, opt, device, state_latent_size=2560, action_latent_size=1024, hidden_size=1024):
        super().__init__()

        self.opt = opt
        self.device = device
        self.hidden_size = hidden_size

        self.stroke_encoder = StrokeEncoder(opt, device, action_latent_size)

        # Main block
        self.q2 = QNet(opt, device, state_latent_size, action_latent_size, hidden_size).to(device)
        self.q1 = QNet(opt, device, state_latent_size, action_latent_size, hidden_size).to(device)

    def forward(self, encoded_states, encoded_actions, hidden_cells=(None, None)):
        # Expects shape [batch_size, n_strokes, ...]
        q1, hc_1 = self.q1(encoded_states, encoded_actions, hidden_cells[0])
        q2, hc_2 = self.q2(encoded_states, encoded_actions, hidden_cells[1])
        return q1,q2,(hc_1, hc_2)

class QNet(nn.Module):
    def __init__(self, opt, device, state_latent_dim, act_latent_dim, hidden_size=256):
        super().__init__()

        self.opt = opt
        self.device = device

        self.state_latent_dim = state_latent_dim
        self.act_latent_dim = act_latent_dim
        self.hidden_size = hidden_size
        
        self.state_net = nn.Sequential(
            nn.LayerNorm(state_latent_dim),
            nn.Linear(state_latent_dim, hidden_size),
        ).to(device)
        self.act_net = nn.Sequential(
            nn.LayerNorm(act_latent_dim),
            nn.Linear(act_latent_dim, hidden_size),
        ).to(device)
        self.combine_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU()
        ).to(device)

        self.LSTM = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        ).to(device)
        self.skip_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        ).to(device)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1),
        ).to(device)
    
    def forward(self, latent_state, latent_action, hidden_cell=None):
        _s = self.state_net(latent_state)
        _a = self.act_net(latent_action)
        x = torch.cat((_s, _a), dim=-1)
        x = self.combine_net(x)

        x_skip = self.skip_net(x)

        reshape = False
        if len(x.shape) == 2:
            # If only one stroke, add a time dimension
            x = x.unsqueeze(1)
            reshape = True
        
        x, hidden_cell = self.LSTM(x, hidden_cell)

        if reshape:
            # Remove time dimension
            x = x.squeeze(1)
        
        x = self.out(x + x_skip)
        return x, hidden_cell