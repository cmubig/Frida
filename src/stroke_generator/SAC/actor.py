import torch
from torch import nn

from stroke_generator.model import StrokePredictor

class StrokeActor(nn.Module):
    def __init__(self, opt, model: StrokePredictor, device):
        super().__init__()

        self.opt = opt
        self.device = device

        self.model = model
    
    def forward(self, encoded_inputs, hx=None):
        # Actor operates in the latent space
        # Expects shape [batch_size, n_strokes, ...]
        # hx is (hidden, cell) state for LSTM
        reshape = False
        if len(encoded_inputs.shape) == 2:
            # If only one stroke, add a time dimension
            encoded_inputs = encoded_inputs.unsqueeze(1)
            reshape = True
        
        out, hx = self.model.main(encoded_inputs, hx)

        if reshape:
            # Remove time dimension
            out = out.squeeze(1)

        return out, hx

    def decode_action(self, x):
        return self.model.stroke_decoder(x)
    
    def decode_and_sample_action(self, latent_action, color_palette):
        lzbaxy_mean, lzbaxy_log_std, rgb_logits = self.decode_action(latent_action)
        lzbaxy_sample = torch.distributions.Normal(lzbaxy_mean, lzbaxy_log_std.exp()).rsample()
        # Translate from [-1,1] to real params
        lzbaxy_sample = lzbaxy_sample * self.model.scale / 2.0 + self.model.bias 

        # Get angle from a_x and a_y
        a_x, a_y = lzbaxy_sample[:,3], lzbaxy_sample[:,4]
        a = torch.atan2(a_x, a_y).unsqueeze(-1)
        lzbaxy = torch.cat([lzbaxy_sample[:, :3],a,lzbaxy_sample[:,-2:]],dim=-1)

        rgb_idx = torch.distributions.Categorical(logits=rgb_logits).sample()

        stroke_tensor = torch.cat([lzbaxy, color_palette[torch.arange(lzbaxy.shape[0]),rgb_idx]],dim=-1)
        return stroke_tensor