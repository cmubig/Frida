import clip
import torch
from torch import nn

from stroke_generator.utils.model_utils import ConvBlock, LinBlock, ScaledSigmoid, ScaledTanh

class StrokeCritic(nn.Module):
    def __init__(self, opt, device='cpu'):
        super().__init__()

        self.opt = opt
        self.device = device

        self.img_size = 224
        self.max_pallete_size = 12
        self.encoding_hidden_size = 512
        self.decoding_hidden_size = 128


        ##################
        ### Main block ###
        ##################
        self.q1 = nn.Sequential(
            nn.GRU(
                input_size=self.encoding_hidden_size*5,
                hidden_size=self.decoding_hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            ),
            LinBlock(self.decoding_hidden_size, 1, is_final_layer=True)
        ).to(device)
        self.q2 = nn.Sequential(
            nn.GRU(
                input_size=self.encoding_hidden_size*5,
                hidden_size=self.decoding_hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            ),
            LinBlock(self.decoding_hidden_size, 1, is_final_layer=True)
        ).to(device)

    def forward(self, encoded_inputs):

        _, hidden = self.main(torch.cat(encoded_inputs, dim=1))
        hidden = hidden[-1]  # Get the last layer's hidden state

        # l_dec = self.dec_l(x)
        # z_dec = self.dec_z(x)
        # b_dec = self.dec_b(x)
        # a_dec = self.dec_a(x)
        # xy_dec = self.dec_xy(x)
        # rgb_dec = self.dec_rgb(x)
        
        # # Returns [l, z, b, a, x, y, r, g, b]
        # return torch.cat([l_dec, z_dec, b_dec, a_dec, xy_dec, rgb_dec], dim=1)

        mu = self.dec_mu(hidden)
        log_std = self.dec_log_std(hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std

