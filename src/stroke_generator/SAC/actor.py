import clip
import torch
from torch import nn
import torchvision.transforms as transforms

from stroke_generator.utils.model_utils import ConvBlock, LinBlock, ScaledSigmoid, ScaledTanh


class StrokeActor(nn.Module):
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
        self.main = nn.GRU(
            input_size=self.encoding_hidden_size*5,
            hidden_size=self.decoding_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        ).to(device)
        # self.main = nn.Sequential(
        #     LinBlock(self.encoding_hidden_size*5, 1028, use_dropout=False),
        #     LinBlock(1028, 512, use_dropout=False),
        #     LinBlock(512, 256, use_dropout=False),
        #     LinBlock(256, self.decoding_hidden_size),
        # ).to(device)

        ################
        ### Decoders ###
        ################
        # l_min, l_max = self.opt.MIN_STROKE_LENGTH, self.opt.MAX_STROKE_LENGTH
        # z_min, z_max = self.opt.MIN_STROKE_Z, 0.95
        # b_scale = self.opt.MAX_BEND

        # self.dec_l = LinBlock(self.decoding_hidden_size, 1, act=ScaledSigmoid(l_min, l_max), is_final_layer=True).to(device)
        # self.dec_z = LinBlock(self.decoding_hidden_size, 1, act=ScaledSigmoid(z_min, z_max), is_final_layer=True).to(device)
        # self.dec_b = LinBlock(self.decoding_hidden_size, 1, act=ScaledTanh(b_scale), is_final_layer=True).to(device)
        # self.dec_a = LinBlock(self.decoding_hidden_size, 1, act=ScaledTanh(torch.pi), is_final_layer=True).to(device)
        # self.dec_xy = LinBlock(self.decoding_hidden_size, 2, act=nn.Tanh(), is_final_layer=True).to(device)
        # self.dec_rgb = LinBlock(self.decoding_hidden_size, 3, act=nn.Sigmoid(), is_final_layer=True).to(device)
        
        # Mean and log_std decoders
        self.dec_mu = LinBlock(self.decoding_hidden_size, 9)  # Predicts (l, z, b, a, xy(2), rgb(3))
        self.dec_log_std = LinBlock(self.decoding_hidden_size, 9)  # Log standard deviation
        self.log_std_min = -20  # Clamping for numerical stability
        self.log_std_max = 2

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
    
