import clip
import torch
from torch import nn
import torchvision.transforms as transforms

from stroke_generator.utils.model_utils import ConvBlock, LinBlock, ScaledSigmoid, ScaledTanh
from stroke_generator.utils.encoders import CanvasEncoder


class StrokePredictor(nn.Module):
    def __init__(self, opt, device='cpu'):
        super().__init__()

        self.opt = opt
        self.device = device

        self.img_size = 224
        self.max_pallete_size = 12
        self.encoding_hidden_size = 512
        self.decoding_hidden_size = 128

        # Encoder
        self.state_encoder = CanvasEncoder(opt, device)

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

        ################
        ### Decoders ###
        ################
        l_min, l_max = self.opt.MIN_STROKE_LENGTH, self.opt.MAX_STROKE_LENGTH
        z_min, z_max = self.opt.MIN_STROKE_Z, 0.95
        b_scale = self.opt.MAX_BEND

        self.dec_l = LinBlock(self.decoding_hidden_size, 1, act=ScaledSigmoid(l_min, l_max), is_final_layer=True).to(device)
        self.dec_z = LinBlock(self.decoding_hidden_size, 1, act=ScaledSigmoid(z_min, z_max), is_final_layer=True).to(device)
        self.dec_b = LinBlock(self.decoding_hidden_size, 1, act=ScaledTanh(b_scale), is_final_layer=True).to(device)
        self.dec_a = LinBlock(self.decoding_hidden_size, 1, act=ScaledTanh(torch.pi), is_final_layer=True).to(device)
        self.dec_xy = LinBlock(self.decoding_hidden_size, 2, act=nn.Tanh(), is_final_layer=True).to(device)
        self.dec_rgb = LinBlock(self.decoding_hidden_size, 3, act=nn.Sigmoid(), is_final_layer=True).to(device)
        
        # # Mean and log_std decoders
        # self.dec_mu = LinBlock(self.decoding_hidden_size, 9)  # Predicts (l, z, b, a, xy(2), rgb(3))
        # self.dec_log_std = LinBlock(self.decoding_hidden_size, 9)  # Log standard deviation
        # self.log_std_min = -20  # Clamping for numerical stability
        # self.log_std_max = 2

    def forward(self, 
                current_canvas, 
                target_img, 
                target_tokenized_txt, 
                remaining_strokes,
                color_palette,
                canvas_mask=None,
                stroke_mask=None):
        if canvas_mask is not None:
            current_canvas = current_canvas * canvas_mask
        if stroke_mask is not None:
            remaining_strokes = remaining_strokes * stroke_mask
        
        reshape = False
        if len(remaining_strokes.shape)==3:
            batch_size, n_strokes = current_canvas.shape[0], current_canvas.shape[1]
            current_canvas = current_canvas.reshape(batch_size*n_strokes,*current_canvas.shape[2:])
            target_img = target_img.reshape(batch_size*n_strokes,*target_img.shape[2:])
            target_tokenized_txt = target_tokenized_txt.reshape(batch_size*n_strokes,*target_tokenized_txt.shape[2:])
            remaining_strokes = remaining_strokes.reshape(batch_size*n_strokes,*remaining_strokes.shape[2:])
            color_palette = color_palette.reshape(batch_size*n_strokes,*color_palette.shape[2:])
            reshape = True
        
        encoded_inputs = self.state_encoder(current_canvas, target_img, target_tokenized_txt, remaining_strokes, color_palette)

        if reshape:
            encoded_inputs = encoded_inputs.reshape(batch_size, n_strokes, -1)
        elif len(encoded_inputs.shape) == 2:
            encoded_inputs = encoded_inputs.unsqueeze(1) # Add sequence length dim

        out, hidden = self.main(encoded_inputs)

        if reshape:
            x = out.reshape(batch_size*n_strokes, -1)
        else:
            x = out[:, -1, :]  # Take last timestep's output
        # self.last_hidden = hidden[-1] # Save the last layer's hidden state

        l_dec = self.dec_l(x)
        z_dec = self.dec_z(x)
        b_dec = self.dec_b(x)
        a_dec = self.dec_a(x)
        xy_dec = self.dec_xy(x)
        rgb_dec = self.dec_rgb(x)
        
        # Returns [l, z, b, a, x, y, r, g, b]
        output = torch.cat([l_dec, z_dec, b_dec, a_dec, xy_dec, rgb_dec], dim=1)
        if reshape:
            output = output.reshape(batch_size, n_strokes, -1)
        
        if stroke_mask is not None:
            output = output * stroke_mask
        
        return output

        # mu = self.dec_mu(hidden)
        # log_std = self.dec_log_std(hidden)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # return mu, log_std