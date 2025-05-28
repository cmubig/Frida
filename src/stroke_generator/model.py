import clip
import torch
from torch import nn
import torchvision.transforms as transforms

from stroke_generator.utils.model_utils import ConvBlock, LinBlock, ScaledSigmoid, ScaledTanh
from stroke_generator.encoders.encoders import CanvasEncoder
from stroke_generator.encoders.decoder import StrokeDecoder


class StrokePredictor(nn.Module):
    def __init__(self, opt, device='cpu'):
        super().__init__()

        self.opt = opt
        self.device = device

        self.img_size = 224
        self.pallete_size = 12
        self.state_latent_dim = 1024
        self.action_latent_dim = 128

        # Encoder
        self.state_encoder = CanvasEncoder(
            opt, 
            device,
            latent_dim=self.state_latent_dim,
            pallete_size=self.pallete_size
        )

        # Main block
        self.main = nn.LSTM(
            input_size=self.state_latent_dim,
            hidden_size=self.action_latent_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        ).to(device)
        self.skip = nn.Sequential(
            nn.Linear(self.state_latent_dim, self.state_latent_dim),
            nn.ReLU(),
            nn.Linear(self.state_latent_dim, self.action_latent_dim),
        ).to(device)

        # Decoder
        self.stroke_decoder = StrokeDecoder(
            opt,
            device,
            latent_dim=self.action_latent_dim,
            pallete_size=self.pallete_size
        )

        # Translates from [-1,1] to real params
        min = torch.tensor([
            self.opt.MIN_STROKE_LENGTH,
            self.opt.MIN_STROKE_Z,
            -self.opt.MAX_BEND,
            -1,-1, # Ax, Ay
            -1, -1, # X, Y
        ])
        max = torch.tensor([
            self.opt.MAX_STROKE_LENGTH,
            0.95,
            self.opt.MAX_BEND,
            1,1, # Ax, Ay
            1, 1, # X, Y
        ])
        self.scale = (max-min).to(device)
        self.bias = ((max+min)/2).to(device)

        self.save_hx = False
        self.hx = None

    def sample(self, 
                current_canvas, 
                target_img, 
                target_tokenized_txt, 
                remaining_strokes,
                color_palette,
                mask=None):
        
        lzbaxy_mean, lzbaxy_log_std, rgb_logits = self.forward(
            current_canvas, 
            target_img, 
            target_tokenized_txt, 
            remaining_strokes,
            color_palette,
            mask
        )

        lzbaxy_sample = torch.distributions.Normal(lzbaxy_mean, lzbaxy_log_std.exp()).sample()
        # Translate from [-1,1] to real params
        lzbaxy_sample = lzbaxy_sample * self.scale / 2.0 + self.bias 

        # Get angle from a_x and a_y
        a_x, a_y = lzbaxy_sample[...,3], lzbaxy_sample[...,4]
        a = torch.atan2(a_x, a_y).unsqueeze(-1)
        lzbaxy = torch.cat([lzbaxy_sample[...,:3],a,lzbaxy_sample[...,-2:]],dim=-1)

        rgb_idx = torch.distributions.Categorical(logits=rgb_logits).sample()

        # Get color from color palette
        if len(color_palette.shape)==4:
            rgb_idx = rgb_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)      # [B, H, 1, 3]
            colors = torch.gather(color_palette, dim=2, index=rgb_idx).squeeze(2)   # [B, H, 3]
        else:
            rgb_idx = rgb_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)          # [B, 1, 3]
            colors = torch.gather(color_palette, dim=1, index=rgb_idx).squeeze(1)   # [B, 3]
        stroke_tensor = torch.cat([lzbaxy, colors],dim=-1)
        return stroke_tensor
    
    def forward(self, 
                current_canvas, 
                target_img, 
                target_tokenized_txt, 
                remaining_strokes,
                color_palette,
                mask=None):
        
        # If given with shape [batch_size, n_strokes, ...], reshape to [batch_size*n_strokes, ...] for encoder
        reshape = False
        if len(remaining_strokes.shape)==3:
            batch_size, n_strokes = current_canvas.shape[0], current_canvas.shape[1]
            current_canvas = current_canvas.reshape(batch_size*n_strokes,*current_canvas.shape[2:])
            target_img = target_img.reshape(batch_size*n_strokes,*target_img.shape[2:])
            target_tokenized_txt = target_tokenized_txt.reshape(batch_size*n_strokes,*target_tokenized_txt.shape[2:])
            remaining_strokes = remaining_strokes.reshape(batch_size*n_strokes,*remaining_strokes.shape[2:])
            color_palette = color_palette.reshape(batch_size*n_strokes,*color_palette.shape[2:])
            mask = mask.reshape(batch_size*n_strokes,1) if mask is not None else None
            reshape = True
        
        # Encode
        encoded_inputs = self.state_encoder(current_canvas, 
                                            target_img, 
                                            target_tokenized_txt, 
                                            remaining_strokes, 
                                            color_palette, 
                                            mask)
        
        # Skip connection
        skip_x = self.skip(encoded_inputs)
        
        # Reshape back to [batch_size, n_strokes, ...] for LSTM
        # or add sequence length dim of 1 for LSTM if no n_strokes dimensino given
        if reshape:
            encoded_inputs = encoded_inputs.reshape(batch_size, n_strokes, -1)
        else:
            encoded_inputs = encoded_inputs.unsqueeze(1) # Add sequence length dim of 1


        # LSTM
        out, (hidden, cell) = self.main(encoded_inputs, self.hx)
        if self.save_hx:
            self.hx = (hidden, cell)  # Save hidden state for next timestep

        # If given with shape [batch_size, n_strokes, ...], reshape to [batch_size*n_strokes, ...] for decoder
        if reshape:
            x = out.reshape(batch_size*n_strokes, -1)
        else:
            x = out[:, -1, :]  # Take last timestep's output

        # Add skip connection
        x = x + skip_x

        # Decode
        lzbaxy_mu, lzbaxy_log_std, rgb_logits = self.stroke_decoder(x)
        
        
        # Apply mask on the way out if given
        if mask is not None:
            lzbaxy_mu = lzbaxy_mu * mask
            lzbaxy_log_std = lzbaxy_log_std * mask
            rgb_logits = rgb_logits * mask
        
        # Reshape back to original input shape
        if reshape:
            lzbaxy_mu = lzbaxy_mu.reshape(batch_size, n_strokes, -1)
            lzbaxy_log_std = lzbaxy_log_std.reshape(batch_size, n_strokes, -1)
            rgb_logits = rgb_logits.reshape(batch_size, n_strokes, -1)
        
        return lzbaxy_mu, lzbaxy_log_std, rgb_logits