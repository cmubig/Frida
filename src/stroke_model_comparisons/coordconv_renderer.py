from torch import nn
from torch.nn import functional as F
import torch

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels+2,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
    def forward(self, x):
        n, _, height, width = x.shape
        idxs_y = torch.arange(height) / (height-1)
        idxs_x = torch.arange(width) / (width-1)
        y_coords, x_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij')
        grid_coords = torch.stack((y_coords, x_coords), dim=0).reshape(1,2,height,width)
        grid_coords = grid_coords.repeat(n,1,1,1)
        grid_coords = grid_coords.to(x.device)
        x = torch.cat((x, grid_coords), dim=1)
        x = self.conv(x)
        return x

class CoordConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(CoordConvTranspose, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels+2,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       output_padding=output_padding)

    def forward(self, x):
        n, _, height, width = x.shape
        idxs_y = torch.arange(height) / (height-1)
        idxs_x = torch.arange(width) / (width-1)
        y_coords, x_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij')
        grid_coords = torch.stack((y_coords, x_coords), dim=0).reshape(1,2,height,width)
        grid_coords = grid_coords.repeat(n,1,1,1)
        grid_coords = grid_coords.to(x.device)
        x = torch.cat((x, grid_coords), dim=1)
        x = self.conv(x)
        return x

class CoordConvRenderer(nn.Module):
    def __init__(self):
        super(CoordConvRenderer, self).__init__()
        
        self.conv_hidden_dims = [128, 64, 32, 16, 8, 8, 8]
        
        self.fc = nn.Sequential(
            nn.Linear(64, self.conv_hidden_dims[0]*2*2),
            nn.LeakyReLU()
        )
        
        decoder_layers = []
        for i in range(len(self.conv_hidden_dims) - 1):
            deconv_block = nn.Sequential(
                CoordConvTranspose(self.conv_hidden_dims[i],
                                   self.conv_hidden_dims[i + 1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(self.conv_hidden_dims[i + 1]),
                nn.LeakyReLU()
            )
            decoder_layers.append(deconv_block)
        decoder_layers.append(nn.Sequential(
            CoordConvTranspose(self.conv_hidden_dims[-1],
                               self.conv_hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.conv_hidden_dims[-1]),
            nn.LeakyReLU(),
            CoordConv(self.conv_hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Sigmoid()
        ))
        self.conv = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(-1, self.conv_hidden_dims[0], 2, 2)
        x = self.conv(x)
        x = torch.squeeze(x, dim=1)
        return x
