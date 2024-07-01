import torch
from torch import nn
from custom_renderer import CustomRenderer

class CustomUnetRenderer(nn.Module):
    def __init__(self):
        super(CustomUnetRenderer, self).__init__()

        self.custom_renderer = CustomRenderer(256)

        # Down part
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # B x 32 x 256 x 256
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2) # B x 32 x 128 x 128

        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # B x 64 x 128 x 128
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2) # B x 64 x 64 x 64

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ) # B x 128 x 64 x 64
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2) # B x 128 x 32 x 32

        # Up part
        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) # B x 256 x 32 x 32
        self.upconv_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # B x 128 x 64 x 64

        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ) # B x 128 x 64 x 64
        self.upconv_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # B x 64 x 128 x 128

        self.conv_6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # B x 64 x 128 x 128
        self.upconv_3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # B x 32 x 256 x 256

        self.conv_7 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # B x 32 x 256 x 256

        # output layers
        self.conv_8 = nn.Conv2d(32, 1, kernel_size=1, stride=1) # B x 1 x 256 x 256
        self.zero_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        nn.init.constant_(self.zero_conv.weight, 0)
        nn.init.constant_(self.zero_conv.bias, 0)
        self.output_activation = nn.Tanh()

    def forward(self, traj):
        # traj: B x n x 2
        darknesses = self.custom_renderer(traj) # B x 256 x 256
        darknesses = darknesses.unsqueeze(1) # B x 1 x 256 x 256

        # unet
        x = self.conv_1(darknesses) # B x 32 x 256 x 256
        x = self.pool_1(x) # B x 32 x 128 x 128
        x = self.conv_2(x) # B x 64 x 128 x 128
        x = self.pool_2(x) # B x 64 x 64 x 64
        x = self.conv_3(x) # B x 128 x 64 x 64
        x = self.pool_3(x) # B x 128 x 32 x 32

        # up
        x = self.conv_4(x) # B x 256 x 32 x 32
        x = self.upconv_1(x) # B x 128 x 64 x 64
        x = torch.cat([x, x], dim=1) # B x 256 x 64 x 64
        x = self.conv_5(x) # B x 128 x 64 x 64
        x = self.upconv_2(x) # B x 64 x 128 x 128
        x = torch.cat([x, x], dim=1) # B x 128 x 128 x 128
        x = self.conv_6(x) # B x 64 x 128 x 128
        x = self.upconv_3(x) # B x 32 x 256 x 256
        x = torch.cat([x, x], dim=1) # B x 64 x 256 x 256
        x = self.conv_7(x) # B x 32 x 256 x 256

        # output
        x = self.conv_8(x) # B x 1 x 256 x 256
        x = self.zero_conv(x) # B x 1 x 256 x 256

        x = self.output_activation(x)
        x = darknesses + x
        x = torch.clamp(x, min=0, max=1)
        x = x.squeeze(1)

        return x
