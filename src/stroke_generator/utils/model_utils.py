import torch
import torch.nn as nn

class LinBlock(nn.Module):
    def __init__(self, in_, out_, act=None, is_final_layer=False, use_dropout=True):
        super().__init__()

        if act is None: act = nn.LeakyReLU(0.2, inplace=True)

        layers = [nn.Linear(in_,out_), act]

        if not is_final_layer:
            layers.append(nn.BatchNorm1d(out_))
            if use_dropout:
                layers.append(nn.Dropout(0.2))
        
        self.main = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.main(x)

class ConvBlock(nn.Module):
    def __init__(self, in_, out_, act=None, downsample=True):
        super().__init__()
        if act is None:
            act = nn.LeakyReLU(0.2, inplace=True)

        layers = [nn.Conv2d(in_, out_, 3, 1, 1), nn.BatchNorm2d(out_), act]
        if downsample:
            layers.append(nn.MaxPool2d(2, 2))  # Downsampling

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ScaledTanh(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.act = nn.Tanh()
        self.scale = scale

    def forward(self, x):
        return self.scale * self.act(x)

class ScaledSigmoid(nn.Module):
    def __init__(self, max=1.0, min=0.0):
        super().__init__()
        self.act = nn.Sigmoid()
        self.max = max
        self.min = min
    
    def forward(self, x):
        return (self.max - self.min) * self.act(x) + self.min