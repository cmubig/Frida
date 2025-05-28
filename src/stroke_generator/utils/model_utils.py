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
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.2, act=None):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act
        self.dropout1 = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.act1 is not None:
            x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.act2 is not None:
            x = self.act2(x)
        x = self.dropout1(x)
        return x

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