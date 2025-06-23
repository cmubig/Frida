import torch
import torch.nn as nn
from torchvision import models

class ResNet50Encoder(nn.Module):
    def __init__(self, output_dim=512, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        
        # Load the pretrained ResNet50 model
        resnet = models.resnet50(pretrained=pretrained)

        # Freeze layers up to layer3
        # Get layer4 and AdaptiveAvgPool2d for trainable layers
        self.resnet_frozen = nn.Sequential(*list(resnet.children())[:-3])
        self.resnet_trainable = nn.Sequential(
            resnet.layer4,
            resnet.avgpool
        )

        # Flatten and project to desired output_dim
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(2048),
            nn.Linear(2048, output_dim),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet_frozen(x)
        x = self.resnet_trainable(x)
        x = self.projection(x)
        return x