import torch
import torch.nn as nn
from torchvision import models

class ResNet50Encoder(nn.Module):
    def __init__(self, output_dim=512, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        
        # Load the pretrained ResNet50 model
        resnet = models.resnet50(pretrained=pretrained)
        for param in resnet.layer1.parameters():
            param.requires_grad = False
        for param in resnet.layer2.parameters():
            param.requires_grad = False
        for param in resnet.layer3.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        # Remove the final fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # output shape: [B, 2048, 1, 1]

        # Flatten and project to desired output_dim
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(2048),
            nn.Linear(2048, output_dim),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = self.projection(x)
        return x