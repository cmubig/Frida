import clip
import torch
from torch import nn
import torchvision.transforms as transforms

from stroke_generator.utils.model_utils import ConvBlock, LinBlock, ScaledSigmoid, ScaledTanh

class CanvasEncoder(nn.Module):
    def __init__(self, opt, device='cpu'):
        super().__init__()

        self.opt = opt
        self.device = device

        self.img_size = 224
        self.max_pallete_size = 12
        self.encoding_hidden_size = 512

        ################
        ### Encoders ###
        ################
        self.current_canvas_encoder = ImageEncoder(opt, self.img_size, self.encoding_hidden_size, device)
        self.target_image_encoder = ImageEncoder(opt, self.img_size, self.encoding_hidden_size, device)

        self.clip_model, _ = clip.load('ViT-B/32', device, jit=False)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.remaining_stroke_encoder = nn.Sequential(
            LinBlock(1, self.encoding_hidden_size),
        ).to(device)
        self.color_palette_encoder = nn.Sequential(
            LinBlock(self.max_pallete_size*3, self.encoding_hidden_size),
        ).to(device)

    def forward(self, current_canvas, target_img, target_tokenized_txt, remaining_strokes, color_palette):
        # Encode inputs #
        canv_enc = self.current_canvas_encoder(current_canvas)
        image_enc = self.target_image_encoder(target_img)
        with torch.no_grad():
            target_txt_enc = self.clip_model.encode_text(target_tokenized_txt).to(torch.float)
        remaining_strokes_enc = self.remaining_stroke_encoder(remaining_strokes)
        color_palette_enc = self.color_palette_encoder(color_palette.flatten(start_dim=1))

        encoded_inputs = torch.cat([
            canv_enc,
            image_enc, 
            target_txt_enc,
            remaining_strokes_enc,
            color_palette_enc
        ], dim=-1)

        return encoded_inputs

class ImageEncoder(nn.Module):
    def __init__(self, opt, in_size, out_size, device='cpu'):
        super().__init__()

        self.opt = opt
        self.device = device
        
        self.transform_img = transforms.Compose([
            PadToSquare(),
            transforms.Resize(in_size, antialias=True),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.layer_1 = nn.Sequential(
            ConvBlock(3, 32, act=nn.LeakyReLU(), downsample=True),
            ConvBlock(32, 64, act=nn.LeakyReLU(), downsample=False),
        ).to(device)
        self.shortcut = nn.Sequential(
            ConvBlock(3, 64, act=None, downsample=True),
        ).to(device)
        self.layer_2 = nn.Sequential(
            ConvBlock(64, 64, downsample=False),
        ).to(device)

        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, 64)),
            nn.Flatten(),
            LinBlock(64*64**2, out_size), # in_size/2**N where N=number of downsample layers
        ).to(device)

    def forward(self, image):
        with torch.no_grad():
            image = self.transform_img(image)
        x = self.layer_1(image) + self.shortcut(image)
        x = self.layer_2(x)
        x = self.out(x)
        return x

class PadToSquare:
    def __call__(self, img):
        h, w = img.shape[-2], img.shape[-1]
        max_dim = max(w, h)
        padding = (
            (max_dim - w) // 2,  # left
            (max_dim - h) // 2,  # top
            (max_dim - w + 1) // 2,  # right
            (max_dim - h + 1) // 2   # bottom
        )
        with torch.no_grad():
            padded_img = transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
        return padded_img
    
class StrokeEncoder(nn.Module):
    def __init__(self, opt, device='cpu'):
        super().__init__()

        self.opt = opt
        self.device = device

        self.stroke_size = 9
        self.encoding_hidden_size = 512

        self.stroke_embedding = nn.Sequential(
            LinBlock(self.stroke_size, 64),
            LinBlock(64, self.encoding_hidden_size, is_final_layer=True),
        ).to(device)
    
    def forward(self, strokes):
        return self.stroke_embedding(strokes)