import clip
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.models as models

from stroke_generator.utils.model_utils import ConvBlock, LinBlock
from stroke_generator.encoders.resnet50Encoder import ResNet50Encoder

from brush_stroke import BrushStrokeBatch
from painting import PaintingBatch

class CanvasEncoder(nn.Module):
    def __init__(self, opt, device='cpu', latent_dim=2048, pallete_size=12):
        super().__init__()

        self.opt = opt
        self.device = device

        self.latent_dim = latent_dim
        self.latent_size_ratios = [0.35,0.35,0.1,0.1,0.1]
        self.latent_sizes = {
            'current_canvas': int(self.latent_dim*self.latent_size_ratios[0]),
            'target_image': int(self.latent_dim*self.latent_size_ratios[1]),
            'remaining_strokes': int(self.latent_dim*self.latent_size_ratios[2]),
            'color_palette': int(self.latent_dim*self.latent_size_ratios[3]),
            'text': int(self.latent_dim*self.latent_size_ratios[4]),
        }
        error = latent_dim - sum(self.latent_sizes.values())
        if error > 0:
            self.latent_sizes['remaining_strokes'] += error
        
        self.pallete_size = pallete_size
        self.img_size = 224

        ################
        ### Encoders ###
        ################
        self.current_canvas_encoder = ResNet50Encoder(self.latent_sizes['current_canvas']).to(device)
        self.target_image_encoder = ResNet50Encoder(self.latent_sizes['current_canvas']).to(device)

        self.clip_model, _ = clip.load('ViT-B/32', device, jit=False)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_text_encoder = nn.Sequential(
            LinBlock(512, self.latent_sizes['text']),
        ).to(device)
        self.remaining_stroke_encoder = nn.Sequential(
            LinBlock(1, self.latent_sizes['remaining_strokes']),
        ).to(device)
        self.color_palette_encoder = nn.Sequential(
            LinBlock(self.pallete_size*3, self.latent_sizes['color_palette']),
        ).to(device)

    def forward(self, current_canvas, target_img, target_tokenized_txt, remaining_strokes, color_palette, mask=None):
        # Encode inputs #
        canv_enc = self.current_canvas_encoder(current_canvas)
        image_enc = self.target_image_encoder(target_img)
        with torch.no_grad():
            clip_text_enc = self.clip_model.encode_text(target_tokenized_txt).to(torch.float)
        target_txt_enc = self.clip_text_encoder(clip_text_enc)
        remaining_strokes_enc = self.remaining_stroke_encoder(remaining_strokes)
        color_palette_enc = self.color_palette_encoder(color_palette.flatten(start_dim=1))

        encoded_inputs = torch.cat([
            canv_enc,
            image_enc, 
            target_txt_enc,
            remaining_strokes_enc,
            color_palette_enc
        ], dim=-1)

        if mask is not None:
            encoded_inputs *= mask

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
            ConvBlock(64, 64, act=nn.LeakyReLU(), downsample=False),
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
    def __init__(self, opt, device='cpu', hidden_size=1024):
        super().__init__()

        self.opt = opt
        self.device = device

        self.stroke_size = 9
        self.hidden_size = hidden_size
        self.canvas_latent_size = int(hidden_size*0.75)
        self.stroke_latent_size = hidden_size - self.canvas_latent_size

        self.img_size = 224
        self.canv_h = int(opt.render_height)
        self.canv_w = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

        self.image_encoder = ImageEncoder(opt, self.img_size, self.canvas_latent_size, device)
        self.stroke_encoder = nn.Sequential(
            LinBlock(self.stroke_size, 64),
            LinBlock(64, 128),
            LinBlock(128, self.stroke_latent_size, is_final_layer=True),
        ).to(device)
    
    def paint_strokes(self, strokes):
        action = strokes.unsqueeze(-1)
        stroke = BrushStrokeBatch(
            self.opt,
            stroke_length=action[:,0],
            stroke_z=action[:,1],
            stroke_bend=action[:,2],
            stroke_alpha=torch.zeros(action.shape[0],1).to(self.device),
            color=action[:,6:].squeeze(-1),
            a=action[:,3],
            xt=action[:,4],
            yt=action[:,5],
            init_differentiably=True,
            ink=None
        )

        blank_canv = torch.ones((strokes.shape[0],3,self.canv_h,self.canv_w)).to(self.device)

        painting = PaintingBatch(self.opt, background_img=blank_canv).to(self.device)
        painted_canv = painting([stroke], self.canv_h, self.canv_w, use_alpha=False, return_alphas=False)

        return painted_canv

    
    def forward(self, strokes):
        painted_strokes = self.paint_strokes(strokes)
        encoded_paint = self.image_encoder(painted_strokes)
        encoded_strokes = self.stroke_encoder(strokes)
        return torch.cat([encoded_paint, encoded_strokes], dim=-1)