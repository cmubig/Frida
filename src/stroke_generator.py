import glob
import json
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode 
bicubic = InterpolationMode.BICUBIC
from torch import nn

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import gzip
from torchvision.transforms.functional import affine

import open_clip
  

class StrokeGenerator(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.encoding_hidden_size = 20
        self.image_channels = 3
        self.clip_size = 224 

        self.main = nn.Sequential(
            LinBlock(1+1024*3, 1024),
            LinBlock(1024, 512),
            LinBlock(512, 9),
        ).to(device)

        # Clip encoders
        self.open_clip_model, _, self.open_clip_preprocess = open_clip.create_model_and_transforms('ViT-g-14', 
                                                                                 pretrained='laion2b_s34b_b88k')
        self.open_clip_tokenizer = open_clip.get_tokenizer('ViT-g-14')
        self.open_clip_model.to(device)

    def forward(self, current_canvas, target_img, target_txt, remaining_strokes, color_palette): 
        # Encode canvas
        cropped_canvas = center_square_crop_tensor(current_canvas, self.clip_size)
        enc_cnvs = self.open_clip_model.encode_image(cropped_canvas)
        # enc_cnvs.shape: [batch,1024]

        # Encode target img
        cropped_img = center_square_crop_tensor(target_img, self.clip_size)
        enc_img = self.open_clip_model.encode_image(cropped_img)
        # enc_img.shape: [batch,1024]

        # Encode text
        tokenized_txt = self.open_clip_tokenizer([target_txt]).to(enc_img.device)
        enc_txt = self.open_clip_model.encode_text(tokenized_txt)
        # enc_txt.shape: [batch,1024]

        # TODO: Encode color palette
        # Not implemented

        encoded_inputs = [
            remaining_strokes*torch.ones(1,1).to(enc_cnvs.device), 
            enc_cnvs, 
            enc_img, 
            enc_txt
        ]

        # Returns [l, z, b, x, y, a, r, g, b] 
        return self.main(torch.concatenate(encoded_inputs,dim=1))

class LinBlock(nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_,out_),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(out_),
            nn.Dropout(0.2),
        )
    
    def forward(self,x):
        return self.main(x)

def center_square_crop_tensor(tensor, size):
    t_c = [tensor.shape[-2]//2, tensor.shape[-1]//2]
    h_s = size//2
    cropped_canvas = tensor[:,:,t_c[0]-h_s:t_c[0]+h_s,t_c[1]-h_s:t_c[1]+h_s]
    return cropped_canvas