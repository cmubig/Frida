import argparse
import math
import os
from librosa.core import audio

import torch
import torchvision
from torch import optim
from tqdm import tqdm
import sys
import librosa
import numpy as np
import random
import torch.nn.functional as F

import cv2

from audio_loss.criteria.id_loss import IDLoss
from audio_loss.models.stylegan2.model import Generator
from audio_loss.criteria.soundclip_loss import SoundCLIPLoss

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def compute_audio_loss(args, img_orig, img_gen): 

    y, sr = librosa.load(args.audio_path, sr=44100)
    n_mels = 128
    time_length = 864
    audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1

    audio_inputs = audio_inputs


    zero = np.zeros((n_mels, time_length))
    resize_resolution = 256
    h, w = audio_inputs.shape                                         
    if w >= time_length:
        j = 0
        j = random.randint(0, w-time_length)
        audio_inputs = audio_inputs[:,j:j+time_length]
    else:
        zero[:,:w] = audio_inputs[:,:w]
        audio_inputs = zero

    
    audio_inputs = cv2.resize(audio_inputs, (n_mels, resize_resolution))
    audio_inputs = np.array([audio_inputs])
    audio_inputs = torch.from_numpy(audio_inputs.reshape((1, 1, n_mels, resize_resolution))).float().cuda()


    # g_ema = Generator(args.stylegan_size, 512, 8)
    # g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    # g_ema.eval()
    # g_ema = g_ema.cuda()
    # mean_latent = g_ema.mean_latent(4096)
    # layer_masking_weight = torch.ones(14)

    # latent_code_init_not_trunc = torch.randn(1, 512).cuda()
    # with torch.no_grad():
    #     img_orig, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
    #                                 truncation=args.truncation, truncation_latent=mean_latent)

    # print('-----------------IMG_ORIG-----------------')
    # print(type(img_orig))
    # print(img_orig.shape)

    # latent = latent_code_init.detach().clone()
    # latent.requires_grad = True

    soundclip_loss = SoundCLIPLoss(args)
    id_loss = IDLoss(args)

    # img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)

    # img_gen = torchvision.transforms.Grayscale()(img_gen)
    # img_gen = torchvision.transforms.Resize((resize_resolution, n_mels))(img_gen)
    img_gen = torchvision.transforms.Resize((resize_resolution, resize_resolution))(img_gen)

    # img_orig = torchvision.transforms.Grayscale()(img_orig)
    # img_orig = torchvision.transforms.Resize((resize_resolution, n_mels))(img_orig)
    img_orig = torchvision.transforms.Resize((resize_resolution, resize_resolution))(img_orig)

    # print('------------Img_gen------------')
    # print(type(img_gen))
    # print(img_gen.shape)

    # print('------------Audio------------')
    # print(type(audio_inputs))
    # print(audio_inputs.shape)


    cosine_distance_loss = soundclip_loss(img_gen, audio_inputs)

    

    # similarity_loss = 0
    # for idx in range(14):
    #     layer_per_loss = F.sigmoid(layer_masking_weight[idx]) * ((latent_code_init[:,idx,:] - latent[:,idx,:]) ** 2).sum()
    #     similarity_loss += layer_per_loss
    #     layer_masking_weight[idx] = layer_masking_weight[idx] - 0.1 * layer_per_loss.item() * (1 - layer_per_loss.item())

    # loss = cosine_distance_loss  + args.lambda_identity * id_loss(img_orig, img_gen)[0]



    return cosine_distance_loss
    




