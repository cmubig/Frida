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

from losses.audio_loss.criteria.id_loss import IDLoss
from losses.audio_loss.models.stylegan2.model import Generator
from losses.audio_loss.criteria.soundclip_loss import audio_loss

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def load_audio_file(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
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
    return audio_inputs

def compute_audio_loss(audio_inputs, painting): 
    resize_resolution = 256
    painting = torchvision.transforms.Resize((resize_resolution, resize_resolution))(painting)

    cosine_distance_loss = audio_loss(painting, audio_inputs)


    return cosine_distance_loss
    




