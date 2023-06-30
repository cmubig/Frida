
import argparse
import copy
import logging
import math
import os
import random
from pathlib import Path

# import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    # StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers import StableDiffusionControlNetPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from controlnet_dataset import FridaControlNetDataset

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from torch import nn
# from torch import nn
# from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel

# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import StableDiffusionPipelineOutput

import matplotlib.pyplot as plt

from diffusers.models.controlnet import zero_module


def add_imgs_together(imgs):
    # Take  list of images and add all the dark parts together
    imgs = np.stack([np.array(img) for img in imgs], axis=0)
    img = np.min(imgs, axis=0)
    return PIL.Image.fromarray(img)


controlnet_model_name_or_path = "./controlnet_models/checkpoint-12000_good/controlnet"
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
weight_dtype = torch.float16
device = 'cuda'

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision=None):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        # revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    # revision=args.revision,
    use_fast=False,
)

# import correct text encoder class
text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path)

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)

controlnet = ControlNetModel.from_pretrained(controlnet_model_name_or_path)
# if controlnet_model_name_or_path:
#     controlnet = ControlNetModel.from_pretrained(controlnet_model_name_or_path)
# else:
#     # controlnet = ControlNetModel.from_unet(unet)
#     #################################
#     t = 1
#     controlnet = ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(16*t, 32*t, 96*t, 256*t))
#     ###########################################


pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=controlnet,
    safety_checker=None,
    # revision=args.revision,
    torch_dtype=weight_dtype,
)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(device)
# pipeline.set_progress_bar_config(disable=True)

validation_prompts = [
    'A chicken riding a bicycle',
    # 'A drawing of a dinosaur wearing a tuxedo',
]
validation_images = [
    "/home/frida/Downloads/current_canvas.jpg",
    # "/home/frida/Downloads/rex.jpg"
]
resolution = 512
c = 0
n_iters = 1
n_tries = 4

for validation_prompt, validation_image in zip(validation_prompts, validation_images):
    validation_image = Image.open(validation_image).convert("RGB")
    # print('validation_image', validation_image.size)
    validation_image = validation_image.resize((resolution, resolution))####
    
    images = []

    fig, ax = plt.subplots(n_tries,n_iters+1)

    with torch.autocast("cuda"), torch.no_grad():
        in_img = validation_image

        # # No init latent conditioning
        for j in range(n_tries):
            
            ax[j,0].imshow(validation_image)
            ax[j,0].set_xticks([])
            ax[j,0].set_yticks([])

            if j > 0: images.append(validation_image)
            in_img = validation_image
            for i in range(n_iters):
                out_img = pipeline(
                # out_img = new_call(pipeline,
                    validation_prompt, in_img, num_inference_steps=20, 
                    # generator=generator,
                    num_images_per_prompt=1,
                    # controlnet_conditioning_scale=1.4,
                ).images[0]
                # out_img = in_img
                # images.append(out_img)
                # images.append(add_imgs_together([in_img, out_img]))
                out_img = add_imgs_together([in_img, out_img])
                in_img = out_img
                ax[j, i+1].imshow(out_img)
                ax[j, i+1].set_xticks([])
                ax[j, i+1].set_yticks([])
        plt.show()