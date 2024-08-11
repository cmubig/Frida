


import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    # StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers import StableDiffusionInstructPix2PixPipeline

import numpy as np
import PIL.Image
import torch

from diffusers.models import AutoencoderKL, UNet2DConditionModel

import matplotlib.pyplot as plt


def add_imgs_together(imgs):
    # Take  list of images and add all the dark parts together
    imgs = np.stack([np.array(img) for img in imgs], axis=0)
    img = np.min(imgs, axis=0)
    return PIL.Image.fromarray(img)


# # controlnet_model_name_or_path = "./controlnet_models/checkpoint-12000_good/controlnet"
# controlnet_model_name_or_path = "./controlnet_models/checkpoint-1500/unet"
# pretrained_model_name_or_path = "timbrooks/instruct-pix2pix"
# weight_dtype = torch.float16
# device = 'cuda'

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

# def get_instruct_pix2pix_model(instruct_pix2pix_path, original_model_name_or_path="timbrooks/instruct-pix2pix", 
#                                device="cuda", weight_dtype=torch.float16):
#     tokenizer = AutoTokenizer.from_pretrained(
#         original_model_name_or_path,
#         subfolder="tokenizer",
#         # revision=args.revision,
#         use_fast=False,
#     )

#     # import correct text encoder class
#     text_encoder_cls = import_model_class_from_model_name_or_path(original_model_name_or_path)

#     # Load scheduler and models
#     noise_scheduler = DDPMScheduler.from_pretrained(original_model_name_or_path, subfolder="scheduler")
#     text_encoder = text_encoder_cls.from_pretrained(
#         original_model_name_or_path, subfolder="text_encoder"
#     )
#     vae = AutoencoderKL.from_pretrained(original_model_name_or_path, subfolder="vae")
#     unet = UNet2DConditionModel.from_pretrained(
#         instruct_pix2pix_path, subfolder="unet", 
#         in_channels=8, out_channels=4,  #local_files_only=True
#     )
#     unet.register_to_config(in_channels=8)

#     pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
#         original_model_name_or_path,
#         unet=unet,
#         text_encoder=text_encoder,
#         vae=vae,
#         # revision=args.revision,
#         torch_dtype=weight_dtype,
#         safety_checker=None
#     )
#     pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
#     pipeline = pipeline.to(device)
#     # pipeline.set_progress_bar_config(disable=True)
#     return pipeline

def get_instruct_pix2pix_model(instruct_pix2pix_path, original_model_name_or_path="timbrooks/instruct-pix2pix",
                               device="cuda", weight_dtype=torch.float16):
    '''
        args:
            lora_weights_path : Either path to local file (parent dir)
                or Huggingface location (e.g., skeeterman/CoFRIDA-Sharpie)
    '''
    # Assuming LoRA was used
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    original_model_name_or_path,
                    torch_dtype=weight_dtype,
                    safety_checker=None,
                )
    pipeline.unet.load_attn_procs(instruct_pix2pix_path)

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


if __name__ == '__main__':
    validation_prompts = [
        'A chicken riding a bicycle',
        'A drawing of a dinosaur wearing a tuxedo',
    ]
    validation_images = [
        "/home/frida/Downloads/current_canvas.jpg",
        "/home/frida/Downloads/rex.jpg"
    ]
    resolution = 512
    c = 0
    n_iters = 2
    n_tries = 2

    def pad_img(img, padding = 5):
        return np.pad(img, ((padding, padding), (padding, padding), (0, 0)),
            mode='constant', constant_values=255)

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")
        # print('validation_image', validation_image.size)
        validation_image = validation_image.resize((resolution, resolution))####
        
        with torch.autocast("cuda"), torch.no_grad():
            in_img = validation_image

            # # No init latent conditioning
            rows = []
            for j in range(n_tries):
                
                in_img = validation_image
                cols = [pad_img(np.array(in_img))]
                for i in range(n_iters):
                    out_img = pipeline(
                        validation_prompt,
                        image=in_img,
                        num_inference_steps=20,
                        image_guidance_scale=1.5,
                        guidance_scale=7,
                        # generator=generator,
                        num_images_per_prompt=1,
                    ).images[0]
                    
                    out_img = add_imgs_together([in_img, out_img])
                    
                    in_img = out_img
                    
                    cols.append(pad_img(np.array(out_img)))
                rows.append(np.concatenate(cols, axis=1))

            whole_img = np.concatenate(rows, axis=0)
            plt.imshow(whole_img)
            plt.title(validation_prompt)
            plt.xticks([])
            plt.yticks([])
            plt.show()