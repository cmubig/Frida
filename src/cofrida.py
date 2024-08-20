import os
import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from tqdm.auto import tqdm

from diffusers import StableDiffusionInstructPix2PixPipeline

import numpy as np
import PIL.Image
import torch
import matplotlib.pyplot as plt
<<<<<<< HEAD
import traceback
=======

>>>>>>> ffd451483ca3dcd3e79e4fd838237c30c732a213

def add_imgs_together(imgs):
    # Take  list of images and add all the dark parts together
    imgs = np.stack([np.array(img) for img in imgs], axis=0)
    img = np.min(imgs, axis=0)
    return PIL.Image.fromarray(img)


def get_instruct_pix2pix_model(lora_weights_path, original_model_name_or_path="timbrooks/instruct-pix2pix", 
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
    
    model_loaded = False
    try:
<<<<<<< HEAD
        print(f"loading lora_weights: {lora_weights_path}")
=======
>>>>>>> ffd451483ca3dcd3e79e4fd838237c30c732a213
        pipeline.unet.load_attn_procs(lora_weights_path)
        print('Loaded LoRA')
        model_loaded = True
    except:
        pass

    try:
        pipeline.unet = pipeline.unet.from_pretrained(
            lora_weights_path, 
            torch_dtype=weight_dtype,
            safetensors=True
        )
        print('Loaded Unet Params. No LoRA')
        model_loaded = True 
<<<<<<< HEAD
    except Exception as e:
        # print(e)
        # traceback.print_exc()
=======
    except:
>>>>>>> ffd451483ca3dcd3e79e4fd838237c30c732a213
        pass 

    if not model_loaded:
        print('Could not load model: ', lora_weights_path)
        raise Exception

    pipeline = pipeline.to(device)
    # pipeline.set_progress_bar_config(disable=True)
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