import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers import StableDiffusionControlNetPipeline
from PIL import Image
import numpy as np
import PIL
from PIL import Image
import os

controlnet_model_name_or_path = "controlnet_models/checkpoint-12000_good/controlnet"
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
weight_dtype = torch.float16
device = 'cuda'
rootdir = '../frida-hci/public'

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
text_encoder = text_encoder_cls.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)

controlnet = ControlNetModel.from_pretrained(controlnet_model_name_or_path)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=weight_dtype,
)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(device)
resolution = 512
c = 0

def combine(imgs):
    # Take  list of images and add all the dark parts together
    canvas = torch.tensor(np.array(imgs[0])).permute(2, 0, 1).float()
    blue_canvas = torch.tensor(np.array(imgs[1])).permute(2, 0, 1).float()
    black_mask = (blue_canvas.mean(axis=0) < 100)
    blue_canvas[2][black_mask] = 255

    with torch.no_grad():
        white_mask = (canvas.mean(axis=0) > 180).unsqueeze(0).repeat_interleave(3, dim=0)
        black_mask = black_mask.unsqueeze(0).repeat_interleave(3, dim=0)
        masks = torch.logical_and(white_mask, black_mask)
        canvas[masks] = blue_canvas[masks]
                        
    return PIL.Image.fromarray(canvas.permute(1, 2, 0).byte().numpy())

def api_manifest(prompt, iter):
    in_img = Image.open(f'{rootdir}/data/camera.jpg').convert("RGB")
    in_img = in_img.resize((resolution, resolution))

    with torch.autocast('cuda'), torch.no_grad():
        out_img = pipeline(
            'line drawing of ' + prompt + ', else empty, black and white', 
            in_img, num_inference_steps=20, 
            num_images_per_prompt=1,
            # controlnet_conditioning_scale=1.1
        ).images[0]
        out_img = combine([in_img, out_img])
        out_img.save(f'{rootdir}/data/stable/{iter}.jpg')
    return

def api_adapt(task, is_first, counter, prompt):
    import numpy as np
    import torch
    import time
    import sys
    import os
    sys.path.append('src/')

    from options_hci import Options

    global opt
    opt = Options()
    opt.gather_options()

    global writer
    writer = None
    ### INK OPTIONS ###
    device_adapt = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # if prompt:
    #     opt.objective = ['clip_fc_loss', 'text']
    #     opt.objective_data = [f'public/data/objectives/{task}.jpg', f"{prompt}"]
    #     opt.objective_weight = [0.2, 1.0]
    # else:
    opt.objective = ['l2']
    opt.objective_data = [f'{rootdir}/data/objectives/{task}.jpg']
    opt.strokes_before_adapting = min(counter, opt.adapt_num_strokes) if counter > 0 else 0
    ###

    from paint_utils3 import load_img
    from plan_hci import load_objectives_data, adapt, plan
    from torchvision.utils import save_image

    writer = None
    local_it = 0 
    plans = []

    stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
    h, w = stroke_shape[0], stroke_shape[1]
    w = int((opt.max_height/h)*w)
    h = int(opt.max_height)

    colors = None

    # Get the background of painting to be the current canvas
    current_canvas = load_img(f'{rootdir}/data/camera.jpg', h=h, w=w).to(device_adapt)/255.0
    load_objectives_data(opt)

    # Start Planning
    plan_f = os.path.join(opt.cache_dir, f"[{task}]_next_brush_strokes.csv")
    if os.path.exists(plan_f):
        if not is_first and counter != 0:
            plan_f = os.path.join(opt.cache_dir, f"next_brush_strokes.csv")
        painting = adapt(plan_f, opt, current_canvas, colors, h, w)
    else:
        painting = plan(opt, current_canvas, colors, h, w)
        painting.sort_strokes()
        f = open(plan_f, "w")
        f.write(painting.to_csv())
        f.close()
    f = open(os.path.join(opt.cache_dir, f"next_brush_strokes.csv"), "w")
    f.write(painting.to_csv())
    f.close()

    images = []
    with torch.no_grad():
        for stroke in range(min(opt.adapt_num_strokes, len(painting.brush_strokes))):
            p, _ = painting(h, w, use_alpha=False, return_alphas=True, num_strokes=stroke + 1, make_blue=True)
            images.append(p.to('cpu'))
        p, _ = painting(h, w, use_alpha=False, return_alphas=True, make_blue=True)
        from torchvision.utils import save_image
        save_image(p, "current_plan.jpg", quality=95)
    return images

    #   with torch.no_grad():
    #         save_image(painting(h*4,w*4, use_alpha=False), 'public/data/camera.jpg')

    # Export the strokes
    # f = open(os.path.join(opt.cache_dir, "next_brush_strokes.csv"), "w")
    # f.write(painting.to_csv())
    # f.close()