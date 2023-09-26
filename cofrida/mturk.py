import torch 
import os
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

# from test_controlnet import pipeline as controlnet
from test_instruct_pix2pix import get_instruct_pix2pix_model


from options import Options
from painting import *
# from stable_diffusion.stable_diffusion_loss2 import stable_diffusion_loss, encode_text_stable_diffusion
from clip_loss import clip_conv_loss, clip_model, clip_text_loss, clip


from paint_utils3 import *

# python3 mturk.py --abstract_planner_model ./controlnet_models_35_70//checkpoint-2000/unet --codraw_eval_setting same_text_add_detail --codraw_metric_data_dir ./codraw_metric_data/new_amt_test/ --use_cache --cache_dir caches/cache_6_6_cvpr/ --ink --max_stroke_length 0.025 --optim_iter 400 --lr_multiplier 0.7 --min_strokes_added 35 --max_strokes_added 105


def plan_from_image(opt, target_img, current_canvas, stroke_batch_size,
                    text=None, text_and_image=False, text_loss='clip'):
    global colors

    # stroke_batch_size = opt.num_strokes#100#64
    iters_per_batch =  opt.optim_iter#100

    if text is None:
        painting = initialize_painting(0, target_img, current_canvas, opt.ink)
    else:
        # painting = Painting(stroke_batch_size,background_img=current_canvas)
        painting = random_init_painting(current_canvas, stroke_batch_size, ink=opt.ink)
        
        with torch.no_grad():
            if text_loss=='clip':
                text_features = clip_model.encode_text(clip.tokenize(text).to(device))
            else:
                # sd_encoded_text = encode_text_stable_diffusion(text)
                sds_loss.embed_text(text)
    painting.to(device)

    c = 0
    total_its = (opt.num_strokes/stroke_batch_size)*iters_per_batch
    for i in range(1):#(range(0, opt.num_strokes, stroke_batch_size)):#, desc="Initializing"):
        with torch.no_grad():
            p = painting(h,w)
        if text is None:
            painting = add_strokes_to_painting(painting, p[:,:3], stroke_batch_size, target_img, current_canvas, opt.ink)
        optims = painting.get_optimizers(multiplier=opt.lr_multiplier, ink=opt.ink)

        # Learning rate scheduling. Start low, middle high, end low
        og_lrs = [o.param_groups[0]['lr'] if o is not None else None for o in optims]

        # for it in tqdm(range(iters_per_batch), desc="Optim. {} Strokes".format(len(painting.brush_strokes))):
        for it in tqdm(range(iters_per_batch)):
            for o in optims: o.zero_grad() if o is not None else None
            

            lr_factor = (1 - 2*np.abs(it/iters_per_batch - 0.5)) + 0.05
            for i_o in range(len(optims)):
                if optims[i_o] is not None:
                    optims[i_o].param_groups[0]['lr'] = og_lrs[i_o]*lr_factor

            p, alphas = painting(h, w, use_alpha=True, return_alphas=True)

            t = c / total_its
            c+=1 
            
            loss = 0
            if text_and_image:
                loss += clip_conv_loss(target_img, p[:,:3]) * (1 - (it/iters_per_batch))
                if text_loss == 'clip':
                    loss += clip_text_loss(p[:,:3], text_features, num_augs=4)[0] * (it/iters_per_batch)
                else:
                    # loss += stable_diffusion_loss(p[:,:3], sd_encoded_text) * (it/iters_per_batch)
                    loss += sds_loss(p[:,:3]) * (it/iters_per_batch)
            else:
                if text is None:
                    loss += clip_conv_loss(target_img, p[:,:3]) 
                else:
                    if text_loss == 'clip':
                        loss += clip_text_loss(p[:,:3], text_features, num_augs=4)
                    else:
                        loss += sds_loss(p[:,:3])

            loss.backward()

            for o in optims: o.step() if o is not None else None
            painting.validate()
            # painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            # for o in optims: o.param_groups[0]['lr'] = o.param_groups[0]['lr'] * 0.95 if o is not None else None
            # if not opt.ink:
            #     painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            # make sure hidden strokes get some attention
            # painting = randomize_brush_stroke_order(painting)

            if (it % 10 == 0 and it > (0.5*iters_per_batch)) or it > 0.9*iters_per_batch:
                if opt.use_colors_from is None:
                    # Cluster the colors from the existing painting
                    if not opt.ink:
                        if opt.colors is None:
                            colors = painting.cluster_colors(opt.n_colors)
                        else:
                            colors = np.array([i.split(',') for i in opt.colors.split('.')]).astype(np.float32)
                            colors = (torch.from_numpy(colors) / 255.).to(device)
                if not opt.ink:
                    discretize_colors(painting, colors)

    # if opt.use_colors_from is None:
    #     colors = painting.cluster_colors(opt.n_colors)
    # with open(os.path.join(opt.cache_dir, 'colors_updated.npy'), 'wb') as f:
    #     np.save(f, (colors.detach().cpu().numpy()*255).astype(np.uint8))

    # discretize_colors(painting, colors)
    # if opt.ink:
    #     painting = sort_brush_strokes_by_location(painting, bin_size=opt.bin_size)

    return painting

global opt
opt = Options()
opt.gather_options()
matplotlib.use('TkAgg')
global h, w, colors, current_canvas, text_features, style_img, sketch
stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
h, w = stroke_shape[0], stroke_shape[1]
w = int((opt.max_height/h)*w)
h = int(opt.max_height)



output_dir = opt.codraw_metric_data_dir
eval_setting = opt.codraw_eval_setting

df_fn = os.path.join(output_dir, 'codraw_metric_data.csv')
df_fn_backup = os.path.join(output_dir, 'codraw_metric_data_saved.csv')



if not os.path.exists(output_dir): os.mkdir(output_dir)

if os.path.exists(df_fn):
    df = pd.read_csv(df_fn)
else:
    df = pd.DataFrame(columns=['text_start', 'text_next', 
                               'x_start', 'x_next', 
                               'x_start_drawn', 'x_next_drawn',
                               'x_only_new_strokes',
                               'method'])

if eval_setting == 'same_text_add_detail' \
        or eval_setting == 'same_text_fill_in'\
        or eval_setting == 'something_from_nothing':
    # dataset = load_dataset("zoheb/sketch-scene")['train']
    dataset = load_dataset("nateraw/parti-prompts")['train']
    # dataset = dataset.filter(lambda example: example["Challenge"] == 'Basic') # Simple Detail
    # dataset = dataset.filter(lambda example: example["Category"] == 'Artifacts')
    dataset = dataset.filter(lambda example: example["Challenge"] == 'Simple Detail') # Simple Detail
    # dataset = dataset.filter(lambda example: example["Category"] == 'Artifacts')
    text_starts  = dataset['Prompt']
    random.Random(4).shuffle(text_starts)
    text_nexts = text_starts
elif eval_setting == 'different_text':
    dataset = load_dataset("nateraw/parti-prompts")['train']
    dataset = dataset.filter(lambda example: example["Challenge"] == 'Basic')
    dataset = dataset.filter(lambda example: example["Category"] == 'Artifacts')
    text_starts, text_nexts = dataset['Prompt'], dataset['Prompt']

    random.Random(4).shuffle(text_starts)
    random.Random(5).shuffle(text_nexts)
elif eval_setting == 'add_background':
    dataset = load_dataset("nateraw/parti-prompts")['train']
    dataset = dataset.filter(lambda example: example["Challenge"] == 'Basic')
    dataset = dataset.filter(lambda example: example["Category"] == 'Artifacts')
    nouns = dataset['Prompt']
    random.Random(4).shuffle(nouns)
    text_starts = [noun for noun in nouns]
    random.Random(5).shuffle(nouns)
    text_nexts = []
    for i in range(len(text_starts)):
        text_nexts.append(text_starts[i] + ' with ' + nouns[i] + ' in the background')

def add_to_df(df, dict):
    df1 = pd.DataFrame(dict)
    df = pd.concat([df, df1])
    return df

n = len(text_starts)

for i in tqdm(range(n)):
    text_start = text_starts[i]
    text_next = text_nexts[i]
    # text_next = "A person on the beach"
    
    print('Start: ', text_start)
    print('Next:  ', text_next)
    
    if ((df['text_start'] == text_start) \
            & (df['text_next'] == text_next)).any():
        x_start_fn = df[(df['text_start'] == text_start) & (df['text_next'] == text_next)]['x_start']
        x_start = Image.open(x_start_fn.iloc[0])
    else:
        # if opt.ink:
        #     with torch.no_grad():
        #         x_start = lora(text_start, 
        #                 num_inference_steps=30, 
        #                 num_images_per_prompt=1).images[0]
        # else:
        #     with torch.no_grad():
        #         stable_diffusion_non_img2img = stable_diffusion_non_img2img.to('cuda')
        #         x_start = stable_diffusion_non_img2img(
        #             prompt="A black and white drawing of " + text_start if opt.ink else text_start,
        #             num_inference_steps=20, num_images_per_prompt=1).images[0]
        #         stable_diffusion_non_img2img = stable_diffusion_non_img2img.to('cpu')
        stable_diffusion_non_img2img = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", #, torch_dtype=torch.float16
            safety_checker=None,
        )
        stable_diffusion_non_img2img.set_progress_bar_config(disable=True)
        stable_diffusion_non_img2img = stable_diffusion_non_img2img.to('cuda')
        x_start = stable_diffusion_non_img2img(
            prompt=text_start,
            num_inference_steps=20, num_images_per_prompt=1).images[0]
        stable_diffusion_non_img2img = stable_diffusion_non_img2img.to('cpu')
        del stable_diffusion_non_img2img

        x_start_fn = os.path.join(output_dir, '{}_start.png'.format(i))
        x_start.save(x_start_fn)

    x_start_drawn_fn = df.loc[(df['text_start'] == text_start) 
                        & (df['text_next'] == text_next), 'x_start_drawn']
    # print(x_start_drawn_fn, len(x_start_drawn_fn), x_start_drawn_fn.iloc[0])
    x_start_drawn_fn = x_start_drawn_fn.iloc[0] if len(x_start_drawn_fn) else ''

    if os.path.exists(x_start_drawn_fn):
        # x_start_drawn_tensor = load_img(x_start_drawn_fn, h=h, w=w).to(device)/255.
        x_start_drawn_pil_img = Image.open(x_start_drawn_fn).resize((512,512))
    else:
        target_img = load_img(x_start_fn, h=h, w=w).to(device)/255.
        current_canvas = load_img('caches/cache_10_8_ink/current_canvas.jpg',h=h, w=w).to(device)/255.
        if eval_setting != 'something_from_nothing':
            painting = plan_from_image(opt, target_img, current_canvas, opt.min_strokes_added) 

            with torch.no_grad():
                if eval_setting == 'same_text_fill_in':
                    from create_copaint_data import remove_strokes_by_region
                    p = remove_strokes_by_region(painting, target_img)
                else:
                    p = painting(h,w, use_alpha=False)
                x_start_drawn_pil_img = Image.fromarray(
                    (p.cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8))
        else:
            # Drawing on blank canvas
            x_start_drawn_pil_img = Image.open('caches/cache_6_6_cvpr/current_canvas.jpg')
        x_start_drawn_pil_img = x_start_drawn_pil_img.resize((512,512))
        x_start_drawn_fn = os.path.join(output_dir, os.path.basename(x_start_fn).replace('.png', '_drawn.png'))
        x_start_drawn_pil_img.save(x_start_drawn_fn)
        df.loc[df['x_start'] == x_start_fn, 'x_start_drawn'] = x_start_drawn_fn
        # with torch.no_grad():
        #     x_start_drawn_tensor = painting(h,w, use_alpha=False)
    
    def draw_and_save(x_next_fn, x_start_drawn_fn, text=None, text_and_image=False):
        target_img = load_img(x_next_fn, h=h, w=w).to(device)/255.
        current_canvas = load_img(x_start_drawn_fn,h=h, w=w).to(device)/255.
        painting = plan_from_image(opt, target_img, current_canvas, opt.max_strokes_added - opt.min_strokes_added, text=text, text_and_image=text_and_image)#, opt.num_strokes) 
        # Save x_next_drawn
        with torch.no_grad():
            x_next_drawn_pil_img = Image.fromarray(
                (painting(h,w, use_alpha=False).cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8))

        x_next_drawn_fn = os.path.join(output_dir, os.path.basename(x_next_fn).replace('.png', '_drawn.png'))
        x_next_drawn_pil_img.save(x_next_drawn_fn)

        # Save only the new strokes
        blank_canvas = load_img('caches/cache_10_8_ink/current_canvas.jpg',h=h, w=w).to(device)/255.
        painting = Painting(brush_strokes=painting.brush_strokes, background_img=blank_canvas)
        with torch.no_grad():
            x_only_new_strokes_pil_img = Image.fromarray(
                (painting(h,w, use_alpha=False).cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8))

        x_only_new_strokes_fn = os.path.join(output_dir, os.path.basename(x_next_fn).replace('.png', '_only_new_strokes.png'))
        x_only_new_strokes_pil_img.save(x_only_new_strokes_fn)

        return x_next_drawn_fn, x_only_new_strokes_fn


    # FRIDA with CLIP Text Loss
    if not ((df['text_start'] == text_start) 
            & (df['text_next'] == text_next)
            & (df['method'] == 'frida_clip')).any():

        current_canvas = load_img(x_start_drawn_fn,h=h, w=w).to(device)/255.
        painting = plan_from_image(opt, None, current_canvas, opt.max_strokes_added - opt.min_strokes_added, text=text_next)
        with torch.no_grad():
            frida_painting_img = painting(h,w, use_alpha=False)
        frida_painting_img = Image.fromarray((frida_painting_img[0].cpu().numpy().transpose(1,2,0) * 255.).astype(np.uint8))
        
        x_next_fn = os.path.join(output_dir, '{}_frida_clip.png'.format(i))
        frida_painting_img.save(x_next_fn)

        # Save only the new strokes
        blank_canvas = load_img('caches/cache_10_8_ink/current_canvas.jpg',h=h, w=w).to(device)/255.
        painting = Painting(brush_strokes=painting.brush_strokes, background_img=blank_canvas)
        with torch.no_grad():
            x_only_new_strokes_pil_img = Image.fromarray(
                (painting(h,w, use_alpha=False).cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8))
        x_only_new_strokes_fn = os.path.join(output_dir, os.path.basename(x_next_fn).replace('.png', '_only_new_strokes.png'))
        x_only_new_strokes_pil_img.save(x_only_new_strokes_fn)


        df = add_to_df(df, {
            'x_start':[x_start_fn],    'x_next':[x_next_fn],
            'x_start_drawn':[x_start_drawn_fn],
            'x_next_drawn':[x_next_fn],
            'x_only_new_strokes':[x_only_new_strokes_fn],
            'text_start':[text_start], 'text_next':[text_next],
            'method':['frida_clip']
        })


    for extra in range(4):
        method_name = 'instruct_pix2pix_ours_{}'.format(extra)
        # Instruct-Pix2Pix Ours
        if not ((df['text_start'] == text_start) 
                & (df['text_next'] == text_next)
                & (df['method'] == method_name)).any():
            instruct_pix2pix_ours = get_instruct_pix2pix_model(
                "timbrooks/instruct-pix2pix", 
                opt.abstract_planner_model, 
                device)
            with torch.no_grad():
                instruct_pix2pix_ours_img = instruct_pix2pix_ours(
                    text_next, x_start_drawn_pil_img, num_inference_steps=20, num_images_per_prompt=1,
                ).images[0]
            x_next_fn = os.path.join(output_dir, '{}_instruct_pix2pix_ours_{}.png'.format(i, extra))
            instruct_pix2pix_ours_img.save(x_next_fn)
            # Draw it
            x_next_drawn_fn, x_only_new_strokes_fn = draw_and_save(x_next_fn, x_start_drawn_fn)
            df = add_to_df(df, {
                'x_start':[x_start_fn],    'x_next':[x_next_fn],
                'x_start_drawn':[x_start_drawn_fn],
                'x_next_drawn':[x_next_drawn_fn],
                'x_only_new_strokes':[x_only_new_strokes_fn],
                'text_start':[text_start], 'text_next':[text_next],
                'method':[method_name]
            })
            del instruct_pix2pix_ours


    # Stable Diffusion Image-to-Image
    if not ((df['text_start'] == text_start) 
            & (df['text_next'] == text_next)
            & (df['method'] == 'stable_diffusion_img2img')).any():
        
        stable_diffusion_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", #, torch_dtype=torch.float16
            safety_checker=None,
        )
        stable_diffusion_img2img.set_progress_bar_config(disable=True)
        stable_diffusion_img2img = stable_diffusion_img2img.to('cuda')
        with torch.no_grad():
            sd_img = stable_diffusion_img2img(
                prompt=text_next, image=x_start_drawn_pil_img, num_inference_steps=20, num_images_per_prompt=1,
                strength=0.75, guidance_scale=7.5
            ).images[0]
        x_next_fn = os.path.join(output_dir, '{}_stable_diffusion_img2img.png'.format(i))
        sd_img.save(x_next_fn)
        # Draw it
        x_next_drawn_fn, x_only_new_strokes_fn = draw_and_save(x_next_fn, x_start_drawn_fn)
        df = add_to_df(df, {
            'x_start':[x_start_fn],    'x_next':[x_next_fn],
            'x_start_drawn':[x_start_drawn_fn],
            'x_next_drawn':[x_next_drawn_fn],
            'x_only_new_strokes':[x_only_new_strokes_fn],
            'text_start':[text_start], 'text_next':[text_next],
            'method':['stable_diffusion_img2img']
        })
        stable_diffusion_img2img = stable_diffusion_img2img.to('cpu')
        del stable_diffusion_img2img


    # # Plain Stable Diffusion 
    # if not ((df['text_start'] == text_start) 
    #         & (df['text_next'] == text_next)
    #         & (df['method'] == 'stable_diffusion')).any():
    #     stable_diffusion_non_img2img = stable_diffusion_non_img2img.to('cuda')
    #     with torch.no_grad():
    #         sd_img = stable_diffusion_non_img2img(
    #             prompt=text_next, 
    #             # image=x_start_drawn_pil_img, 
    #             num_inference_steps=20, num_images_per_prompt=1,
    #             # strength=1.0, 
    #             guidance_scale=7.5
    #         ).images[0]
    #     x_next_fn = os.path.join(output_dir, '{}_stable_diffusion.png'.format(i))
    #     sd_img.save(x_next_fn)
    #     # Draw it
    #     x_next_drawn_fn, x_only_new_strokes_fn = draw_and_save(x_next_fn, x_start_drawn_fn)
    #     df = add_to_df(df, {
    #         'x_start':[x_start_fn],    'x_next':[x_next_fn],
    #         'x_start_drawn':[x_start_drawn_fn],
    #         'x_next_drawn':[x_next_drawn_fn],
    #         'x_only_new_strokes':[x_only_new_strokes_fn],
    #         'text_start':[text_start], 'text_next':[text_next],
    #         'method':['stable_diffusion']
    #     })
    #     stable_diffusion_non_img2img = stable_diffusion_non_img2img.to('cpu')

    # Original Instruct-Pix2Pix
    if not ((df['text_start'] == text_start) 
            & (df['text_next'] == text_next)
            & (df['method'] == 'instruct_pix2pix_orig')).any():
        instruct_pix2pix_orig = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",#, torch_dtype=torch.float16
            safety_checker=None,
        )
        instruct_pix2pix_orig.set_progress_bar_config(disable=True)
        instruct_pix2pix_orig = instruct_pix2pix_orig.to('cuda')
        with torch.no_grad():
            instruct_pix2pix_orig_img = instruct_pix2pix_orig(
                text_next, x_start_drawn_pil_img, num_inference_steps=20, num_images_per_prompt=1
            ).images[0]
        x_next_fn = os.path.join(output_dir, '{}_instruct_pix2pix_original.png'.format(i))
        instruct_pix2pix_orig_img.save(x_next_fn)
        # Draw it
        x_next_drawn_fn, x_only_new_strokes_fn = draw_and_save(x_next_fn, x_start_drawn_fn)
        df = add_to_df(df, {
            'x_start':[x_start_fn],    'x_next':[x_next_fn],
            'x_start_drawn':[x_start_drawn_fn],
            'x_next_drawn':[x_next_drawn_fn],
            'x_only_new_strokes':[x_only_new_strokes_fn],
            'text_start':[text_start], 'text_next':[text_next],
            'method':['instruct_pix2pix_orig']
        })
        instruct_pix2pix_orig = instruct_pix2pix_orig.to('cpu')
        del instruct_pix2pix_orig

    df.to_csv(df_fn)
    df.to_csv(df_fn_backup)