
##########################################################
#################### Copyright 2023 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

"""
Create a dataset to be used to fine-tune Stable Diffusion using LoRA
"""

import argparse
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import kornia
from torchvision import transforms

import tkinter
import matplotlib
import matplotlib.pyplot as plt
import pickle
import shutil
matplotlib.use('TkAgg')

from plan import *

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
import torchvision.transforms as transforms


# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID).to(device)

preprocess = CLIPProcessor.from_pretrained(model_ID)
tokenizer = CLIPTokenizer.from_pretrained(model_ID)
text_encoder = CLIPTextModel.from_pretrained(model_ID).to(device)

# Define a function to load an image and preprocess it for CLIP
def load_image(image_path):
    response = requests.get(image_path, timeout=10)
    image = Image.open(BytesIO(response.content))

    return image

def image_text_similarity(image_path, text):

    with torch.no_grad():
        image = load_image(image_path)
        inputs = preprocess(text=[text], images=image, return_tensors="pt", padding=True)
        # [i.to(device) for i in inputs]
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['pixel_values'] = inputs['pixel_values'].to(device)

        # inputs['input_ids'] = inputs['input_ids'].to(device)
        # print(inputs)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image#logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return probs



def log_progress(painting, opt, log_freq=5, force_log=False, title='plan'):
    global local_it, plans
    local_it +=1
    if (local_it %log_freq==0) or force_log:
        with torch.no_grad():
            p = painting(h,w, use_alpha=False)
            p = format_img(p)
            opt.writer.add_image('images/{}'.format(title), p, local_it)
            
            plans.append((p*255.).astype(np.uint8))

def plan_from_image(opt, num_strokes, target_img, current_canvas, clip_lr=1.0):
    global colors
    painting = random_init_painting(current_canvas, num_strokes, ink=opt.ink)

    painting = initialize_painting(num_strokes, target_img, current_canvas, opt.ink)
    
    attn = get_attention(target_img)
    opt.writer.add_image('target/attention', format_img(torch.from_numpy(attn)[None,None,:,:]), 0)
    opt.writer.add_image('target/target', format_img(target_img), 0)


    painting = initialize_painting(0, target_img, current_canvas, opt.ink)
    painting.to(device)

    c = 0
    painting = add_strokes_to_painting(painting, num_strokes, target_img, opt.ink)
    optims = painting.get_optimizers(multiplier=opt.lr_multiplier, ink=opt.ink)

    # Learning rate scheduling. Start low, middle high, end low
    og_lrs = [o.param_groups[0]['lr'] if o is not None else None for o in optims]

    for it in tqdm(range(opt.n_iters), desc="Optim. {} Strokes".format(len(painting.brush_strokes))):
        for o in optims: o.zero_grad() if o is not None else None

        # lr_factor = (1 - 2*np.abs(it/opt.n_iters - 0.5)) + 0.1
        lr_factor = (1 - np.abs(it/opt.n_iters)) + 0.01 # 1.1 -> 0.01
        for i_o in range(len(optims)):
            if optims[i_o] is not None:
                optims[i_o].param_groups[0]['lr'] = og_lrs[i_o]*lr_factor

        p = painting(h, w, use_alpha=True, return_alphas=False)

        t = c / opt.n_iters
        c+=1 
        
        loss = 0
        # loss += parse_objective('l2', target_img, p[:,:3], weight=1-t)
        # loss += parse_objective('clip_conv_loss', target_img, p[:,:3], weight=clip_lr)
        loss += parse_objective('l2', target_img, p[:,:3], weight=1)
        loss += parse_objective('clip_conv_loss', target_img, p[:,:3], weight=1)

        loss.backward()

        for o in optims: o.step() if o is not None else None
        painting.validate()

        if not opt.ink:
            painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
        
        if (it % 10 == 0 and it > (0.5*opt.n_iters)) or it > 0.9*opt.n_iters:
            if not opt.ink:
                discretize_colors(painting, colors)
        # log_progress(painting, opt, log_freq=opt.log_frequency)#, force_log=True)

    return painting


def load_img_internet(url, h=None, w=None):
    try:
        response = requests.get(url, timeout=10)
        im = Image.open(BytesIO(response.content))
    except:
        return None
    # im = Image.open(path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im = np.array(im)
    # if im.shape[1] > max_size:
    #     fact = im.shape[1] / max_size
    im = cv2.resize(im, (w,h)) if h is not None and w is not None else im
    im = torch.from_numpy(im)
    im = im.permute(2,0,1)
    return im.unsqueeze(0).float() / 255.



def get_image_text_pair(dataset):
    datums = []
    least_complicated_value = 1e9
    best_datum = None
    resize = transforms.Resize((256,256))
    while len(datums) < opt.num_images_to_consider_for_simplicity:
        datum = dataset[np.random.randint(len(dataset))]
        img = load_img_internet(datum['URL'])
        if img is not None:
            datum['img'] = img
            mag, edges = kornia.filters.canny(resize(img))
            
            try:
                text_img_sim = image_text_similarity(datum['URL'], datum['TEXT'])
            except:
                continue
            # print(text_img_sim)
            if text_img_sim < 27: # Filter out images where the text caption isn't accurate
                continue
            
            datums.append(datum)

            if mag.sum() < least_complicated_value:
                least_complicated_value = mag.sum()
                best_datum = datum

    # all_imgs = torch.cat([resize(d['img']) for d in datums], dim=3)
    # all_imgs = torch.cat([all_imgs, resize(best_datum['img'])], dim=3)
    # show_img(all_imgs)

    # best_datum = dataset[np.random.randint(len(dataset))]
    # im = best_datum['image']
    # if im.mode != 'RGB':
    #     im = im.convert('RGB')
    # im = np.array(im)
    # # if im.shape[1] > max_size:
    # #     fact = im.shape[1] / max_size
    # im = cv2.resize(im, (w,h)) if h is not None and w is not None else im
    # im = torch.from_numpy(im)
    # im = im.permute(2,0,1)
    # best_datum['img'] = im.unsqueeze(0).float() / 255.

    return best_datum

def remove_strokes_randomly(painting, min_strokes_added, max_strokes_added):
    to_delete = set(random.sample(range(len(painting.brush_strokes)), max_strokes_added-min_strokes_added))
    enumerate(painting.brush_strokes)
    bs = [x for i,x in enumerate(painting.brush_strokes) if not i in to_delete]
    painting.brush_strokes = nn.ModuleList(bs)
    return painting

def remove_strokes_by_region(painting, target_img, keep_important=False):
    from clip_attn.clip_attn import get_attention
    attn = get_attention(target_img) 
    # attn = transforms.Resize((target_img.shape[2], target_img.shape[3]))(attn)
    # print('atnn', attn.shape, attn.min(), attn.max(), attn.mean())
    remaining_bs = []
    for bs in painting.brush_strokes:
        bs_x, bs_y = bs.transformation.xt*0.5 + .5, bs.transformation.yt*0.5 + .5
        # print(bs.transformation.xt, bs.transformation.yt)
        # print(bs_x, bs_y)
        bs_x_ind, bs_y_ind = int(bs_x*attn.shape[1]), int(bs_y*attn.shape[0])
        bs_x_ind, bs_y_ind = max(min(attn.shape[1]-1, bs_x_ind),0), max(min(attn.shape[0]-1, bs_y_ind),0)
        # print(bs_x_ind, bs_y_ind)
        attn_val = attn[bs_y_ind, bs_x_ind]
        if (attn_val < 0.25 and not keep_important) or (attn_val > 0.25 and keep_important):
            remaining_bs.append(bs)
    # print(len(painting.brush_strokes), len(remaining_bs))
    painting.brush_strokes = nn.ModuleList(remaining_bs)
    return painting


if __name__ == '__main__':
    global opt
    opt = Options()
    # python3 create_data_controlnet.py --use_cache --cache_dir caches/cache_6_6_cvpr/  --lr_multiplier 0.7 --output_parent_dir testing

    opt.gather_options()

    if not os.path.exists(opt.output_parent_dir): os.mkdir(opt.output_parent_dir)

    data_dict_fn = os.path.join(opt.output_parent_dir, 'data_dict.pkl')

    opt.writer = create_tensorboard(log_dir=opt.tensorboard_dir)

    # global h, w, colors, current_canvas, text_features, style_img, sketch
    stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
    h, w = stroke_shape[0], stroke_shape[1]
    w = int((opt.max_height/h)*w)
    h = int(opt.max_height)

    # Get the background of painting to be the current canvas
    if os.path.exists(os.path.join(opt.cache_dir, 'current_canvas.jpg')):
        current_canvas = load_img(os.path.join(opt.cache_dir, 'current_canvas.jpg'), h=h, w=w).to(device)/255.
    else:
        current_canvas = torch.ones(1,3,h,w).to(device)
    default_current_canvas = copy.deepcopy(current_canvas)

    # dataset = load_dataset("laion/laion-art")['train']
    # dataset = load_dataset("zoheb/sketch-scene")['train']
    dataset = load_dataset(opt.controlnet_dataset)['train']
    
    crop = transforms.RandomResizedCrop((h, w), scale=(0.7, 1.0), 
                                        ratio=(0.95,1.05))
    
    data_dict = []
    if os.path.exists(data_dict_fn):
        data_dict = pickle.load(open(data_dict_fn,'rb'))

    for i in range(opt.max_images):
        # Get a new image
        datum = get_image_text_pair(dataset)
        target_img = crop(datum['img']).to(device)
        colors = get_colors(cv2.resize(target_img.cpu().numpy()[0].transpose(1,2,0), (256, 256))*255., 
                n_colors=opt.n_colors)
        # print(datum)
        datum_no_img = copy.deepcopy(datum)
        datum_no_img['img'] = None # Don't save the image directly, just path
        current_canvas = default_current_canvas

        # Make sub-directories so single directories don't get too big
        output_dir = os.path.join(opt.output_parent_dir,
                                  str(int(np.floor(len(data_dict)/100))),)
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        
        target_img_path = os.path.join(output_dir, 'id{}_target.jpg'.format(len(data_dict)))
        save_image(target_img[:,:3],target_img_path)
        
        start_img_path = os.path.join(output_dir, 'id{}_start.jpg'.format(len(data_dict)))
        save_image(current_canvas[:,:3],start_img_path)
        
        painting = plan_from_image(opt, opt.max_strokes_added, target_img, current_canvas)
        with torch.no_grad():
            p = painting(h,w)
        
        final_img_path = os.path.join(output_dir, 'id{}_{}strokes.jpg'.format(len(data_dict), opt.max_strokes_added))
        save_image(p[:,:3],final_img_path)

        if opt.removal_method == 'random':
            # Randomly remove strokes to get the start image
            painting = remove_strokes_randomly(painting, opt.min_strokes_added, opt.max_strokes_added)
        elif opt.removal_method == 'salience':
            # Remove strokes by region
            painting = remove_strokes_by_region(painting, target_img)
        else:
            print("Not sure which removal method you mean")
            1/0

        with torch.no_grad():
            p = painting(h,w)
        save_image(p[:,:3],start_img_path)

        d = {'id':len(data_dict),
                'num_strokes_added':opt.max_strokes_added-opt.min_strokes_added,
                'num_prev_strokes':opt.min_strokes_added,
                'start_img':start_img_path,
                'final_img':final_img_path,
                'target_img':target_img_path,
            #  'text':datum['text'],#sketches
                'text':datum['TEXT'],
                'dataset_info':datum_no_img}

        # current_canvas = p.detach()
        # start_img_path = final_img_path

        data_dict.append(d)

        if os.path.exists(data_dict_fn):
            shutil.copyfile(data_dict_fn,
                        os.path.join(opt.output_parent_dir, 'data_dict_saved.pkl'))
                            
        with open(data_dict_fn,'wb') as f:
            pickle.dump(data_dict, f)
