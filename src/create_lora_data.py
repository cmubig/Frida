
##########################################################
#################### Copyright 2023 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

"""
Create a dataset to be used to fine-tune Stable Diffusion using LoRA
"""

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


# python3 create_lora_data.py --use_cache --cache_dir caches/cache_6_6/ --lr_multiplier 1.0 --ink
min_strokes_added = 100
max_strokes_added = 101
max_strokes_total = 100
num_images_to_consider_for_simplicity = 10
n_iters = 700 # Optimization iterations
output_parent_dir = 'lora_quality_data'
if not os.path.exists(output_parent_dir): os.mkdir(output_parent_dir)

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

    painting = add_strokes_to_painting(painting, painting(h,w)[:,:3], num_strokes, 
                                        target_img, current_canvas, opt.ink)
    optims = painting.get_optimizers(multiplier=opt.lr_multiplier, ink=opt.ink)

    # Learning rate scheduling. Start low, middle high, end low
    og_lrs = [o.param_groups[0]['lr'] if o is not None else None for o in optims]

    for it in tqdm(range(n_iters), desc="Optim. {} Strokes".format(len(painting.brush_strokes))):
        for o in optims: o.zero_grad() if o is not None else None

        # lr_factor = (1 - 2*np.abs(it/n_iters - 0.5)) + 0.1
        lr_factor = (1 - np.abs(it/n_iters)) + 0.01 # 1.1 -> 0.1
        for i_o in range(len(optims)):
            if optims[i_o] is not None:
                optims[i_o].param_groups[0]['lr'] = og_lrs[i_o]*lr_factor

        p = painting(h, w, use_alpha=True, return_alphas=False)

        t = c / n_iters
        c+=1 
        
        loss = 0
        # loss += parse_objective('l2', target_img, p[:,:3], weight=1-t)
        # loss += parse_objective('clip_conv_loss', target_img, p[:,:3], weight=clip_lr)
        # loss += parse_objective('l2', target_img, p[:,:3], weight=1)
        loss += parse_objective('clip_conv_loss', target_img, p[:,:3], weight=1)

        loss.backward()

        for o in optims: o.step() if o is not None else None
        painting.validate()

        if not opt.ink:
            painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
        
        if (it % 10 == 0 and it > (0.5*n_iters)) or it > 0.9*n_iters:
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
    while len(datums) < num_images_to_consider_for_simplicity:
        datum = dataset[np.random.randint(len(dataset))]
        img = load_img_internet(datum['URL'])
        if img is not None:
            datum['img'] = img
            datums.append(datum)
            mag, edges = kornia.filters.canny(resize(img))

            if mag.sum() < least_complicated_value:
                least_complicated_value = mag.sum()
                best_datum = datum

    # # all_imgs = torch.cat([resize(d['img']) for d in datums], dim=3)
    # # all_imgs = torch.cat([all_imgs, resize(best_datum['img'])], dim=3)
    # # show_img(all_imgs)

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



data_dict_fn = os.path.join(output_parent_dir, 'data_dict.pkl')

if __name__ == '__main__':
    global opt
    opt = Options()
    opt.gather_options()

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

    dataset = load_dataset("laion/laion-art")['train']
    # dataset = load_dataset("zoheb/sketch-scene")['train']
    
    crop = transforms.RandomResizedCrop((h, w), scale=(0.7, 1.0), 
                                        ratio=(0.95,1.05))
    
    data_dict = []
    if os.path.exists(data_dict_fn):
        data_dict = pickle.load(open(data_dict_fn,'rb'))

    for i in range(20000):
        # Get a new image
        datum = get_image_text_pair(dataset)
        target_img = crop(datum['img']).to(device)
        colors = get_colors(cv2.resize(target_img.cpu().numpy()[0].transpose(1,2,0), (256, 256))*255., 
                n_colors=opt.n_colors)
        # print(datum)
        datum_no_img = copy.deepcopy(datum)
        datum_no_img['img'] = None # Don't save the image directly, just path
        num_prev_strokes = 0
        current_canvas = default_current_canvas
        paint_its = 0

        # Make sub-directories so single directories don't get too big
        output_dir = os.path.join(output_parent_dir,
                                  str(int(np.floor(len(data_dict)/100))),)
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        
        target_img_path = os.path.join(output_dir, 'id{}_target.jpg'.format(len(data_dict)))
        save_image(target_img[:,:3],target_img_path)
        
        start_img_path = os.path.join(output_dir, 'id{}_start.jpg'.format(len(data_dict)))
        save_image(current_canvas[:,:3],start_img_path)
        
        # for paint_its in range(5):
        # for paint_its in range(1):
        while(num_prev_strokes < max_strokes_total):
            num_strokes_added = np.random.randint(low=min_strokes_added, 
                                                  high=max_strokes_added)
            
            painting = plan_from_image(opt, num_strokes_added, target_img, current_canvas,
                                       clip_lr=0.6*(1.2-((num_strokes_added+num_prev_strokes)/max_strokes_total)))
            with torch.no_grad():
                p = painting(h,w)
            final_img_path = os.path.join(output_dir, 'id{}_pass{}_{}strokes.jpg'.format(len(data_dict), paint_its, num_prev_strokes+num_strokes_added))
            save_image(p[:,:3],final_img_path)
            d = {'id':len(data_dict),
                 'num_strokes_added':num_strokes_added,
                 'num_prev_strokes':num_prev_strokes,
                 'start_img':start_img_path,
                 'final_img':final_img_path,
                 'target_img':target_img_path,
                #  'text':datum['text'],#sketches
                 'text':datum['TEXT'],
                 'dataset_info':datum_no_img}

            num_prev_strokes += num_strokes_added
            current_canvas = p.detach()
            paint_its += 1
            start_img_path = final_img_path

            data_dict.append(d)

        if os.path.exists(data_dict_fn):
            shutil.copyfile(data_dict_fn,
                        os.path.join(output_parent_dir, 'data_dict_saved.pkl'))
                            
        with open(data_dict_fn,'wb') as f:
            pickle.dump(data_dict, f)
