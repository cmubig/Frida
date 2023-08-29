import torch 
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob


# methods = [
#     # 'instruct-pix2pix_ours',
#     'instruct-pix2pix_ours_ckpt_3000',
#     'stable_diffusion',
#     'stable_diffusion_img2img',
#     # 'instruct-pix2pix_original',
# ]
output_dir = 'codraw_metric_data9_0_30'
output_dir = 'codraw_metric_data7_100_100'
output_dir = 'codraw_metric_data10_30_30_same_text'
output_dir = 'codraw_metric_data/35_70_same_text_add_detail3'
output_dir = 'codraw_metric_data/105_something_from_nothing'

method_fns = glob.glob(os.path.join(output_dir, '0_*_drawn.png'))

# print(method_fns)
methods = ['frida_clip']
for method_fn in method_fns:

    method = method_fn.replace(output_dir + '/0_', '').replace('_drawn.png','')
    # print(method_fn, method)

    if method != 'start':
        methods.append(method)
print(methods)


df = pd.read_csv(os.path.join(output_dir, 'codraw_metric_data.csv'))
df = pd.read_csv(os.path.join(output_dir, 'codraw_metric_data_with_clip_score.csv'))


norm_cols = ['x_next_drawn_clipscore', 'x_next_drawn_blipscore', 'x_next_drawn_open_clipscore']
for col_name in norm_cols:
    df[col_name] = df[col_name].astype(float)#.values[0]
    print(col_name, df[col_name].mean(), df[col_name].std(skipna=True))
    # print(df[col_name])
    df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std(skipna=True)

for i in range(30):
    fig, ax = plt.subplots(5,len(methods), figsize=(13,9))

    start_img = Image.open(os.path.join(output_dir, 
                "{}_start.png".format(i)))
    start_img = start_img.resize((256,256))
    imstart_imgg = np.array(start_img)
    start_img_drawn = Image.open(os.path.join(output_dir, 
                "{}_start_drawn.png".format(i)))
    start_img_drawn = start_img_drawn.resize((256,256))
    start_img_drawn = np.array(start_img_drawn)
    # start_img = np.concatenate([start_img, start_img_drawn], axis=0)

    t = df.loc[df['x_only_new_strokes'] == os.path.join(output_dir, 
                        "{}_{}{}".format(i,method, '_only_new_strokes.png'))]
    text_next = t['text_next'].astype(str).values[0]
    text_start = t['text_start'].astype(str).values[0]
    
    
    col_ind = 0
    for method in methods:
        row_ind = 0

        t = df.loc[df['x_only_new_strokes'] == os.path.join(output_dir, 
                        "{}_{}{}".format(i,method, '_only_new_strokes.png'))]
        
        sem_sim2real_gap = t['sem_sim2real_gap'].astype(float).values[0]
        clip_score = t['x_next_drawn_clipscore'].astype(float).values[0]
        blip_score = t['x_next_drawn_blipscore'].astype(float).values[0]
        open_clip_score = t['x_next_drawn_open_clipscore'].astype(float).values[0]

        for out_type in ['start', 'start_img_drawn', '.png', 'padding', '_drawn.png']:
            if out_type == 'start':
                img = start_img
            elif out_type == 'start_img_drawn':
                img = start_img_drawn
            elif out_type == 'padding':
                img = np.ones((8,128,3))
            elif method == 'frida_clip':
                img = Image.open(os.path.join(output_dir, 
                            "{}_{}.png".format(i,method)))
                img = img.resize((256,256))
                img = np.array(img)
            else:
                img = Image.open(os.path.join(output_dir, 
                            "{}_{}{}".format(i,method, out_type)))
                img = img.resize((256,256))
                img = np.array(img)

            ax[row_ind, col_ind].imshow(img)
            # ax[row_ind, col_ind].set_xticks([])
            # ax[row_ind, col_ind].set_yticks([])
            ax[row_ind, col_ind].axis('off')
            if row_ind == 0 and col_ind == 1:
                ax[row_ind, col_ind].set_title(text_next)
            elif row_ind == 2:
                ax[row_ind, col_ind].set_title('{}'.format(method\
                                    .replace('instruct-pix2pix_ours_ckpt', 'ours') \
                                    .replace('instruct-pix2pix_original', 'p2p_') \
                                    .replace('stable_diffusion', 'sd_')))
            elif row_ind == 4:
                ax[row_ind, col_ind].set_title('cs={:.3f}\nbs={:.3f}\nocs={:.3f}\ns2r={:.3f}'.format(
                                    clip_score, blip_score, open_clip_score, sem_sim2real_gap))

            row_ind += 1
        col_ind += 1
            
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(output_dir, '{}_display_img.png'.format(i)))


# for i in range(1000):
#     fig, ax = plt.subplots(1,2, figsize=(10,4))

#     start_img = Image.open(os.path.join(output_dir, 
#                 "{}_start.png".format(i)))
#     start_img = start_img.resize((256,256))
#     imstart_imgg = np.array(start_img)
#     start_img_drawn = Image.open(os.path.join(output_dir, 
#                 "{}_start_drawn.png".format(i)))
#     start_img_drawn = start_img_drawn.resize((256,256))
#     start_img_drawn = np.array(start_img_drawn)
#     start_img = np.concatenate([start_img, start_img_drawn], axis=0)
    
    
#     whole_img = None
#     for method in methods:
#         col = None
#         for out_type in ['.png', '_only_new_strokes.png', '_drawn.png']:
#             img = Image.open(os.path.join(output_dir, 
#                         "{}_{}{}".format(i,method, out_type)))
#             img = img.resize((256,256))
#             img = np.array(img)
#             if col is None:
#                 col = img
#             else:
#                 col = np.concatenate([col, img], axis=0)
#         if whole_img is None:
#             whole_img = col 
#         else:
#             whole_img = np.concatenate([whole_img, col], axis=1)
#     t = df.loc[df['x_only_new_strokes'] == os.path.join(output_dir, 
#                         "{}_{}{}".format(i,method, '_only_new_strokes.png'))]
#     text_next = t['text_next'].astype(str).values[0]
#     text_start = t['text_start'].astype(str).values[0]

#     ax[0].imshow(start_img)
#     ax[0].set_title(text_start)
#     ax[0].set_xticks([])
#     ax[0].set_yticks([])
#     ax[1].imshow(whole_img)
#     ax[1].set_title(text_next)
#     ax[1].set_xticks([])
#     ax[1].set_yticks([])
#     # plt.show()
#     fig.savefig(os.path.join(output_dir, '{}_display_img.png'.format(i)))