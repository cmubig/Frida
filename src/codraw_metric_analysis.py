import copy
import os
import pathlib
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip 
import sys 

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'

if not os.path.exists('./clipscore/'):
    print('You have to clone the clipscore repo here from Github.')
sys.path.append('./clipscore')
from clipscore import get_clip_score, extract_all_images

from lavis.models import load_model_and_preprocess
blip_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)

def blip_score(raw_image, caption):
    # caption = "a large fountain spewing water into the air"
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](caption)
    sample = {"image": image, "text_input": [text_input]}

    # features_multimodal = model.extract_features(sample)
    # print(features_multimodal.multimodal_embeds.shape)
    # # torch.Size([1, 12, 768]), use features_multimodal[:,0,:] for multimodal classification tasks

    features_image = blip_model.extract_features(sample, mode="image")
    features_text = blip_model.extract_features(sample, mode="text")
    # print(features_image.image_embeds.shape)
    # # torch.Size([1, 197, 768])
    # print(features_text.text_embeds.shape)
    # # torch.Size([1, 12, 768])

    # # low-dimensional projected features
    # print(features_image.image_embeds_proj.shape)
    # # torch.Size([1, 197, 256])
    # print(features_text.text_embeds_proj.shape)
    # # torch.Size([1, 12, 256])
    similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
    similarity = similarity[0][0].item()
    # print(similarity)
    return similarity
# tensor([[0.2622]])


import open_clip
open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-g-14', 
                                                                                 pretrained='laion2b_s34b_b88k')
open_clip_tokenizer = open_clip.get_tokenizer('ViT-g-14')

def open_clip_score(raw_image, caption):

    image = open_clip_preprocess(raw_image).unsqueeze(0)
    text = open_clip_tokenizer([caption])

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = open_clip_model.encode_image(image)
        text_features = open_clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T)#.softmax(dim=-1)
    text_probs = text_probs.item()
    return text_probs


output_dir = 'codraw_metric_data5_less_strokes'
output_dir = 'codraw_metric_data6_150_50'
output_dir = 'codraw_metric_data4'
output_dir = 'codraw_metric_data7_100_100'
# output_dir = 'codraw_metric_data8_30_30'
# output_dir = 'codraw_metric_data9_0_30'
# output_dir = 'codraw_metric_data10_30_30_same_text'
output_dir = 'codraw_metric_data10_100_100_same_text'
output_dir = 'codraw_metric_data_add_background'
output_dir = 'codraw_metric_data/35_70_same_text_add_detail3'
# output_dir = 'codraw_metric_data/105_something_from_nothing'

df_fn = os.path.join(output_dir, 'codraw_metric_data.csv')

df = pd.read_csv(df_fn)
df['x_next_drawn_clipscore'] = -1
df['x_next_drawn_blipscore'] = -1
df['x_delta'] = -1
df['text_sim'] = -1
df['x_next_prev_prompt_clipscore'] = -1
df['sem_sim2real_gap'] = 0
df['x_only_new_clipscore'] = -1
df['metric'] = -1

model, transform = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

def clip_score(text_fn, img_fn):
    image_paths = [img_fn]
    candidates = [text_fn]

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    # get image-text clipscore
    with torch.no_grad():
        _, per_instance_image_text, candidate_feats = get_clip_score(
            model, image_feats, candidates, device)

    return per_instance_image_text[0]

for i in tqdm(range(len(df))):
    df.loc[i, 'x_next_drawn_blipscore'] = blip_score(Image.open(df.iloc[i]['x_next_drawn']), 
                                                     df.iloc[i]['text_next'])
    df.loc[i, 'x_next_drawn_open_clipscore'] = open_clip_score(Image.open(df.iloc[i]['x_next_drawn']), 
                                                               df.iloc[i]['text_next'])
    df.loc[i, 'x_next_drawn_clipscore'] = clip_score(df.iloc[i]['text_next'], df.iloc[i]['x_next_drawn'])
    df.loc[i, 'x_next_clipscore'] = clip_score(df.iloc[i]['text_next'], df.iloc[i]['x_next'])
    # df.loc[i, 'x_next_prev_prompt_clipscore'] = clip_score(df.iloc[i]['text_start'], df.iloc[i]['x_next_drawn'])
    # df.loc[i, 'x_only_new_clipscore'] = clip_score(df.iloc[i]['text_next'], df.iloc[i]['x_only_new_strokes'])
    # df.loc[i, 'metric'] = df.loc[i, 'x_next_drawn_clipscore'] - df.loc[i, 'x_only_new_clipscore']
    df.loc[i, 'sem_sim2real_gap'] = df.loc[i, 'x_next_clipscore'] - df.loc[i, 'x_next_drawn_clipscore']

    # df.loc[i, 'x_next_drawn_clipscore'] = np.random.rand(1)[0]

    # x_start = np.array(Image.open(df.iloc[i]['x_start_drawn'])).astype(np.float32) / 255.
    x_start = np.array(Image.open(df.iloc[i]['x_next']).resize((256,256))).astype(np.float32) / 255.
    x_next =  np.array(Image.open(df.iloc[i]['x_next_drawn' ]).resize((256,256))).astype(np.float32) / 255.
    delta = np.mean(np.abs(x_start - x_next))
    df.loc[i, 'x_delta'] = delta

    # print(df.iloc[i]['method'])
    # plt.imshow(Image.open(df.iloc[i]['x_next_drawn']))
    # plt.show()

    # # Text similarity
    # with torch.no_grad():
    #     text_features_start = model.encode_text(clip.tokenize(df.iloc[i]['text_start']).to(device))
    #     text_features_next  = model.encode_text(clip.tokenize(df.iloc[i]['text_next']).to(device))
    #     cos_sim = torch.cosine_similarity(text_features_start, text_features_next, dim=1)[0]
    #     df.loc[i, 'text_sim'] = float(cos_sim)
    # print(df.groupby('method')['sem_sim2real_gap'].mean())
    # print(df.groupby('method')['x_next_drawn_open_clipscore'].mean())
    
# print(df)
df.to_csv('codraw_metric_results.csv')
print(df.groupby('method')['x_next_drawn_clipscore'].mean())
print()
print(df.groupby('method')['x_next_drawn_blipscore'].mean())
print()
# print(df.groupby('method')['x_delta'].mean())
# print()
# print(df.groupby('method')['x_next_prev_prompt_clipscore'].median())
# print()

# print(df.groupby('method')['x_only_new_clipscore'].mean())
# print()
# print(df.groupby('method')['metric'].mean())
# print()
print(df.groupby('method')['sem_sim2real_gap'].mean())


df_output_fn = os.path.join(output_dir, 'codraw_metric_data_with_clip_score.csv')
df.to_csv(df_output_fn)

plt.figure()

methods = df['method'].unique()
print(methods)
for method in methods:
    plt.scatter(
                x=df[df['method'] == method]['sem_sim2real_gap'],
                # x=df[df['method'] == method]['text_sim'],
                y=df[df['method'] == method]['x_next_drawn_clipscore'],
                # y=df[df['method'] == method]['x_delta'],
                alpha=0.6,
                label=method)
plt.legend()
plt.show()

plt.figure()
for method in methods:
    plt.scatter(
                x=df[df['method'] == method]['x_next_drawn_blipscore'],
                # x=df[df['method'] == method]['text_sim'],
                y=df[df['method'] == method]['x_next_drawn_clipscore'],
                # y=df[df['method'] == method]['x_delta'],
                alpha=0.6,
                label=method)
plt.legend()
plt.show()