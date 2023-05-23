
##########################################################
#################### Copyright 2023 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

"""
Create a dataset to be used to fine-tune Stable Diffusion using LoRA
"""

import os
import pickle
from PIL import Image
import numpy as np

import datasets
import torch
from torch.utils.data import Dataset

def load_img(path, h=None, w=None):
    im = Image.open(path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return im

class FridaLoraDataset(Dataset):
    def __init__(self, data_dict_path):
        """
        Arguments:
            root_dir (string): Directory with all the images.
        """
        if os.path.exists(data_dict_path):
            self.data_dict = pickle.load(open(data_dict_path,'rb'))
        else:
            print('could not find data pickle file', data_dict_path)
        self.transform = lambda x : x
        self.data_source = self.data_dict

        # Only from zero prev strokes
        # filtered_dict = []
        # for d in self.data_dict:
        #     if d['num_prev_strokes'] == 0:
        #         filtered_dict.append(d)
        # self.data_dict = filtered_dict
        # filtered_dict = []
        # for d in self.data_dict:
        #     if d['num_prev_strokes'] > 100:
        #         filtered_dict.append(d)
        # self.data_dict = filtered_dict
        filtered_dict = []
        for d in self.data_dict:
            d['target_img'] = d['target_img'].replace("controlnet_data_ink_quality_100", "controlnet_data_ink_quality_100_vetted3")
            d['final_img'] = d['final_img'].replace("controlnet_data_ink_quality_100", "controlnet_data_ink_quality_100_vetted3")
            d['start_img'] = d['start_img'].replace("controlnet_data_ink_quality_100", "controlnet_data_ink_quality_100_vetted3")
            if os.path.exists(d['target_img']) and os.path.exists(d['final_img']):
                filtered_dict.append(d)
        self.data_dict = filtered_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx, h=None, w=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print('idx', idx)
        d = self.data_dict[idx]
        start_img = load_img(os.path.join(d['start_img']), h=h, w=w)
        final_img = load_img(os.path.join(d['final_img']), h=h, w=w)
        # final_img = load_img(os.path.join(d['target_img']), h=h, w=w)
        text = d['text']

        num_strokes = d['num_strokes_added'] + d['num_prev_strokes']
        num_strokes_img = np.array(start_img.copy()).astype(np.float32)
        num_strokes_img = (num_strokes_img * 0 + 1) # All 1's
        max_strokes = 400
        num_strokes_img = num_strokes_img*(num_strokes / max_strokes * 255.) # 0-255. scaled by num_strokes
        num_strokes_img = Image.fromarray(num_strokes_img.astype(np.uint8))
        # out = {
        #     'pixel_values':start_img,
        #     'conditioning_img':final_img, 
        #     'input_ids':text, 
        # }
        # out = {
        #     'img_with_strokes':[final_img],
        #     'img_without_strokes':[start_img], 
        #     'text':[text], 
        # }
        out = {
            'img_with_strokes':[final_img],
            'img_without_strokes':[num_strokes_img], 
            'text':[text], 
        }

        return self.transform(out)
    
    def with_transform(self, transform):
        self.transform = transform
        return self
    
# dataset = FridaLoraDataset('controlnet_data/data_dict.pkl')
