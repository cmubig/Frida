
import csv
import json
import os
import pickle
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2

import datasets
import torch
from torch.utils.data import Dataset




def load_img(path, h=None, w=None):
    im = Image.open(path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    # im = np.array(im)
    # # if im.shape[1] > max_size:
    # #     fact = im.shape[1] / max_size
    # im = cv2.resize(im, (w,h)) if h is not None and w is not None else im
    # im = torch.from_numpy(im)
    # im = im.permute(2,0,1)
    # return im.unsqueeze(0).float()
    return im

class FridaControlNetDataset(Dataset):
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
        self.parent_dir = os.path.dirname(data_dict_path)
        print('\n\n\n\nparent dir', self.parent_dir)
        # Only from zero prev strokes
        # filtered_dict = []
        # for d in self.data_dict:
        #     if d['num_prev_strokes'] == 0:
        #         filtered_dict.append(d)
        # self.data_dict = filtered_dict

        # Only from zero prev strokes
        # filtered_dict = []
        # for d in self.data_dict:
        #     if d['num_prev_strokes'] < 200:
        #         filtered_dict.append(d)
        # self.data_dict = filtered_dict


        # # Only certain methods
        # # valid_methods = ['random', 'salience', 'not_salience', 'object', 'all']
        # valid_methods = ['random', 'object', 'all']
        # filtered_dict = []
        # for d in self.data_dict:
        #     if d['method'] in valid_methods:
        #         filtered_dict.append(d)
        # self.data_dict = filtered_dict

        # Check to make sure the image exists
        filtered_dict = []
        for d in self.data_dict:
            print(d['start_img'])
            print(os.path.join(self.parent_dir, d['start_img']))
            if os.path.exists(os.path.join(self.parent_dir, d['start_img'])):
                filtered_dict.append(d)
        self.data_dict = filtered_dict

        # # Only use examples with good clipscores
        # filtered_dict = []
        # for d in self.data_dict:
        #     # if d['clip_score'] > 0.65:
        #     if d['photo_to_sketch_diff'] > 0.03 and d['clip_score'] > 0.60:
        #         filtered_dict.append(d)
        # self.data_dict = filtered_dict
        # Only use examples with good clipscores
        filtered_dict = []
        for d in self.data_dict:
            # if d['clip_score'] > 0.65:
            if d['photo_to_sketch_diff'] > 0.03 and d['clip_score'] > 0.50:
                filtered_dict.append(d)
        self.data_dict = filtered_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx, h=None, w=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print('idx', idx)
        d = self.data_dict[idx]
        # print(d)
        start_img = load_img(os.path.join(self.parent_dir, d['start_img']), h=h, w=w)
        final_img = load_img(os.path.join(self.parent_dir, d['final_img']), h=h, w=w)
        text = d['text']

        # #####
        # blank = "/home/frida/paint/FridaXArm/src/lora_quality_data/1/id100_start.jpg"
        # import random
        # if random.randint(0,1):
        #     start_img = load_img(blank, h=h, w=w)
        # #######

        # ############
        # text = 'A black and white drawing of random lines' # d['text']
        # ############
        out = {
            'img_with_strokes':[final_img],
            'img_without_strokes':[start_img], # Num strokes is fixed
            'text':[text], 
        }

        # delta = (((np.array(final_img, dtype=np.float32) - np.array(start_img, dtype=np.float32))+255)/2).clip(0,255).astype(np.uint8)
        # # import matplotlib.pyplot as plt
        # # print(delta.max())
        # # plt.imshow(delta)
        # # plt.show()
        # # delta = np.array(final_img, dtype=np.float32) - np.array(start_img, dtype=np.float32)
        # # unchanged_pixel_inds = np.abs(delta) < 5
        # # change = np.array(final_img)
        # # change[unchanged_pixel_inds] = 0
        # out = {
        #     'img_with_strokes':[Image.fromarray(delta)],
        #     # 'img_with_strokes':[Image.fromarray(change)],
        #     'img_without_strokes':[start_img], # Num strokes is fixed
        #     'text':[text], 
        # }

        return self.transform(out)
    
    def with_transform(self, transform):
        self.transform = transform
        return self


class PickAndPlaceDataset(Dataset):
    def __init__(self, data_dict_path='/home/frida/Downloads/pick_and_place/dataset/'):
        """
        Arguments:
            root_dir (string): Directory with all the images.
        """
        import json
        if os.path.exists(data_dict_path):
            # self.data_dict = pickle.load(open(os.path.join(data_dict_path, 'en.train.jsonl'),'rb'))
            with open(os.path.join(data_dict_path, 'en.train.jsonl'),'rb') as json_file:
                json_list = list(json_file)
                # print(len(json_list))
                # 1/0
                for json_str in json_list:
                    result = json.loads(json_str)
                    # print(f"result: {result}")
                    print(result['image_file'])
                    print(result['pcd_file'])
                    print(result['objects'])
                    print(result.keys())
                    print(isinstance(result, dict))
                    1/0
        else:
            print('could not find data pickle file', data_dict_path)
        # self.transform = lambda x : x
        # self.data_source = self.data_dict
        self.parent_dir = os.path.dirname(os.path.dirname(data_dict_path))

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx, h=None, w=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print('idx', idx)
        d = self.data_dict[idx]
        start_img = load_img(os.path.join(self.parent_dir, d['start_img']), h=h, w=w)
        final_img = load_img(os.path.join(self.parent_dir, d['final_img']), h=h, w=w)
        text = d['text']

        out = {
            'img_with_strokes':[final_img],
            'img_without_strokes':[start_img], # Num strokes is fixed
            'text':[text], 
        }

        return out
# PickAndPlaceDataset()