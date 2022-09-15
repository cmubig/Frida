import pickle
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import requests
from PIL import Image
import io
import cv2
from tqdm import tqdm
import os
import PIL.Image, PIL.ImageDraw
from io import BytesIO
import lpips
import copy
import colour
import random
import gzip
import kornia as K
import datetime
from tqdm import tqdm

from options import Options
from tensorboard import TensorBoard

# from torch_painting_models import *
# from style_loss import compute_style_loss
from strokes import all_strokes


# from clip_loss import clip_conv_loss, clip_model, clip_text_loss
# import clip
# import kornia as K

# from plan_all_strokes import load_brush_strokes, show_img

# import pydiffvg

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn 
    return pp

def to_full_param(length, bend, z):
    full_param = torch.zeros((1,12)).to(device)
    
    # X
    full_param[0,0] = 0
    full_param[0,3] = length/3 
    full_param[0,6] = 2*length/3
    full_param[0,9] = length
    # Y
    full_param[0,1] = 0
    full_param[0,4] = bend
    full_param[0,7] = bend
    full_param[0,10] = 0
    # Z
    full_param[0,2] = 0.2
    full_param[0,5] = z
    full_param[0,8] = z
    full_param[0,11] = 0.2

    return full_param

def display_all_permutations(model):
    # Length v Bend
    n_img = 10
    lengths = torch.arange(n_img)/(n_img-1)*(0.05-0.01) + 0.01
    bends = torch.arange(n_img)/(n_img-1)*0.04 - 0.02
    zs = torch.arange(n_img)/(n_img-1)

    fig, ax = plt.subplots(n_img, n_img, figsize=(30,30))

    for i in range(n_img):
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], bends[j], 0.5)
            s = model(trajectory)
            # print(s.shape)
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

            ax[i,j].imshow(s, cmap='gray')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(n_img, n_img, figsize=(30,30))

    for i in range(n_img):
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], 0.0, zs[j])
            s = model(trajectory)
            # print(s.shape)
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

            ax[i,j].imshow(s, cmap='gray')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    plt.tight_layout()
    plt.show()



# class StrokeParametersToImage(nn.Module):
#     def __init__(self, h, w):
#         super(StrokeParametersToImage, self).__init__()
#         # self.fc1 = nn.Linear(12, w*h)
#         # self.fc1 = nn.Linear(12, 1000)
#         # self.fc2 = nn.Linear(1000, 1000)
#         # self.fc3 = nn.Linear(1000, 1000)
#         # self.fc4 = nn.Linear(1000, w*h)
#         nh = 30
#         self.main = nn.Sequential(
#             nn.BatchNorm1d(12),
#             #
#             nn.Linear(12, nh),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm1d(nh),
#             nn.Linear(nh, nh),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm1d(nh),
#             nn.Linear(nh, nh),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm1d(nh),
#             nn.Linear(nh, nh),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm1d(nh),
#             nn.Linear(nh, nh),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(nh, w*h),
#             nn.Sigmoid()
#         )
#         self.w = w 
#         self.h = h 

#     def forward(self, x):
#         # return self.fc1(x).view(-1, self.h, self.w)
#         # return nn.Sigmoid()(self.fc4(nn.ReLU()(self.fc3(nn.ReLU()(self.fc2(nn.ReLU()(self.fc1(x))))))).view(-1, h, w))
#         return self.main(x).view(-1, self.h, self.w)



# class StrokeParametersToImage(nn.Module):
#     def __init__(self, h, w):
#         super(StrokeParametersToImage, self).__init__()
#         nh = 30
#         self.nc = 1
#         self.main = nn.Sequential(
#             nn.BatchNorm1d(12),
#             #
#             nn.Linear(12, nh),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm1d(nh),
#             # nn.Linear(nh, nh),
#             # nn.LeakyReLU(0.2, inplace=True),
#             # nn.BatchNorm1d(nh),
#             # nn.Linear(nh, nh),
#             # nn.LeakyReLU(0.2, inplace=True),
#             # nn.BatchNorm1d(nh),
#             # nn.Linear(nh, nh),
#             # nn.LeakyReLU(0.2, inplace=True),
#             # nn.BatchNorm1d(nh),
#             # nn.Linear(nh, nh),
#             # nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(nh, self.nc*w*h),
#             nn.Sigmoid()
#         )
#         self.conv = nn.Sequential(
#             nn.Conv2d(self.nc, 5, kernel_size=4, padding='same', dilation=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm2d(5),
#             nn.Conv2d(5, 1, kernel_size=4, padding='same', dilation=1),
#             nn.Sigmoid()
#         )
#         self.w = w 
#         self.h = h 

#     def forward(self, x):
#         # return self.fc1(x).view(-1, self.h, self.w)
#         # return nn.Sigmoid()(self.fc4(nn.ReLU()(self.fc3(nn.ReLU()(self.fc2(nn.ReLU()(self.fc1(x))))))).view(-1, h, w))
#         return self.conv(self.main(x).view(-1, self.nc, self.h, self.w))[:,0]


class StrokeParametersToImage(nn.Module):
    def __init__(self, h, w):
        super(StrokeParametersToImage, self).__init__()
        nh = 10
        self.nc = 10
        self.main = nn.Sequential(
            nn.BatchNorm1d(12),
            #
            nn.Linear(12, nh),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(nh),
            # nn.Linear(nh, nh),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm1d(nh),
            # nn.Linear(nh, nh),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm1d(nh),
            # nn.Linear(nh, nh),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm1d(nh),
            # nn.Linear(nh, nh),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nh, 64*64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.nc, kernel_size=4, padding='same', dilation=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.nc),
            nn.Conv2d(self.nc, 1, kernel_size=4, padding='same', dilation=1),
            nn.Sigmoid()
        )
        self.w = w 
        self.h = h 
        self.res = transforms.Resize((self.h,self.w))
        self.res2 = transforms.Resize((self.h,self.w))

    def forward(self, x):
        # return self.fc1(x).view(-1, self.h, self.w)
        # return nn.Sigmoid()(self.fc4(nn.ReLU()(self.fc3(nn.ReLU()(self.fc2(nn.ReLU()(self.fc1(x))))))).view(-1, h, w))
        # return self.conv(self.main(x).view(-1, self.nc, self.h, self.w))[:,0]
        x = self.res2(self.conv((self.main(x).view(-1, 1, 64, 64))))[:,0]
        # x = 1/(1+torch.exp(-1.*(x*2-1) / 0.05))
        return x

    def forward_full(self, x):
        x = self.forward(x)
        return transforms.Pad((ws, hs, w_og-we, h_og-he))(x)


if __name__ == '__main__':
    global opt, strokes
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    writer = TensorBoard('painting_stroke_modelling/model_stroke{}'.format(run_name))
    opt.writer = writer

    # strokes = np.load('/tmp/extended_stroke_library_intensities.npy', allow_pickle=True) 
    # trajectories = np.load('/tmp/extended_stroke_library_trajectories.npy', allow_pickle=True, encoding='bytes') 
    # print(trajectories)
    strokes = np.load('extended_stroke_library_intensities.npy') 
    trajectories = np.load('extended_stroke_library_trajectories.npy', allow_pickle=True, encoding='bytes') 

    # strokes = strokes[:60]
    # trajectories = trajectories[:60]

    strokes = torch.from_numpy(strokes).to(device).float().nan_to_num()
    trajectories = torch.from_numpy(trajectories.astype(np.float32)).to(device).float().nan_to_num()
    print(strokes.shape, trajectories.shape)
    print(trajectories[0:2])

    scale_factor = 4
    strokes = transforms.Resize((int(strokes.shape[1]/scale_factor), int(strokes.shape[2]/scale_factor)))(strokes)

    # Randomize
    rand_ind = torch.randperm(strokes.shape[0])
    strokes = strokes[rand_ind]
    trajectories = trajectories[rand_ind]

    # Discrete
    strokes[strokes >= 0.1] = 1.
    strokes[strokes < 0.1] = 0.

    h, w = strokes[0].shape[0], strokes[0].shape[1]
    h_og, w_og = h, w

    # Crop
    hs, he = int(.4*h), int(0.6*h)
    ws, we = int(0.45*w), int(0.75*w)
    strokes = strokes[:, hs:he, ws:we]

    h, w = strokes[0].shape[0], strokes[0].shape[1]
    print(len(strokes), strokes[0].shape, strokes[0].max())


    trans = StrokeParametersToImage(h,w).to(device)
    print('parameters', get_n_params(trans))

    val_prop = .3

    # for stroke_ind in range(len(strokes)):
    #     writer.add_image('images/{}_real_stroke'.format(stroke_ind), 
    #         np.clip(strokes[stroke_ind].detach().cpu().numpy(), a_min=0, a_max=1)*255, 0)

    trans = StrokeParametersToImage(h,w).to(device)
    optim = torch.optim.Adam(trans.parameters(), lr=1e-3)
    best_model = None
    best_val_loss = 999
    best_hasnt_changed_for = 0

    n = len(strokes)

    train_strokes = strokes[int(val_prop*n):]
    train_trajectories = trajectories[int(val_prop*n):]
    val_strokes = strokes[:int(val_prop*n)]
    val_trajectories = trajectories[:int(val_prop*n)]

    for it in tqdm(range(2000)):
        if best_hasnt_changed_for >= 200:
            break
        optim.zero_grad()

        noise = torch.randn(train_trajectories.shape).to(device)*0.005 # For robustness
        pred_strokes = trans(train_trajectories + noise)

        # loss = nn.L1Loss()(pred_strokes, train_strokes)
        loss = nn.MSELoss()(pred_strokes, train_strokes)

        ep_loss = loss.item()
        loss.backward()
        optim.step()

        writer.add_scalar('loss/train_loss', ep_loss, it)
        
        n_view = 10
        with torch.no_grad():
            trans.eval()
            pred_strokes_val = trans(val_trajectories)
            if it % 50 == 0:
                for val_ind in range(min(n_view,len(val_strokes))):
                    writer.add_image('images/val_{}_sim_stroke'.format(val_ind), 
                        np.clip(pred_strokes_val.detach().cpu().numpy()[val_ind], a_min=0, a_max=1)*255, it)
                    if it == 0:
                        writer.add_image('images/val_{}_real_stroke'.format(val_ind), 
                            np.clip(val_strokes.detach().cpu().numpy()[val_ind], a_min=0, a_max=1)*255, it)
                pred_strokes_train = trans(train_trajectories)
                for train_ind in range(min(n_view,len(train_strokes))):
                    writer.add_image('images/train_{}_sim_stroke'.format(train_ind), 
                        np.clip(pred_strokes_train.detach().cpu().numpy()[train_ind], a_min=0, a_max=1)*255, it)
                    if it == 0:
                        writer.add_image('images/train_{}_real_stroke'.format(train_ind), 
                            np.clip(train_strokes.detach().cpu().numpy()[train_ind], a_min=0, a_max=1)*255, it)

            loss = nn.MSELoss()(pred_strokes_val, val_strokes)
            writer.add_scalar('loss/val_loss', loss.item(), it)

            if loss.item() < best_val_loss and it > 50:
                best_val_loss = loss.item()
                best_hasnt_changed_for = 0
                best_model = copy.deepcopy(trans)
            best_hasnt_changed_for += 1
            trans.train()
    with torch.no_grad():
        best_model.eval()
        pred_strokes_val = best_model(val_trajectories)
        for val_ind in range(min(n_view,len(val_strokes))):
            writer.add_image('images/val_{}_sim_stroke_best'.format(val_ind), 
                np.clip(pred_strokes_val.detach().cpu().numpy()[val_ind], a_min=0, a_max=1)*255, 1)

        pred_strokes_train = best_model(train_trajectories)
        for train_ind in range(min(n_view,len(train_strokes))):
            writer.add_image('images/train_{}_sim_stroke_best'.format(train_ind), 
                np.clip(pred_strokes_train.detach().cpu().numpy()[train_ind], a_min=0, a_max=1)*255, 1)

        # Make them into full sized strokes
        pred_strokes_val = best_model(val_trajectories)
        pred_strokes_val = 1/(1+torch.exp(-1.*((pred_strokes_val*2-1)+0.5) / 0.05))
        pred_strokes_val = transforms.Pad((ws, hs, w_og-we, h_og-he))(pred_strokes_val)
        val_strokes_full = transforms.Pad((ws, hs, w_og-we, h_og-he))(val_strokes)
        for val_ind in range(min(n_view,len(val_strokes))):
            writer.add_image('images/val_{}_sim_stroke_best_full'.format(val_ind), 
                np.clip(pred_strokes_val.detach().cpu().numpy()[val_ind], a_min=0, a_max=1)*255, 1)
            writer.add_image('images/val_{}_real_stroke_full'.format(val_ind), 
                np.clip(val_strokes_full.detach().cpu().numpy()[val_ind], a_min=0, a_max=1)*255, 1)
    display_all_permutations(best_model)
    torch.save(best_model.cpu().state_dict(), 'param2img.pt')


    # val_stroke_ind = 7

    # for stroke_ind in range(len(strokes)):
    #     writer.add_image('images/{}_real_stroke'.format(stroke_ind), 
    #         np.clip(strokes[stroke_ind].detach().cpu().numpy(), a_min=0, a_max=1)*255, 0)

    # for val_ind in range(len(strokes)):
    #     trans = StrokeParametersToImage(h,w).to(device)
    #     optim = torch.optim.Adam(trans.parameters(), lr=1e-2)
    #     best_model = None
    #     best_val_loss = 999

    #     train_strokes = torch.cat([strokes[:val_ind], strokes[val_ind+1:]], dim=0)
    #     train_trajectories = torch.cat([trajectories[:val_ind], trajectories[val_ind+1:]], dim=0)
    #     # train_strokes = strokes 
    #     # train_trajectories = trajectories 

    #     for it in range(3000):
    #         optim.zero_grad()
    #         # print(train_trajectories.shape, train_strokes.shape)
    #         # print(train_trajectories[0:2])
    #         pred_strokes = trans(train_trajectories)

    #         # loss = nn.L1Loss()(pred_strokes, train_strokes)
    #         loss = nn.MSELoss()(pred_strokes, train_strokes)
    #         # loss = ((torch.abs(pred_strokes - train_strokes)+1e-5)**.5).mean()
    #         # loss = (torch.abs(pred_strokes - train_strokes)**4).mean()

    #         ep_loss = loss.item()
    #         loss.backward()
    #         optim.step()

    #         writer.add_scalar('loss/train_loss', ep_loss, it)
            
    #         with torch.no_grad():
    #             trans.eval()
    #             pred_stroke = trans(trajectories[val_ind:val_ind+1])
    #             if it % 50 == 0:
    #                 writer.add_image('images/{}_sim_stroke'.format(val_ind), 
    #                     np.clip(pred_stroke.detach().cpu().numpy()[0], a_min=0, a_max=1)*255, it)

    #             real_stroke = strokes[val_ind].unsqueeze(0)
    #             loss = nn.MSELoss()(pred_stroke, real_stroke)
    #             writer.add_scalar('loss/val_loss_{}'.format(val_ind), loss.item(), it)

    #             if loss.item() < best_val_loss and it > 50:
    #                 best_val_loss = loss.item()
    #                 best_model = copy.deepcopy(trans)
    #             trans.train()
    #     with torch.no_grad():
    #         best_model.eval()
    #         pred_stroke = best_model(trajectories[val_ind:val_ind+1])
    #         writer.add_image('images/{}_sim_stroke'.format(val_ind), 
    #                 np.clip(pred_stroke.detach().cpu().numpy()[0], a_min=0, a_max=1)*255, 10000)

    #         non_val_ind = val_ind-1 if val_ind > 0 else len(strokes)-1
    #         pred_stroke = best_model(trajectories[non_val_ind:non_val_ind+1])
    #         writer.add_image('images/{}_sim_stroke_train'.format(non_val_ind), 
    #                 np.clip(pred_stroke.detach().cpu().numpy()[0], a_min=0, a_max=1)*255, 10000)

    #     # t = trajectories[:1].detach().clone()
    #     # t.requires_grad = True
    #     # c_opt = torch.optim.Adam([t], lr=1e-3)
    #     # for j in range(30):
    #     #     c_opt.zero_grad()
    #     #     canvas = torch.zeros(strokes[:1].shape).to(device)
    #     #     for i in range(1000):
    #     #         canvas += best_model(t)
    #     #     loss = nn.L1Loss()(canvas, torch.rand(canvas.shape).to(device))
    #     #     loss.backward()
    #     #     c_opt.step()


    1/0 
    ############
    # optim = torch.optim.Adam(trans.parameters(), lr=1e-2)

    # for it in range(300):
    #     ep_loss = 0.
    #     for stroke_ind in range(len(strokes)):
    #         optim.zero_grad()

    #         stroke_real_params = all_strokes[stroke_ind]().trajectory
    #         x = torch.tensor(np.array(stroke_real_params).flatten()).float().to(device)
    #         #print('x', x)
    #         pred_stroke = trans(x)
    #         #print('stroke predict size', pred_stroke.shape, strokes[0].shape)

    #         real_stroke = strokes[stroke_ind][:,:,-1].unsqueeze(0)
    #         loss = nn.L1Loss()(pred_stroke, real_stroke)

    #         loss.backward()
    #         optim.step()
    #         ep_loss += loss.item()

    #         if it % 50 == 0:
    #             writer.add_image('images/{}_sim_stroke'.format(stroke_ind), 
    #                 np.clip(pred_stroke.detach().cpu().numpy()[0], a_min=0, a_max=1)*255, it)
    #     #print(ep_loss)
    #     writer.add_scalar('loss/train_loss', ep_loss, it)
