# import pickle
import numpy as np
import torch
from torchvision import transforms
from torch import nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import requests
# from PIL import Image
import io
# import cv2
from tqdm import tqdm
import os
# import PIL.Image, PIL.ImageDraw
# from io import BytesIO
# import lpips
import copy
# import colour
# import random
import gzip
# import kornia as K
# import datetime

from options import Options
from my_tensorboard import TensorBoard


opt = Options()
opt.gather_options()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

def special_sigmoid(x):
    # return 1/(1+torch.exp(-1.*((x*2-1)+0.2) / 0.05))
    return x

    # x[x < 0.1] = 1/(1+torch.exp(-1.*((x[x < 0.1]*2-1)+0.2) / 0.05))
    # return x

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn 
    return pp


def to_full_param(length, bend, z, alpha=0.0):
    full_param = torch.zeros((1,16)).to(device)
    
    # X
    full_param[0,0] = 0
    full_param[0,4] = length/3 
    full_param[0,8] = 2*length/3
    full_param[0,12] = length
    # Y
    full_param[0,1] = 0
    full_param[0,5] = bend
    full_param[0,9] = bend
    full_param[0,13] = 0
    # Z
    full_param[0,2] = 0.2
    full_param[0,6] = z
    full_param[0,10] = z
    full_param[0,14] = 0.2
    # alpha
    full_param[0,3] = alpha
    full_param[0,7] = alpha
    full_param[0,11] = alpha
    full_param[0,15] = alpha

    return full_param

def process_img(img):
    return np.clip(img.detach().cpu().numpy(), a_min=0, a_max=1)*255

def log_all_permutations(model, writer):
    # Length v Bend
    n_img = 5
    lengths = torch.linspace(opt.MIN_STROKE_LENGTH, opt.MAX_STROKE_LENGTH, steps=n_img)#torch.arange(n_img)/(n_img-1)*(opt.MAX_STROKE_LENGTH-opt.MIN_STROKE_LENGTH) + opt.MIN_STROKE_LENGTH
    bends = torch.linspace(-1.0*opt.MAX_BEND, opt.MAX_BEND, steps=n_img)#torch.arange(n_img)/(n_img-1)*(2*opt.MAX_BEND) - opt.MAX_BEND
    zs = torch.linspace(opt.MIN_STROKE_Z, 1.0, steps=n_img)#torch.arange(n_img)/(n_img-1)
    alphas = torch.linspace(-1.*opt.MAX_ALPHA, opt.MAX_ALPHA, steps=n_img)

    fig, ax = plt.subplots(n_img, n_img, figsize=(10,12))

    for i in range(n_img):
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], bends[j], 0.5)
            s = 1-model(trajectory)
            # print(s.shape)
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

            ax[i,j].imshow(s, cmap='gray')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    fig.tight_layout()
    # plt.show()
    writer.add_figure('images_stroke_modeling/stroke_modelling_bend_vs_length', fig, 0)

    fig, ax = plt.subplots(n_img, n_img, figsize=(10,12))

    for i in range(n_img):
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], 0.0, zs[j])
            s = 1-model(trajectory)
            # print(s.shape)
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

            ax[i,j].imshow(s, cmap='gray')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    fig.tight_layout()
    writer.add_figure('images_stroke_modeling/stroke_modelling_thickness_vs_length', fig, 0)
    # plt.show()

    whole_thing = []
    for i in range(n_img):
        row = []
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], 0.0, 0.5, alphas[j])
            s = 1-model(trajectory)
            # print(s.shape)
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
            row.append(s)
        whole_thing.append(np.concatenate(row, axis=1))
    whole_thing = np.concatenate(whole_thing, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(10,12))    
    ax.imshow(whole_thing, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    writer.add_figure('images_stroke_modeling/length_vs_alpha', fig, 0)

    img_fig = None
    for i in range(n_img):
        trajectory = to_full_param(.05, 0.0, zs[i])
        s = 1-model(trajectory)
        # print(s.shape)
        s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
        s = np.pad(s, (3), 'constant', constant_values=(1.))
        if img_fig is None:
            img_fig = s 
        else:
            img_fig = np.concatenate([img_fig, s], axis=1)
    plt.figure(figsize=(19,3))
    plt.imshow(img_fig, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('thickness.png')
    # plt.show()

    img_fig = None
    for i in range(n_img):
        trajectory = to_full_param(.05, bends[i], .5)
        s = 1-model(trajectory)
        # print(s.shape)
        s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
        s = np.pad(s, (3), 'constant', constant_values=(1.))
        if img_fig is None:
            img_fig = s 
        else:
            img_fig = np.concatenate([img_fig, s], axis=1)
    plt.figure(figsize=(19,3))
    plt.imshow(img_fig, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('bends.png')
    # plt.show()

    img_fig = None
    for i in range(n_img):
        trajectory = to_full_param(lengths[i], 0, .5)
        s = 1-model(trajectory)
        # print(s.shape)
        s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
        s = np.pad(s, (3), 'constant', constant_values=(1.))
        if img_fig is None:
            img_fig = s 
        else:
            img_fig = np.concatenate([img_fig, s], axis=1)
    plt.figure(figsize=(19,3))
    plt.imshow(img_fig, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('lengths.png')
    # plt.show()

    # fig, ax = plt.subplots(1, n_img, figsize=(12,1))
    # for i in range(n_img):
    #     trajectory = to_full_param(.04, 0.0, zs[i])
    #     s = 1-model(trajectory)
    #     # print(s.shape)
    #     s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

        

    #     ax[i].imshow(s, cmap='gray')
    #     ax[i].set_xticks([])
    #     ax[i].set_yticks([])
    # fig.tight_layout()
    # plt.show()
    # writer.add_figure('images_stroke_modeling/stroke_modelling_thickness', fig, 0)

    # fig, ax = plt.subplots(1, n_img, figsize=(12,1))
    # for i in range(n_img):
    #     trajectory = to_full_param(.04, bends[i], .5)
    #     s = 1-model(trajectory)
    #     # print(s.shape)
    #     s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

    #     ax[i].imshow(s, cmap='gray')
    #     ax[i].set_xticks([])
    #     ax[i].set_yticks([])
    # fig.tight_layout()
    # plt.show()
    # writer.add_figure('images_stroke_modeling/stroke_modelling_bend', fig, 0)

    # fig, ax = plt.subplots(1, n_img, figsize=(12,1))
    # for i in range(n_img):
    #     trajectory = to_full_param(lengths[i], 0, .5)
    #     s = 1-model(trajectory)
    #     # print(s.shape)
    #     s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

    #     ax[i].imshow(s, cmap='gray')
    #     ax[i].set_xticks([])
    #     ax[i].set_yticks([])
    # fig.tight_layout()
    # plt.show()
    # writer.add_figure('images_stroke_modeling/stroke_modelling_length', fig, 0)


def log_images(imgs, labels, label, writer, step=0):
    fig, ax = plt.subplots(1, len(imgs), figsize=(5*len(imgs),5))

    for i in range(len(imgs)):
        # print(imgs[i].min(), imgs[i].max())
        ax[i].imshow(imgs[i], cmap='gray', vmin=0, vmax=255)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(labels[i])
    fig.tight_layout()
    writer.add_figure(label, fig, step)

class StrokeParametersToImage(nn.Module):
    def __init__(self, h, w):
        super(StrokeParametersToImage, self).__init__()
        nh = 20
        self.nc = 20
        self.size = 48#64
        self.size_x = 64#48
        self.size_y = 32#24
        self.main = nn.Sequential(
            nn.BatchNorm1d(16),
            #
            nn.Linear(16, nh),
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
            # nn.Linear(nh, 64*64),
            nn.Linear(nh, self.size_x*self.size_y),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.nc, kernel_size=5, padding='same', dilation=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.nc),
            nn.Conv2d(self.nc, 1, kernel_size=5, padding='same', dilation=1),
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
        # x = self.res2(self.conv((self.main(x).view(-1, 1, 64, 64))))[:,0]
        x = self.res2(self.conv((self.main(x).view(-1, 1, self.size_y, self.size_x))))[:,0]
        # x = self.res2(self.conv((self.main(x).view(-1, 1, 32, 32))))[:,0]
        # x = 1/(1+torch.exp(-1.*(x*2-1) / 0.05))
        return x

    def forward_full(self, x):
        x = self.forward(x)
        return transforms.Pad((ws, hs, w_og-we, h_og-he))(x)

loss_fcn = nn.L1Loss()#nn.MSELoss()
def custom_loss(pred, real):
    # Weighted difference by where the stroke is present
    # Guides model to make bigger strokes because it cares less about non-stroke regions
    diff = torch.abs(pred - real)
    # loss = (torch.abs(real)+1e-2) * diff
    #loss = (torch.abs(real)+.4) * diff
    loss = diff
    weight_loss = (pred.mean(dim=[1,2]) - real.mean(dim=[1,2])).abs().mean()
    return loss.mean() + weight_loss# + (pred.mean() - real.mean())**2
loss_fcn = custom_loss

def train_param2stroke(opt):
    #strokes = np.load(os.path.join(opt.cache_dir, 'extended_stroke_library_intensities.npy')).astype(np.float32)/255.
    with gzip.GzipFile(os.path.join(opt.cache_dir, 'extended_stroke_library_intensities.npy'),'r') as f:
        strokes = np.load(f).astype(np.float32)/255.
    trajectories = np.load(os.path.join(opt.cache_dir, 'extended_stroke_library_trajectories.npy'), 
            allow_pickle=True, encoding='bytes') 

    strokes = torch.from_numpy(strokes).float().nan_to_num()
    trajectories = torch.from_numpy(trajectories.astype(np.float32)).float().nan_to_num()
    
    n = len(strokes)
    
    # scale_factor = opt.max_height / strokes.shape[1]
    stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
    h, w = stroke_shape[0], stroke_shape[1]
    # strokes = transforms.Resize((int(strokes.shape[1]*scale_factor), int(strokes.shape[2]*scale_factor)))(strokes)
    strokes = transforms.Resize((h,w))(strokes)

    # Randomize
    rand_ind = torch.randperm(strokes.shape[0])
    strokes = strokes[rand_ind]
    trajectories = trajectories[rand_ind]

    # Discrete. Makes the model push towards making bolder strokes
    # strokes[strokes >= 0.5] = 1.
    # strokes[strokes < 0.5] = 0.

    # h, w = strokes[0].shape[0], strokes[0].shape[1]
    h_og, w_og = h, w

    # Crop
    hs, he = int(.4*h), int(0.6*h)
    ws, we = int(0.45*w), int(0.75*w)
    strokes = strokes[:, hs:he, ws:we]

    for i in range(len(strokes)):
        strokes[i] -= strokes[i].min()
        if strokes[i].max() > 0.01:
            strokes[i] /= strokes[i].max()
        # strokes[i] *= 0.95

    h, w = strokes[0].shape[0], strokes[0].shape[1]
    ##########FOR SPEED#########################################
    scale_factor = 4
    strokes = transforms.Resize((int(h/scale_factor), int(w/scale_factor)))(strokes) 
    h, w = strokes[0].shape[0], strokes[0].shape[1]

    strokes = strokes.to(device)
    trajectories = trajectories.to(device)

    for model_ind in range(opt.n_stroke_models):
        trans = StrokeParametersToImage(h,w).to(device)
        print('# parameters in StrokeParam2Image model:', get_n_params(trans))
        optim = torch.optim.Adam(trans.parameters(), lr=1e-3)
        best_model = None
        best_val_loss = 999
        best_hasnt_changed_for = 0

        val_prop = .2

        train_strokes = strokes[int(val_prop*n):]
        train_trajectories = trajectories[int(val_prop*n):]

        val_strokes = strokes[:int(val_prop*n)]
        val_trajectories = trajectories[:int(val_prop*n)]
        print('{} training strokes. {} validation strokes'.format(len(train_strokes), len(val_strokes)))

        for it in tqdm(range(2000)):
            if best_hasnt_changed_for >= 200:
                break # all done :)
            optim.zero_grad()

            noise = torch.randn(train_trajectories.shape).to(device)*0.001 # For robustness
            pred_strokes = trans(train_trajectories + noise)

            loss = loss_fcn(pred_strokes, train_strokes)
            # loss = nn.MSELoss()(pred_strokes, train_strokes) # MSE loss produces crisper stroke images

            ep_loss = loss.item()
            loss.backward()
            optim.step()

            opt.writer.add_scalar('loss/train_loss_stroke_model', ep_loss, it)
            
            n_view = 10
            if it % 5 == 0:
                with torch.no_grad():
                    trans.eval()
                    pred_strokes_val = trans(val_trajectories)
                    # if it % 50 == 0:
                    #     for val_ind in range(min(n_view,len(val_strokes))):
                    #         log_images([process_img(val_strokes[val_ind]),
                    #             process_img(pred_strokes_val[val_ind])], 
                    #             ['real','sim'], 'images_stroke_modeling/val_{}_stroke'.format(val_ind), opt.writer, step=it)
                    #     pred_strokes_train = trans(train_trajectories)
                    #     for train_ind in range(min(n_view,len(train_strokes))):
                    #         log_images([process_img(train_strokes[train_ind]),
                    #             process_img(pred_strokes_train[train_ind])], 
                    #             ['real','sim'], 'images_stroke_modeling/train_{}_stroke'.format(train_ind), opt.writer, step=it)

                    # loss = nn.MSELoss()(pred_strokes_val, val_strokes)
                    loss = loss_fcn(pred_strokes_val, val_strokes)
                    if it % 15 == 0: 
                        opt.writer.add_scalar('loss/val_loss_stroke_model', loss.item(), it)
                    if loss.item() < best_val_loss and it > 50:
                        best_val_loss = loss.item()
                        best_hasnt_changed_for = 0
                        best_model = copy.deepcopy(trans)
                    best_hasnt_changed_for += 5
                    trans.train()

        if model_ind == 0:
            with torch.no_grad():
                best_model.eval()
                pred_strokes_val = best_model(val_trajectories)
                for val_ind in range(min(n_view,len(val_strokes))):
                    # print(val_trajectories[val_ind])
                    b, l, z, alpha = val_trajectories[val_ind][5], val_trajectories[val_ind][12], val_trajectories[val_ind][6], val_trajectories[val_ind][-1]
                    log_images([process_img(1-val_strokes[val_ind]),
                        process_img(1-special_sigmoid(pred_strokes_val[val_ind]))], 
                        ['real','sim'], 'images_stroke_modeling/val_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}_alph{:.2f}'.format(val_ind, b, l, z, alpha), opt.writer)

                pred_strokes_train = best_model(train_trajectories)
                for train_ind in range(min(n_view,len(train_strokes))):
                    b, l, z, alpha = train_trajectories[train_ind][5], train_trajectories[train_ind][12], train_trajectories[train_ind][6], train_trajectories[train_ind][-1]
                    log_images([process_img(1-train_strokes[train_ind]),
                        process_img(1-special_sigmoid(pred_strokes_train[train_ind]))], 
                        ['real','sim'], 'images_stroke_modeling/train_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}_alph{:.2f}'.format(train_ind, b, l, z, alpha), opt.writer)

                # # Make them into full sized strokes
                # best_model.res2 = transforms.Resize((h_og, w_og))
                # pred_strokes_val = best_model(val_trajectories)
                # pred_strokes_val = special_sigmoid(pred_strokes_val)
                # pred_strokes_val = transforms.Pad((ws, hs, w_og-we, h_og-he))(pred_strokes_val)
                # val_strokes_full = transforms.Pad((ws, hs, w_og-we, h_og-he))(best_model.res2(val_strokes))
                # for val_ind in range(min(n_view,len(val_strokes))):
                #     log_images([process_img(1-val_strokes_full[val_ind]),
                #         process_img(1-pred_strokes_val[val_ind])], 
                #         ['real','sim'], 'images_stroke_modeling/val_{}_stroke_full'.format(val_ind), opt.writer)
            log_all_permutations(best_model, opt.writer)
        torch.save(best_model.cpu().state_dict(), os.path.join(opt.cache_dir, 'param2img{}.pt'.format(model_ind)))

    return h_og, w_og

def n_stroke_test(opt):
    #strokes = np.load(os.path.join(opt.cache_dir, 'extended_stroke_library_intensities.npy')).astype(np.float32)/255.
    with gzip.GzipFile(os.path.join(opt.cache_dir, 'extended_stroke_library_intensities.npy'),'r') as f:
        strokes = np.load(f).astype(np.float32)/255.
    trajectories = np.load(os.path.join(opt.cache_dir, 'extended_stroke_library_trajectories.npy'), 
            allow_pickle=True, encoding='bytes') 

    strokes = torch.from_numpy(strokes).to(device).float().nan_to_num()
    trajectories = torch.from_numpy(trajectories.astype(np.float32)).to(device).float().nan_to_num()

    # Randomize
    rand_ind = torch.randperm(strokes.shape[0])
    strokes = strokes[rand_ind]
    trajectories = trajectories[rand_ind]

    stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
    h, w = stroke_shape[0], stroke_shape[1]
    strokes = transforms.Resize((h,w))(strokes)

    h_og, w_og = h, w

    # Crop
    hs, he = int(.4*h), int(0.6*h)
    ws, we = int(0.45*w), int(0.75*w)
    strokes = strokes[:, hs:he, ws:we]




    n = len(strokes)
    print(n)
    test_prop = 0.2 
    test_strokes = strokes[:int(test_prop*n)]
    test_trajectories = trajectories[:int(test_prop*n)]
    strokes = strokes[int(test_prop*n):]
    trajectories = trajectories[int(test_prop*n):]
    n = len(strokes)
    print(n, 'test strokes', len(test_strokes), len(strokes))

    disp_traj = to_full_param(0.04, 0.02, .5)



    h, w = strokes[0].shape[0], strokes[0].shape[1]

    n = len(strokes)

    plot_x = []
    plot_y = []

    plot_x_std = []
    plot_y_std = []
    
    for n_strokes in range(15, n, 5):
        s = strokes[:n_strokes]
        t = trajectories[:n_strokes]



        avg_test_err = 0
        test_err = []
        n_folds = 5
        for fold in range(n_folds):
            trans = StrokeParametersToImage(h,w).to(device)
            #print('# parameters in StrokeParam2Image model:', get_n_params(trans))
            optim = torch.optim.Adam(trans.parameters(), lr=1e-3)
            best_model = None
            best_val_loss = 999
            best_hasnt_changed_for = 0

            #val_prop = .3

            train_strokes = torch.cat([s[:int((fold/n_folds)*n_strokes)], s[int(((fold+1)/n_folds)*n_strokes):]], dim=0)
            train_trajectories = torch.cat([t[:int((fold/n_folds)*n_strokes)], t[int(((fold+1)/n_folds)*n_strokes):]], dim=0)
            val_strokes = s[int((fold/n_folds)*n_strokes):int(((fold+1)/n_folds)*n_strokes)]
            val_trajectories = t[int((fold/n_folds)*n_strokes):int(((fold+1)/n_folds)*n_strokes)]
            if fold == 0:
                print('{} training strokes. {} validation strokes'.format(len(train_strokes), len(val_strokes)))

            for it in tqdm(range(2000)):
                if best_hasnt_changed_for >= 200:
                    break # all done :)
                optim.zero_grad()

                noise = torch.randn(train_trajectories.shape).to(device)*0.005 # For robustness
                pred_strokes = trans(train_trajectories + noise)

                loss = loss_fcn(pred_strokes, train_strokes)
                # loss = nn.MSELoss()(pred_strokes, train_strokes) # MSE loss produces crisper stroke images

                ep_loss = loss.item()
                loss.backward()
                optim.step()

                opt.writer.add_scalar('loss/train_loss_stroke_model', ep_loss, it)
                
                n_view = 10
                with torch.no_grad():
                    trans.eval()
                    pred_strokes_val = trans(val_trajectories)
                    # loss = nn.MSELoss()(pred_strokes_val, val_strokes)
                    loss = loss_fcn(pred_strokes_val, val_strokes)
                    # print(pred_strokes_val.shape, val_strokes.shape)
                    opt.writer.add_scalar('loss/val_loss_stroke_model', loss.item(), it)

                    if loss.item() < best_val_loss and it > 50:
                        best_val_loss = loss.item()
                        best_hasnt_changed_for = 0
                        best_model = copy.deepcopy(trans)
                    best_hasnt_changed_for += 1
                    trans.train()
            with torch.no_grad():
                trans.eval()
                pred_strokes_test = best_model(test_trajectories)
                # loss = nn.MSELoss()(pred_strokes_test, test_strokes)
                test_loss = loss_fcn(pred_strokes_test, test_strokes)
                
                trans.train()

            with torch.no_grad():
                trans.eval()# The display model
                import matplotlib.pyplot as plt 
                fig, ax = plt.subplots(1)
                disp_stroke = 1 - best_model(disp_traj)
                disp_stroke = np.clip(disp_stroke.detach().cpu().numpy()[0], a_min=0, a_max=1)
                ax.imshow(disp_stroke, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()
                writer.add_figure('limit_data_stroke_model/n_strokes{}'.format(n_strokes), fig, fold)
                trans.train()
            avg_test_err += test_loss.item()/n_folds
            test_err.append(test_loss.item())
        print('test error', avg_test_err)
        opt.writer.add_scalar('loss/test_loss_stroke_model', avg_test_err, n_strokes)
        test_err_std = np.array(test_err).std()
        opt.writer.add_scalar('loss/test_loss_std_stroke_model', test_err_std, n_strokes)
        
        plot_x.append(n_strokes)
        plot_y.append(avg_test_err)
        plot_x_std.append(n_strokes)
        plot_y_std.append(test_err_std)

        import matplotlib.pyplot as plt 
        plt.rcParams["font.family"] = "Times New Roman"
        # plot_x = [1,2,3]
        # plot_y = [.1,.05, .01]
        plt.plot(plot_x, plot_y)
        csfont = {'fontname':'Times New Roman'}
        plt.title('Stroke Shape Model Performance Versus Number of Training Examples')
        plt.ylabel('Average Absolute Test Error')
        plt.xlabel('Number of Training and Validation Brush Strokes')
        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig('err_v_strokes.svg', format='svg')
        plt.show()

        fig, ax = plt.subplots()

        ax.plot(plot_x, plot_y, color='red')
        ax.set_yscale('log')
        ax.tick_params(axis='y', labelcolor='red')

        ax2 = ax.twinx()
        ax2.plot(plot_x_std, plot_y_std, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax.tick_params(axis='y', labelcolor='red')
        plt.savefig('err_v_strokes_with_std.svg', format='svg')
        plt.show()


        fig, ax = plt.subplots()

        ax.plot(plot_x, plot_y, 'brown')
        ax.fill_between(plot_x, np.array(plot_y) - np.array(plot_y_std), np.array(plot_y) + np.array(plot_y_std),
            color='lightsalmon')
        ax.set_yscale('log')
        ax.set_title('Stroke Shape Model Performance Versus Number of Training Examples')
        ax.set_ylabel('Log Average Absolute Test Error')
        ax.set_xlabel('Number of Training and Validation Brush Strokes')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax2.tick_params(axis='y', labelcolor='green')
        # ax.tick_params(axis='y', labelcolor='red')
        plt.savefig('err_v_strokes_with_std2.svg', format='svg')
        plt.show()


    return h_og, w_og

if __name__ == '__main__':
    # global opt, strokes
    opt = Options()
    opt.gather_options()

    b = './painting'
    all_subdirs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
    tensorboard_dir = max(all_subdirs, key=os.path.getmtime) # most recent tensorboard dir is right
    if '_planner' not in tensorboard_dir:
        tensorboard_dir += '_planner'
    writer = TensorBoard(tensorboard_dir)
    opt.writer = writer

    # n_stroke_test(opt)
    train_param2stroke(opt)