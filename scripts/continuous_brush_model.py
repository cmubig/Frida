# import pickle
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
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
import kornia as K
import datetime
from tqdm import tqdm

from options import Options
from tensorboard import TensorBoard


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

def special_sigmoid(x):
    # return 1/(1+torch.exp(-1.*((x*2-1)+0.2) / 0.05))
    # return x

    x[x < 0.1] = 1/(1+torch.exp(-1.*((x[x < 0.1]*2-1)+0.2) / 0.05))
    return x

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

def process_img(img):
    return np.clip(img.detach().cpu().numpy(), a_min=0, a_max=1)*255

def log_all_permutations(model, writer):
    # Length v Bend
    n_img = 7
    lengths = torch.arange(n_img)/(n_img-1)*(0.05-0.01) + 0.01
    bends = torch.arange(n_img)/(n_img-1)*0.04 - 0.02
    zs = torch.arange(n_img)/(n_img-1)

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

    img_fig = None
    for i in range(n_img):
        trajectory = to_full_param(.04, 0.0, zs[i])
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
        trajectory = to_full_param(.04, bends[i], .5)
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
        ax[i].imshow(imgs[i], cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(labels[i])
    fig.tight_layout()
    writer.add_figure(label, fig, step)

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
            # nn.Linear(nh, 64*64),
            nn.Linear(nh, 48*48),
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
        x = self.res2(self.conv((self.main(x).view(-1, 1, 48, 48))))[:,0]
        # x = 1/(1+torch.exp(-1.*(x*2-1) / 0.05))
        return x

    def forward_full(self, x):
        x = self.forward(x)
        return transforms.Pad((ws, hs, w_og-we, h_og-he))(x)



def train_param2stroke(opt):
    #strokes = np.load(os.path.join(opt.cache_dir, 'extended_stroke_library_intensities.npy')).astype(np.float32)/255.
    with gzip.GzipFile(os.path.join(opt.cache_dir, 'extended_stroke_library_intensities.npy'),'r') as f:
        strokes = np.load(f).astype(np.float32)/255.
    trajectories = np.load(os.path.join(opt.cache_dir, 'extended_stroke_library_trajectories.npy'), 
            allow_pickle=True, encoding='bytes') 

    strokes = torch.from_numpy(strokes).to(device).float().nan_to_num()
    trajectories = torch.from_numpy(trajectories.astype(np.float32)).to(device).float().nan_to_num()
    
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

    h, w = strokes[0].shape[0], strokes[0].shape[1]

    for model_ind in range(opt.n_stroke_models):
        trans = StrokeParametersToImage(h,w).to(device)
        print('# parameters in StrokeParam2Image model:', get_n_params(trans))
        optim = torch.optim.Adam(trans.parameters(), lr=1e-3)
        best_model = None
        best_val_loss = 999
        best_hasnt_changed_for = 0

        val_prop = .3

        train_strokes = strokes[int(val_prop*n):]
        train_trajectories = trajectories[int(val_prop*n):]
        val_strokes = strokes[:int(val_prop*n)]
        val_trajectories = trajectories[:int(val_prop*n)]
        print('{} training strokes. {} validation strokes'.format(len(train_strokes), len(val_strokes)))

        for it in tqdm(range(2000)):
            if best_hasnt_changed_for >= 200:
                break # all done :)
            optim.zero_grad()

            noise = torch.randn(train_trajectories.shape).to(device)*0.005 # For robustness
            pred_strokes = trans(train_trajectories + noise)

            loss = nn.L1Loss()(pred_strokes, train_strokes)
            # loss = nn.MSELoss()(pred_strokes, train_strokes) # MSE loss produces crisper stroke images

            ep_loss = loss.item()
            loss.backward()
            optim.step()

            opt.writer.add_scalar('loss/train_loss_stroke_model', ep_loss, it)
            
            n_view = 10
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
                loss = nn.L1Loss()(pred_strokes_val, val_strokes)
                opt.writer.add_scalar('loss/val_loss_stroke_model', loss.item(), it)
                if loss.item() < best_val_loss and it > 50:
                    best_val_loss = loss.item()
                    best_hasnt_changed_for = 0
                    best_model = copy.deepcopy(trans)
                best_hasnt_changed_for += 1
                trans.train()

        if model_ind == 0:
            with torch.no_grad():
                best_model.eval()
                pred_strokes_val = best_model(val_trajectories)
                for val_ind in range(min(n_view,len(val_strokes))):
                    # print(val_trajectories[val_ind])
                    b, l, z = val_trajectories[val_ind][4], val_trajectories[val_ind][9], val_trajectories[val_ind][5]
                    log_images([process_img(1-val_strokes[val_ind]),
                        process_img(1-special_sigmoid(pred_strokes_val[val_ind]))], 
                        ['real','sim'], 'images_stroke_modeling/val_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}'.format(val_ind, b, l, z), opt.writer)

                pred_strokes_train = best_model(train_trajectories)
                for train_ind in range(min(n_view,len(train_strokes))):
                    b, l, z = train_trajectories[train_ind][4], train_trajectories[train_ind][9], train_trajectories[train_ind][5]
                    log_images([process_img(1-train_strokes[train_ind]),
                        process_img(1-special_sigmoid(pred_strokes_train[train_ind]))], 
                        ['real','sim'], 'images_stroke_modeling/train_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}'.format(train_ind, b, l, z), opt.writer)

                # Make them into full sized strokes
                pred_strokes_val = best_model(val_trajectories)
                pred_strokes_val = special_sigmoid(pred_strokes_val)
                pred_strokes_val = transforms.Pad((ws, hs, w_og-we, h_og-he))(pred_strokes_val)
                val_strokes_full = transforms.Pad((ws, hs, w_og-we, h_og-he))(val_strokes)
                for val_ind in range(min(n_view,len(val_strokes))):
                    log_images([process_img(1-val_strokes_full[val_ind]),
                        process_img(1-pred_strokes_val[val_ind])], 
                        ['real','sim'], 'images_stroke_modeling/val_{}_stroke_full'.format(val_ind), opt.writer)
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


    h, w = strokes[0].shape[0], strokes[0].shape[1]

    n = len(strokes)

    plot_x = []
    plot_y = []
    
    for n_strokes in range(10, n, 5):
        s = strokes[:n_strokes]
        t = trajectories[:n_strokes]


        avg_test_err = 0
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

                loss = nn.L1Loss()(pred_strokes, train_strokes)
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
                    loss = nn.L1Loss()(pred_strokes_val, val_strokes)
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
                test_loss = nn.L1Loss()(pred_strokes_test, test_strokes)
                
                trans.train()
            avg_test_err += test_loss.item()/n_folds
        print('test error', avg_test_err)
        opt.writer.add_scalar('loss/test_loss_stroke_model', avg_test_err, n_strokes)

        plot_x.append(n_strokes)
        plot_y.append(avg_test_err)

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


    return h_og, w_og

if __name__ == '__main__':
    global opt, strokes
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