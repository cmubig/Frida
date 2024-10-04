import glob
import os
import numpy as np
import gzip
import torch
import pickle
import sys
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import wandb

sys.path.append('../')
from brush_stroke import BrushStroke

''' Read and preprocess trajectories and canvas befores/afters '''
# cache_dir = '../caches/vae_brush_final'
cache_dir = '../caches/vae_sharpie_final'
canvases_before_fns = glob.glob(os.path.join(cache_dir, 'stroke_library', 'canvases_before_*.npy'))
canvases_after_fns = glob.glob(os.path.join(cache_dir, 'stroke_library', 'canvases_after_*.npy'))
brush_strokes_fns = glob.glob(os.path.join(cache_dir, 'stroke_library', 'stroke_parameters*.npy'))

# Ensure each of the files is in the order they were recorded in
canvases_before_fns = sorted(canvases_before_fns)
canvases_after_fns = sorted(canvases_after_fns)
brush_strokes_fns = sorted(brush_strokes_fns)

# Load data
canvases_before = None
for canvases_before_fn in canvases_before_fns:
    with gzip.GzipFile(canvases_before_fn,'r') as f:
        s = np.load(f, allow_pickle=True).astype(np.float32)/255.
        canvases_before = s if canvases_before is None else np.concatenate([canvases_before, s])

canvases_after = None
for canvases_after_fn in canvases_after_fns:
    with gzip.GzipFile(canvases_after_fn,'r') as f:
        s = np.load(f, allow_pickle=True).astype(np.float32)/255.
        canvases_after = s if canvases_after is None else np.concatenate([canvases_after, s])

brush_strokes = []
for brush_strokes_fn in brush_strokes_fns:
    bs = pickle.load(open(brush_strokes_fn,'rb'))
    brush_strokes = bs if brush_strokes is None else np.concatenate([brush_strokes, bs]) 
for b in brush_strokes:
    b.vae_name = os.path.join('..', b.vae_name)

canvases_before = torch.from_numpy(canvases_before).float().nan_to_num()
canvases_after = torch.from_numpy(canvases_after).float().nan_to_num()
canvases_before = torch.mean(canvases_before, dim=3)
canvases_after = torch.mean(canvases_after, dim=3)

with open(os.path.join(cache_dir, 'stroke_library', 'stroke_settings_during_library.json'), 'r') as f:
    settings = json.load(f)
    CANVAS_WIDTH_M = settings['CANVAS_WIDTH_M']
    CANVAS_HEIGHT_M = settings['CANVAS_HEIGHT_M']

paths = []
starts = []
for b in brush_strokes:
    path = b.get_path()[:,:2]
    path = torch.flip(path, dims=(1,)) 
    path[:, 0] *= -1
    starts.append([b.yt, b.xt])
    paths.append(path.detach())
starts = torch.tensor(starts)
paths = torch.stack(paths, dim=0)

'''Preprocess data so that strokes start at center of image'''
def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def shift_to_center(path, start, canvas_before, canvas_after):
    path = path.clone().numpy()
    start = start.clone().numpy()
    canvas_before = canvas_before.clone().numpy()
    canvas_after = canvas_after.clone().numpy()

    shift = 0.5 - start
    h, w = canvas_before.shape
    canvas_shift = np.array([shift[0] * h, shift[1] * w])
    canvas_shift = np.round(canvas_shift).astype(int)
    canvas_before = shift_image(canvas_before, canvas_shift[1], canvas_shift[0])
    canvas_after = shift_image(canvas_after, canvas_shift[1], canvas_shift[0])

    path[:, 0] /= CANVAS_HEIGHT_M
    path[:, 1] /= CANVAS_WIDTH_M
    path += 0.5
    path = np.flip(path, axis=1)

    return path, canvas_before, canvas_after

xs = []
ys = []
for i in range(len(paths)):
    path, canvas_before, canvas_after = shift_to_center(paths[i], starts[i], canvases_before[i], canvases_after[i])
    path = torch.from_numpy(path.copy())
    canvas_before = torch.from_numpy(canvas_before.copy())
    canvas_after = torch.from_numpy(canvas_after.copy())
    xs.append(path)
    ys.append((canvas_before, canvas_after))


''' Create datasets '''
num_val = 10
train_xs = xs[:-num_val]
train_ys = ys[:-num_val]
test_xs = xs[-num_val:]
test_ys = ys[-num_val:]

class StrokeDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

train_dataset = StrokeDataset(train_xs, train_ys)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = StrokeDataset(test_xs, test_ys)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


''' Training '''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_renderer(renderer, dataloader):
    for name, param in renderer.named_parameters():
        if 'radius' in name:
            print(f"radius: {param.data}")
        elif 'dark_mult' in name:
            print(f"dark_mult: {param.data}")
        elif 'dark_exp' in name:
            print(f"dark_exp: {param.data}")
    with torch.no_grad():
        renderer.eval()

        tot_loss = 0
        cnt = 0
        results = []
        for traj, (canvas_before, canvas_after) in dataloader:
            traj = traj.to(device)
            canvas_before = canvas_before.to(device)
            canvas_after = canvas_after.to(device)

            pred = renderer(traj)
            pred = transforms.Resize((canvas_after.shape[1], canvas_after.shape[2]), antialias=True)(pred)
            pred = pred + canvas_before*(1-pred)
            print("canvas_before", torch.min(canvas_before), torch.max(canvas_before))
            print("pred", torch.min(pred), torch.max(pred))
            print("canvas_after", torch.min(canvas_after), torch.max(canvas_after))

            for i in range(len(pred)):
                results.append((pred[i].detach().cpu().numpy(), canvas_after[i].detach().cpu().numpy()))

            loss = torch.mean((pred - canvas_after)**2)
            tot_loss += loss*len(pred)
            cnt += len(pred)
        loss = tot_loss / cnt
        
        renderer.train()
        return loss, results

def train(renderer, train_dataloader, test_dataloader, epochs, lr, eval_freq, finetune_epochs=None):
    def eval():
        loss, results = evaluate_renderer(renderer, test_dataloader)
        wandb.log({'test_loss': loss, 'images': [wandb.Image(np.concatenate((r[0], r[1]), axis=1)) for r in results]})
    
    if finetune_epochs is not None:
        for name, param in renderer.named_parameters():
            if 'zero_conv' in name:
                param.requires_grad = False

    optim = torch.optim.Adam(renderer.parameters(), lr=lr)
    for epoch in range(epochs):
        if epoch % eval_freq == 0:
            eval()

        for traj, (canvas_before, canvas_after) in train_dataloader:
            traj = traj.to(device)
            canvas_before = canvas_before.to(device)
            canvas_after = canvas_after.to(device)

            pred = renderer(traj)
            pred = transforms.Resize((canvas_after.shape[1], canvas_after.shape[2]), antialias=True)(pred)
            pred = pred + canvas_before*(1-pred)

            loss = torch.mean((pred - canvas_after)**2)
            optim.zero_grad()
            loss.backward()
            optim.step()
    eval()

    if finetune_epochs is not None:
        for name, param in renderer.named_parameters():
            if 'zero_conv' in name:
                param.requires_grad = True
        for epoch in range(finetune_epochs):
            if epoch % eval_freq == 0:
                eval()

            for traj, (canvas_before, canvas_after) in train_dataloader:
                traj = traj.to(device)
                canvas_before = canvas_before.to(device)
                canvas_after = canvas_after.to(device)

                pred = renderer(traj)
                pred = transforms.Resize((canvas_after.shape[1], canvas_after.shape[2]), antialias=True)(pred)
                pred = pred + canvas_before*(1-pred)

                loss = torch.mean((pred - canvas_after)**2)
                optim.zero_grad()
                loss.backward()
                optim.step()
        eval()

from custom_renderer import CustomRenderer
from conv_renderer import ConvRenderer
from coordconv_renderer import CoordConvRenderer
# from diffvg_renderer import DiffVGRenderer
from custom_unet_renderer import CustomUnetRenderer

configs = [
    {
        'renderer': CustomRenderer(256),
        'epochs': 500,
        'lr': 1e-2,
        'eval_freq': 10
    },
    {
        'renderer': ConvRenderer(),
        'epochs': 5000,
        'lr': 1e-3,
        'eval_freq': 100
    },
    {
        'renderer': CoordConvRenderer(),
        'epochs': 5000,
        'lr': 1e-3,
        'eval_freq': 100
    },
    {
        'renderer': CustomUnetRenderer(),
        'epochs': 40,
        'lr': 1e-3,
        'eval_freq': 10,
        'finetune_epochs': 500
    },
    # {
    #     'renderer': DiffVGRenderer(256),
    #     'epochs': 40,
    #     'lr': 1e-1
    # }
]

idx = 1
config = configs[idx]
renderer = config['renderer']
renderer.to(device)
epochs = config['epochs']
lr = config['lr']
eval_freq = config['eval_freq']
finetune_epochs = config['finetune_epochs'] if 'finetune_epochs' in config else None

wandb.init(project="stroke-model-comparisons", config=config)
train(renderer, train_dataloader, test_dataloader, epochs, lr, eval_freq, finetune_epochs)
