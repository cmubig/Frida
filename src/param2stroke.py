import glob
import json
import numpy as np
import pickle
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, Resize

from brush_stroke import BrushStroke
bicubic = InterpolationMode.BICUBIC
from torch import nn

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import gzip
from torchvision.transforms.functional import affine

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import matplotlib
def show_img(img, display_actual_size=True, title=""):
    if type(img) is torch.Tensor:
        img = img.detach().cpu().numpy()

    img = img.squeeze()
    if img.shape[0] < 5:
        img = img.transpose(1,2,0)

    if img.max() > 4:
        img = img / 255.
    img = np.clip(img, a_min=0, a_max=1)

    if display_actual_size:
        # Display at actual size: https://stackoverflow.com/questions/60144693/show-image-in-its-original-resolution-in-jupyter-notebook
        # Acquire default dots per inch value of matplotlib
        dpi = matplotlib.rcParams['figure.dpi']
        # Determine the figures size in inches to fit your image
        height, width = img.shape[0], img.shape[1]
        figsize = width / float(dpi), height / float(dpi)

        plt.figure(figsize=figsize)

    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.title(title)
    # plt.scatter(img.shape[1]/2, img.shape[0]/2)
    plt.show()


class MatMulLayer(nn.Module):
    def __init__(self, size, layers, activation=torch.nn.LeakyReLU):
        super(MatMulLayer, self).__init__()
        self.activation = activation()
        self.layers = layers
        self.weights = torch.nn.Parameter(torch.ones((layers,)+size, dtype=torch.float))
        self.biases = torch.nn.Parameter(torch.zeros((layers,)+size, dtype=torch.float))
    def forward(self, x):
        for i in range(self.layers):
            x = x * self.weights[i] + self.biases[i]
            if i != (self.layers-1):
                x = self.activation(x)
        return x

class BezierRenderer(nn.Module):
    def __init__(self, size_x, size_y, num_pts):
        super(BezierRenderer, self).__init__()
        self.size_x = size_x # grid dimensions (size_x x size_y)
        self.size_y = size_y
        self.P = num_pts
        # Physical canvas width / height. Distances are measured in canvas-height
        # units so a stroke's width doesn't depend on its direction.
        self.aspect = 1.0

        self.set_render_size(size_x, size_y)

        self.thick_mult = nn.Parameter(torch.ones((1)))
        self.thick_bias = nn.Parameter(torch.zeros((1))+0.1)
        self.dark_exp = nn.Parameter(torch.ones((1)))

    def set_render_size(self, size_x, size_y):
        '''
        Set the internal dimensions that strokes should be rendered at.
        grid_coords is registered as a non-persistent buffer so .to(device) moves it once.
        Shape: (1, 1, size_y, size_x, 2) for broadcast with batched segment endpoints.
        '''
        old_grid = self._buffers.get('grid_coords')
        device = old_grid.device if old_grid is not None else None
        self.size_x = size_x 
        self.size_y = size_y
        idxs_x = torch.arange(size_x, dtype=torch.float32, device=device) / size_x
        idxs_y = torch.arange(size_y, dtype=torch.float32, device=device) / size_y
        x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # G x G
        grid = torch.stack((x_coords, y_coords), dim=-1)  # (size_y, size_x, 2)
        self.register_buffer(
            'grid_coords',
            grid.view(1, 1, size_y, size_x, 2),
            persistent=False,
        )

    def set_aspect(self, aspect):
        self.aspect = float(aspect)

    def forward(self, trajectories, heights, use_conv=False, grid=None):
        # trajectories: (B, 2, self.P)
        # heights: (B, 1, self.P)
        # grid: optional (B or 1, 1, Hy, Wx, 2) canvas proportions (y, x) to render at
        # use_conv kept for API compatibility; soft distance-field path only.
        stroke = trajectories.permute(0, 2, 1)  # (B, P, 2)
        heights_ = (heights * 2 + 0.5) / 64
        return self.render_stroke(stroke, heights_, grid=grid)

    def curve_to_stroke(self, curve):
        # curve: (2, self.P)
        return curve.T

    def render_stroke(self, stroke, heights, grid=None):
        '''
        Soft distance-field rasterization of polyline strokes.

        stroke: (P, 2) or (B, P, 2) in [0,1] canvas proportions
        heights: (1, P) or (B, 1, P)
        grid: optional (B or 1, 1, Hy, Wx, 2) canvas proportions; defaults to the full canvas
        returns: (Hy, Wx) or (B, Hy, Wx)

        Uses broadcasting (no torch.tile). Keep B modest when grads are enabled:
        intermediates are (B, P-1, H, W).
        '''
        squeeze_batch = stroke.dim() == 2
        if squeeze_batch:
            stroke = stroke.unsqueeze(0)
            if heights.dim() == 2:
                heights = heights.unsqueeze(0)

        if grid is None:
            grid = self.grid_coords
        # Proportions -> canvas-height units (x scaled by width / height)
        scale = stroke.new_tensor([1.0, self.aspect])
        stroke = stroke * scale
        grid = grid * scale

        # Endpoints (B, P-1, 1, 1, 2) broadcast with grid (B or 1, 1, Hy, Wx, 2)
        vs = stroke[:, :-1, :].unsqueeze(2).unsqueeze(3)
        ws = stroke[:, 1:, :].unsqueeze(2).unsqueeze(3)
        distances, fraction = self.dist_line_segment(grid, vs, ws)  # (B, P-1, Hy, Wx)

        start_heights = heights[:, 0, :-1].unsqueeze(2).unsqueeze(3)
        end_heights = heights[:, 0, 1:].unsqueeze(2).unsqueeze(3)
        interpolated_heights = start_heights * (1 - fraction) + end_heights * fraction

        thickness = torch.clamp(
            interpolated_heights * self.thick_mult + self.thick_bias, min=1e-8
        )
        darkness = torch.clamp((thickness - distances) / thickness, min=0.0, max=1.0)
        darkness = (darkness + 1e-4) ** self.dark_exp
        darkness = torch.max(darkness, dim=1).values  # (B, Hy, Wx)

        if squeeze_batch:
            return darkness[0]
        return darkness

    # distance from point p to line segment v--w
    # also returns the fraction of the length of v--w that the point p is along
    def dist_line_segment(self, p, v, w):
        # p, v, w broadcast to (..., Hy, Wx, 2); last dim is xy
        d = torch.linalg.norm(v - w, dim=-1)
        dot = (p - v) * (w - v)
        dot_sum = torch.sum(dot, dim=-1) / (d ** 2 + 1e-5)
        t = torch.clamp(dot_sum, min=0, max=1).unsqueeze(-1)
        proj = v + t * (w - v)
        return torch.linalg.norm(p - proj, dim=-1), t.squeeze(-1)

    def validate(self):
        pass

def _library_canvas_dims_m(opt):
    '''(height_m, width_m) of the canvas the stroke library was painted on, or None.'''
    for h_name, w_name in [('STROKE_LIBRARY_CANVAS_HEIGHT_M', 'STROKE_LIBRARY_CANVAS_WIDTH_M'),
                           ('CANVAS_HEIGHT_M', 'CANVAS_WIDTH_M')]:
        try:
            return float(getattr(opt, h_name)), float(getattr(opt, w_name))
        except (KeyError, AttributeError, TypeError):
            continue
    return None

def get_param2img(opt, device='cuda'):
    dims_m = _library_canvas_dims_m(opt)
    param2img = StrokeParametersToImage(output_dim_meters=dims_m)
    saved_model_fn = os.path.join(opt.cache_dir, 'param2img.pt')
    if os.path.exists(saved_model_fn):
        state = torch.load(saved_model_fn)
        if 'aspect' not in state:
            # Checkpoints from before the aspect fix: use the library canvas aspect
            state['aspect'] = param2img.aspect.clone()
        param2img.load_state_dict(state)
        param2img.sync_aspect()
    else:
        print('Instatiating param2img without a saved trained model. Just using defaults.')
    param2img.eval()
    param2img.requires_grad_(False)
    param2img.to(device)
    return param2img


def special_sigmoid(x):
    return 1/(1+torch.exp(-1.*((x*2-1)+0.2) / 0.05))
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

def process_img(img):
    return np.clip(img.detach().cpu().numpy(), a_min=0, a_max=1)*255


class StrokeParametersToImage(nn.Module):
    def __init__(self, output_dim_meters=None, render_height=200):
        super(StrokeParametersToImage, self).__init__()
        self.size_y = render_height
        self.size_x = render_height

        aspect = 1.0 if output_dim_meters is None else output_dim_meters[1] / output_dim_meters[0]
        self.register_buffer('aspect', torch.tensor(float(aspect)))

        if output_dim_meters is not None:
            # input parameters are in meters. Translate these to proportions for the renderer.
            # Also account for the starting position of the stroke (Which should be 0,0)
            self.x_m = 1.0 / output_dim_meters[1]
            self.y_m = 1.0 / output_dim_meters[0]

            self.x_m = nn.Parameter(torch.tensor(self.x_m))
            self.y_m = nn.Parameter(torch.tensor(self.y_m))
        else:
            self.x_m = nn.Parameter(torch.tensor(1.0))
            self.y_m = nn.Parameter(torch.tensor(1.0))

        self.x_b = nn.Parameter(torch.tensor(0.0))
        self.y_b = nn.Parameter(torch.tensor(0.0))
        
        self.renderer = BezierRenderer(size_x=self.size_x, 
                                       size_y=self.size_y,
                                       num_pts=32)
        self.sync_aspect()

    def sync_aspect(self):
        '''Match the renderer's distance metric and default grid to self.aspect. Call after load_state_dict.'''
        aspect = float(self.aspect)
        self.renderer.set_aspect(aspect)
        self.size_x = max(1, int(round(self.size_y * aspect)))
        self.renderer.set_render_size(size_x=self.size_x, size_y=self.size_y)

    @staticmethod
    def _batch_offset(val, batch, device, dtype):
        '''Scalar or length-B tensor -> shape that broadcasts with (B, P).'''
        if not torch.is_tensor(val):
            val = torch.as_tensor(val, device=device, dtype=dtype)
        else:
            val = val.to(device=device, dtype=dtype)
        val = val.reshape(-1)
        if val.numel() == 1:
            return val.view(1, 1)
        if val.numel() != batch:
            raise ValueError(f'Expected scalar or length-{batch} offset, got {val.numel()}')
        return val.view(batch, 1)

    def get_rotated_trajectory(self, rotation, trajectory):
        '''
        Rotate trajectories around (0,0).
        args:
            rotation : scalar or (B,) radians
            trajectory : (B, n_points, 2) with y in dim 0, x in dim 1
        returns:
            (B, n_points, 2)
        '''
        x = trajectory[:, :, 1]
        y = trajectory[:, :, 0]

        if not torch.is_tensor(rotation):
            rotation = torch.as_tensor(rotation, device=trajectory.device, dtype=trajectory.dtype)
        else:
            rotation = rotation.to(device=trajectory.device, dtype=trajectory.dtype)
        rot = rotation.reshape(-1)
        if rot.numel() == 1:
            cos_r = torch.cos(rot)
            sin_r = torch.sin(rot)
            x_rot = x * cos_r + y * sin_r
            y_rot = -1 * x * sin_r + y * cos_r
        else:
            if rot.numel() != trajectory.shape[0]:
                raise ValueError(
                    f'Expected scalar or length-{trajectory.shape[0]} rotation, got {rot.numel()}'
                )
            cos_r = torch.cos(rot).view(-1, 1)
            sin_r = torch.sin(rot).view(-1, 1)
            x_rot = x * cos_r + y * sin_r
            y_rot = -1 * x * sin_r + y * cos_r

        return torch.stack([y_rot, x_rot], dim=2)

    def forward(self, parameter, x_start, y_start, theta, use_conv=False, grid=None):
        '''
        args:
            parameter : (batch, num_pts, 3)
            x_start : scalar or (batch,) start x in [0,1]
            y_start : scalar or (batch,) start y in [0,1]
            theta : scalar or (batch,) rotation in radians
            use_conv : unused (API compat); soft renderer only
            grid : optional (batch or 1, 1, Hy, Wx, 2) canvas proportions (y, x) to render at
        '''
        traj = self.trajectory_proportions(parameter, x_start, y_start, theta)
        heights = parameter[:, :, 2:3]  # (batch, num_pts, 1)

        # batch, ctrl_point, 2 -> batch, 2, ctrl_points
        traj = torch.permute(traj, (0, 2, 1))
        heights = torch.permute(heights, (0, 2, 1))

        # Run through Differentiable renderer
        x = self.renderer(traj, heights, use_conv=use_conv, grid=grid)  # (batch, Hy, Wx)

        return x

    def trajectory_proportions(self, parameter, x_start, y_start, theta):
        '''Stroke path in canvas proportions, (batch, num_pts, 2) ordered (y, x).'''
        batch = parameter.shape[0]
        traj = parameter[:, :, 0:2]  # (batch, num_pts, 2)

        # x,y to y,x
        traj = torch.flip(traj, dims=(2,))

        # To be consistent with the way BrushStroke.execute runs trajectories
        # y = -y yeah weird; then add biases (out-of-place for autograd)
        traj = torch.stack([
            -1.0 * traj[:, :, 0] + self.y_b,
            traj[:, :, 1] + self.x_b,
        ], dim=2)

        # Rotate before converting to proportions (errors when canvas isn't square)
        traj = self.get_rotated_trajectory(theta, traj)

        # traj from meters to proportion, then offset to start pose
        y_off = self._batch_offset(y_start, batch, traj.device, traj.dtype)
        x_off = self._batch_offset(x_start, batch, traj.device, traj.dtype)
        traj = torch.stack([
            traj[:, :, 0] * self.y_m + y_off,
            traj[:, :, 1] * self.x_m + x_off,
        ], dim=2)
        return traj
    
l1_loss = nn.L1Loss()
# def shift_invariant_loss(pred, real, before, fnl_weight=0.5, n=3, delta=0.02):
#     losses = None
#     for dx in torch.linspace(start=-1.0*delta, end=delta, steps=n):
#         for dy in torch.linspace(start=-1.0*delta, end=delta, steps=n):
#             x = int(dx*real.shape[2])
#             y = int(dy*real.shape[1])
#             # Translate the real stroke slightly
#             translated_pred  = affine(pred, angle=0, translate=(x, y), fill=0, scale=1.0, shear=0)

#             # L2
#             diff = (translated_pred - real)**2
#             l = diff.mean(dim=(1,2))
            
#             losses = l[None,:] if losses is None else torch.cat([losses, l[None,:]], dim=0)

#     # Only use the loss from the shift that gave the least loss value
#     loss, inds = torch.min(losses, dim=0)

#     # Extra loss for missing a stroke
#     stroke_areas = (real - before).abs() > 0.1
#     false_negative_loss = l1_loss(pred[stroke_areas], real[stroke_areas]) * fnl_weight
    
#     return loss.mean() + l1_loss(pred, real) + false_negative_loss

# Training images are resized to this height (~0.36 mm/pixel on letter paper)
TRAIN_RENDER_HEIGHT = 384
# Padding around each stroke's commanded path when cropping; also bounds how far
# off the calibration can be and still be seen by the loss.
CROP_MARGIN_M = 0.008
# Max (strokes x crop pixels) rendered together; bounds VRAM of the distance field.
CROP_MAX_PIXELS = 500_000
STROKE_AREA_WEIGHT = 0.5
VAL_PROP = 0.25

def _canvas_batch_needs_invert(sample_u8):
    '''Legacy saves used uint8(u8*255) which wraps white paper to ~mean 50-80.'''
    return float(np.asarray(sample_u8).mean()) < 128.0

def _load_stroke_library(opt, render_height):
    '''Returns (canvases_before, canvases_after) as (N, 3, render_height, W) on CPU, and the strokes.'''
    lib_dir = os.path.join(opt.cache_dir, 'stroke_library')
    before_fns = sorted(glob.glob(os.path.join(lib_dir, 'canvases_before_*.npy')))
    after_fns = sorted(glob.glob(os.path.join(lib_dir, 'canvases_after_*.npy')))
    strokes_fns = sorted(glob.glob(os.path.join(lib_dir, 'stroke_parameters*.npy')))

    def load_canvases(fns):
        out = []
        n_invert, n_direct = 0, 0
        for fn in fns:
            with gzip.GzipFile(fn, 'r') as f:
                raw = np.load(f, allow_pickle=True)
            # Per-file: rebuilds used to leave old wrapped batches next to new rgb_uint8 ones.
            invert = _canvas_batch_needs_invert(raw)
            s = raw.astype(np.float32) / 255.
            s = torch.from_numpy(s).nan_to_num().permute(0, 3, 1, 2)
            if invert:
                s = 1.0 - s
                n_invert += 1
            else:
                n_direct += 1
            w = int(round(render_height * s.shape[3] / s.shape[2]))
            out.append(Resize((render_height, w), bicubic, antialias=True)(s))
        print(f'Stroke library canvases: {n_direct} rgb_uint8, {n_invert} legacy-wrapped (1-x).')
        return torch.cat(out)

    brush_strokes = []
    for fn in strokes_fns:
        with open(fn, 'rb') as f:
            brush_strokes.extend(list(pickle.load(f)))

    return load_canvases(before_fns), load_canvases(after_fns), brush_strokes

def _stroke_tensors(brush_strokes, device):
    '''Library strokes are fixed during training, so compute their paths and poses once.'''
    with torch.no_grad():
        strokes = [bs.to(device) for bs in brush_strokes]
        paths = torch.stack([bs.get_path() for bs in strokes]).detach()
        xts = torch.cat([bs.xt.reshape(-1) for bs in strokes]).detach()
        yts = torch.cat([bs.yt.reshape(-1) for bs in strokes]).detach()
        thetas = torch.cat([bs.a.reshape(-1) for bs in strokes]).detach()
    return paths, xts, yts, thetas

def _build_crop_chunks(trans, idxs, paths, xts, yts, thetas, canvases_before, canvases_after,
                       margin_pix, max_pixels):
    '''
    Group strokes into batches of equal-size square crops around their commanded paths.
    Strokes are sorted by size so each batch is padded to a similar crop size; crops
    are shifted to stay inside the canvas.
    '''
    if len(idxs) == 0:
        return []
    device = paths.device
    H, W = canvases_before.shape[2], canvases_before.shape[3]
    with torch.no_grad():
        traj = trans.trajectory_proportions(paths[idxs], xts[idxs], yts[idxs], thetas[idxs]).cpu()
    rows, cols = traj[:, :, 0] * H, traj[:, :, 1] * W
    r_lo, r_hi = rows.min(dim=1).values - margin_pix, rows.max(dim=1).values + margin_pix
    c_lo, c_hi = cols.min(dim=1).values - margin_pix, cols.max(dim=1).values + margin_pix
    sides = torch.maximum(r_hi - r_lo, c_hi - c_lo).ceil().clamp(min=1, max=min(H, W)).long()

    groups, cur = [], []
    for k in torch.argsort(sides).tolist():
        if cur and (len(cur) + 1) * int(sides[k]) ** 2 > max_pixels:
            groups.append(cur)
            cur = []
        cur.append(k)
    groups.append(cur)

    chunks = []
    for group in groups:
        S = int(sides[group[-1]])
        ar = torch.arange(S, dtype=torch.float32)
        grids, crops_before, crops_after = [], [], []
        for k in group:
            r0 = int(round(float(r_lo[k] + r_hi[k]) / 2 - S / 2))
            c0 = int(round(float(c_lo[k] + c_hi[k]) / 2 - S / 2))
            r0, c0 = min(max(r0, 0), H - S), min(max(c0, 0), W - S)
            gy, gx = torch.meshgrid((r0 + ar) / H, (c0 + ar) / W, indexing='ij')
            grids.append(torch.stack((gy, gx), dim=-1))
            crops_before.append(canvases_before[idxs[k], :, r0:r0 + S, c0:c0 + S])
            crops_after.append(canvases_after[idxs[k], :, r0:r0 + S, c0:c0 + S])
        sel = [idxs[k] for k in group]
        chunks.append({
            'idx': sel,
            'paths': paths[sel], 'xt': xts[sel], 'yt': yts[sel], 'a': thetas[sel],
            'grid': torch.stack(grids).unsqueeze(1).to(device),  # (B, 1, S, S, 2)
            'before': torch.stack(crops_before).to(device),      # (B, 3, S, S)
            'after': torch.stack(crops_after).to(device),
        })
    return chunks

def _predict_crops(trans, chunk):
    alpha = trans(chunk['paths'], chunk['xt'], chunk['yt'], chunk['a'], grid=chunk['grid']).unsqueeze(1)
    # Library strokes are modeled as black ink over the before photo
    return chunk['before'] * (1 - alpha)

def _per_stroke_losses(pred, real, before, change_thresh=0.1):
    '''
    pred, real, before: (B, 3, S, S).
    Returns per-stroke L1 over the whole crop and over pixels the real stroke changed, each (B,).
    '''
    diff = (pred - real).abs()
    crop_l1 = diff.mean(dim=(1, 2, 3))
    changed = ((real - before).abs() > change_thresh).any(dim=1, keepdim=True).float()
    n_changed = changed.sum(dim=(1, 2, 3)) * diff.shape[1]
    stroke_l1 = (diff * changed).sum(dim=(1, 2, 3)) / n_changed.clamp(min=1)
    return crop_l1, stroke_l1

def _evaluate(trans, chunks, n):
    '''Mean (objective, crop L1, stroke-area L1) over the strokes in chunks.'''
    crop_total, stroke_total = 0.0, 0.0
    with torch.no_grad():
        for chunk in chunks:
            crop_l1, stroke_l1 = _per_stroke_losses(_predict_crops(trans, chunk), chunk['after'], chunk['before'])
            crop_total += crop_l1.sum().item()
            stroke_total += stroke_l1.sum().item()
    n = max(n, 1)
    crop, stroke = crop_total / n, stroke_total / n
    return crop + STROKE_AREA_WEIGHT * stroke, crop, stroke

def _log_stroke_images(opt, trans, chunks, idxs, canvases_after, canvases_before,
                       paths, xts, yts, thetas, name, it, n_log, h_log=128):
    '''One row per stroke: real full canvas | predicted full canvas | real crop | predicted crop.'''
    lookup = {i: (chunk, j) for chunk in chunks for j, i in enumerate(chunk['idx'])}
    w_log = int(round(h_log * canvases_after.shape[3] / canvases_after.shape[2]))
    res_full = Resize((h_log, w_log), bicubic, antialias=True)
    res_crop = Resize((h_log, h_log), bicubic, antialias=True)
    rows = []
    with torch.no_grad():
        for i in idxs[:n_log]:
            alpha = trans(paths[i:i+1], xts[i:i+1], yts[i:i+1], thetas[i:i+1]).unsqueeze(1)
            after_full = res_full(canvases_after[i:i+1].to(alpha.device))
            before_full = res_full(canvases_before[i:i+1].to(alpha.device))
            pred_full = before_full * (1 - res_full(alpha).clamp(0, 1))

            chunk, j = lookup[i]
            one = {k: chunk[k][j:j+1] for k in ('paths', 'xt', 'yt', 'a', 'grid', 'before', 'after')}
            pred_crop = _predict_crops(trans, one)
            rows.append(torch.cat([after_full, pred_full, res_crop(one['after']), res_crop(pred_crop)], dim=3))
    if not rows:
        return
    img = torch.cat(rows, dim=2)[0].permute(1, 2, 0)
    opt.writer.add_image('images_stroke_modeling_stroke/' + name, process_img(img).astype(np.uint8), it)

def train_param2stroke(opt, device='cuda', n_log=8):
    torch.random.manual_seed(0)
    np.random.seed(0)

    canvases_before, canvases_after, brush_strokes = _load_stroke_library(opt, TRAIN_RENDER_HEIGHT)
    n = len(brush_strokes)
    assert len(canvases_before) == n and len(canvases_after) == n, \
        'Stroke library has {} strokes but {}/{} before/after canvases'.format(n, len(canvases_before), len(canvases_after))
    paths, xts, yts, thetas = _stroke_tensors(brush_strokes, device)

    rand_ind = torch.randperm(n).tolist()
    n_val = int(VAL_PROP * n)
    val_idx, train_idx = rand_ind[:n_val], rand_ind[n_val:]
    print('{} training strokes. {} validation strokes'.format(len(train_idx), len(val_idx)))

    dims_m = _library_canvas_dims_m(opt)
    trans = StrokeParametersToImage(output_dim_meters=dims_m).to(device)
    print('# parameters in Param2Image model:', get_n_params(trans))

    margin_pix = CROP_MARGIN_M / dims_m[0] * TRAIN_RENDER_HEIGHT
    train_chunks = _build_crop_chunks(trans, train_idx, paths, xts, yts, thetas,
                                      canvases_before, canvases_after, margin_pix, CROP_MAX_PIXELS)
    val_chunks = _build_crop_chunks(trans, val_idx, paths, xts, yts, thetas,
                                    canvases_before, canvases_after, margin_pix, CROP_MAX_PIXELS)
    # Too few strokes for a validation split: select on training loss instead
    select_chunks, n_select = (val_chunks, len(val_idx)) if val_idx else (train_chunks, len(train_idx))

    optim = torch.optim.Adam(trans.parameters(), lr=1e-3)


    # Keep track of the best model (judged by validation error)
    best_model = copy.deepcopy(trans)
    best_val_loss = float('inf')
    best_hasnt_changed_for = 0

    # Main Training Loop: one full-batch step per iteration
    for it in tqdm(range(3000)):
        if best_hasnt_changed_for >= 400 and it > 1000:
            break # all done :)
        optim.zero_grad()

        train_loss = 0.0
        for chunk in train_chunks:
            crop_l1, stroke_l1 = _per_stroke_losses(_predict_crops(trans, chunk), chunk['after'], chunk['before'])
            loss = (crop_l1 + STROKE_AREA_WEIGHT * stroke_l1).sum() / len(train_idx)
            loss.backward()
            train_loss += loss.item()
        if not np.isfinite(train_loss):
            print('Nan')
            break
        torch.nn.utils.clip_grad_norm_(trans.parameters(), max_norm=1.0)
        optim.step()

        if it % 20 == 0:
            opt.writer.add_scalar('loss/train_loss_stroke_model', train_loss, it)
            for name, param in trans.named_parameters():
                if param.numel() == 1:
                    opt.writer.add_scalar('parameter/{}'.format(name), param.item(), it)

        # Compute validation error. Save best model
        if it % 10 == 0:
            val_loss, val_crop, val_stroke = _evaluate(trans, select_chunks, n_select)
            if val_loss < best_val_loss and it > 50:
                best_val_loss = val_loss
                best_hasnt_changed_for = 0
                best_model = copy.deepcopy(trans)
            best_hasnt_changed_for += 10
            opt.writer.add_scalar('loss/val_loss_stroke_model', val_loss, it)
            opt.writer.add_scalar('loss/val_crop_l1', val_crop, it)
            opt.writer.add_scalar('loss/val_stroke_area_loss', val_stroke, it)

        # Log images
        if it % 50 == 0:
            for log_chunks, log_idx, log_name in [(train_chunks, train_idx, 'train'), (val_chunks, val_idx, 'val')]:
                _log_stroke_images(opt, trans, log_chunks, log_idx, canvases_after, canvases_before,
                                   paths, xts, yts, thetas, log_name, it, n_log)

    print('Best stroke model loss: {:.4f}'.format(best_val_loss))
    torch.save(best_model.cpu().state_dict(), os.path.join(opt.cache_dir, 'param2img.pt'))

if __name__ == '__main__':
    from options import Options
    opt = Options()
    opt.gather_options()

    torch.manual_seed(0)
    np.random.seed(0)

    from paint_utils3 import create_tensorboard
    opt.writer = create_tensorboard()

    # Train brush strokes
    train_param2stroke(opt)