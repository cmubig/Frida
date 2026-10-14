import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode 
bicubic = InterpolationMode.BICUBIC
import numpy as np
from brush_stroke import BrushStroke
from param2stroke import get_param2img

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Soft-rasterize this many strokes per param2img call.
# Grad chunks must stay small: autograd keeps (B, P-1, H, W) intermediates.
# No-grad chunks can be larger (no saved tensors).
_DEFAULT_GRAD_RENDER_CHUNK = 4
_DEFAULT_NOGRAD_RENDER_CHUNK = 32


def _opt_get(opt, name, default):
    # Options.__getattr__ raises KeyError (not AttributeError), so getattr defaults fail.
    if hasattr(opt, 'opt') and isinstance(opt.opt, dict):
        return opt.opt.get(name, default)
    return getattr(opt, name, default)


class Painting(nn.Module):
    def __init__(self, opt, n_strokes=None, background_img=None, brush_strokes=None):
        # h, w are canvas height and width in pixels
        super(Painting, self).__init__()
        self.background_img = background_img
        self.strokes_per_batch = opt.strokes_per_batch
        self.grad_render_chunk = int(_opt_get(opt, 'grad_render_chunk', _DEFAULT_GRAD_RENDER_CHUNK))
        self.nograd_render_chunk = int(_opt_get(opt, 'nograd_render_chunk', _DEFAULT_NOGRAD_RENDER_CHUNK))

        if self.background_img is not None and self.background_img.shape[1] == 3: # add alpha channel
            t =  torch.zeros((1,1,self.background_img.shape[2],self.background_img.shape[3])).to(device)
            # t[:,:3] = self.background_img
            self.background_img = torch.cat((self.background_img, t), dim=1)

        if brush_strokes is None:
            self.brush_strokes = nn.ModuleList([BrushStroke(opt) for _ in range(n_strokes)])
        else:
            self.brush_strokes = nn.ModuleList(brush_strokes)
        
        self.param2img = get_param2img(opt)

    def get_optimizers(self, multiplier=1.0, ink=False):
        xt = []
        yt = []
        a = []
        z = []
        color = []
        path_params = []
        
        for n, p in self.named_parameters():
            leaf = n.split('.')[-1]
            if leaf in ("vertices", "radii"): path_params.append(p)
            if leaf == "xt": xt.append(p)
            if leaf == "yt": yt.append(p)
            if leaf == "a": a.append(p)
            if leaf == "z": z.append(p)
            if leaf == "color_transform": color.append(p)

        path_opt = torch.optim.RMSprop(path_params, lr=1e-3)
        position_opt = torch.optim.RMSprop(xt + yt, lr=5e-3*multiplier)
        height_opt = torch.optim.RMSprop(z, lr=5e-3*multiplier)
        rotation_opt = torch.optim.RMSprop(a, lr=1e-2*multiplier)
        color_opt = None if ink else torch.optim.RMSprop(color, lr=5e-3*multiplier)

        return position_opt, height_opt, rotation_opt, color_opt, path_opt 

    def _paths_and_poses(self, indices):
        strokes = [self.brush_strokes[i] for i in indices]
        paths = torch.stack([bs.get_path() for bs in strokes], dim=0)
        xts = torch.cat([bs.xt.reshape(-1) for bs in strokes], dim=0)
        yts = torch.cat([bs.yt.reshape(-1) for bs in strokes], dim=0)
        thetas = torch.cat([bs.a.reshape(-1) for bs in strokes], dim=0)
        return paths, xts, yts, thetas

    def _param2img_chunk(self, paths, xts, yts, thetas):
        return self.param2img(paths, xts, yts, thetas, use_conv=False)

    def _render_stroke_maps(self, indices, h, w, with_grad, chunk_size):
        '''Render stroke alpha maps in chunks. No-grad uses larger chunks safely.'''
        if len(indices) == 0:
            return {}

        rendered = {}
        chunk = max(1, int(chunk_size))
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            for start in range(0, len(indices), chunk):
                chunk_idx = indices[start:start + chunk]
                paths, xts, yts, thetas = self._paths_and_poses(chunk_idx)
                if with_grad:
                    # Recompute distance-field in backward instead of saving (B,P-1,H,W).
                    maps = checkpoint(
                        self._param2img_chunk, paths, xts, yts, thetas,
                        use_reentrant=False,
                    )
                else:
                    maps = self._param2img_chunk(paths, xts, yts, thetas)
                maps = maps.unsqueeze(1)  # (B, 1, Hy, Wx)
                if maps.shape[2] != h or maps.shape[3] != w:
                    maps = T.Resize((h, w), bicubic, antialias=True)(maps)
                for j, i in enumerate(chunk_idx):
                    # Materialize per-stroke so the chunk activation can be freed.
                    rendered[i] = maps[j:j + 1].contiguous()
        return rendered

    def forward(self, h, w, use_alpha=True, return_alphas=False, opacity_factor=1.0):
        if self.background_img is None:
            canvas = torch.ones((1,4,h,w)).to(device)
        else:
            canvas = T.Resize((h,w), bicubic, antialias=True)(self.background_img).detach()
        canvas[:,3] = 1 # alpha channel

        alpha_sum = None
        n_strokes = len(self.brush_strokes)
        if n_strokes == 0:
            if return_alphas:
                return canvas, torch.zeros((1, h, w), device=canvas.device)
            return canvas

        # Pick strokes_per_batch random indices to compute gradients on
        num_with_grad = min(self.strokes_per_batch, n_strokes)
        grad_idx = torch.randperm(n_strokes)[:num_with_grad].tolist()
        grad_set = set(grad_idx)
        nograd_idx = [i for i in range(n_strokes) if i not in grad_set]

        # Grad: small chunks (VRAM). No-grad: larger chunks (throughput).
        rendered = {}
        rendered.update(self._render_stroke_maps(
            grad_idx, h, w, with_grad=True, chunk_size=self.grad_render_chunk))
        rendered.update(self._render_stroke_maps(
            nograd_idx, h, w, with_grad=False, chunk_size=self.nograd_render_chunk))

        for i, brush_stroke in enumerate(self.brush_strokes):
            stroke_map = rendered[i]  # (1, 1, h, w)
            # Match BrushStroke.forward: replicate alpha into 4 channels, then paint RGB
            x = torch.cat([stroke_map, stroke_map, stroke_map, stroke_map], dim=1)
            single_stroke = torch.cat(
                (x[:, :3] * 0 + brush_stroke.color_transform[None, :, None, None], x[:, 3:]),
                dim=1,
            )

            if return_alphas:
                a = single_stroke[:, 3:]
                alpha_sum = a if alpha_sum is None else alpha_sum + a

            if use_alpha:
                canvas = canvas * (1 - single_stroke[:, 3:] * opacity_factor) + single_stroke[:, 3:] * opacity_factor * single_stroke
            else:
                canvas = canvas[:, :3] * (1 - single_stroke[:, 3:] * opacity_factor) + single_stroke[:, 3:] * opacity_factor * single_stroke[:, :3]

        if return_alphas:
            # Keep (1, H, W) instead of stacking all stroke alphas (saves a lot of VRAM).
            return canvas, alpha_sum[:, 0]

        return canvas

    def get_alpha(self, h, w):
        # return the alpha values of the strokes of the painting

        stroke_alphas = []
        for brush_stroke in self.brush_strokes:
            single_stroke = brush_stroke(h,w)

            stroke_alphas.append(single_stroke[:,3:])

        alphas = torch.cat(stroke_alphas, dim=1)
        alphas, _ = torch.max(alphas, dim=1)#alphas.max(dim=1)
        return alphas

    def validate(self):
        ''' Make sure all brush strokes have valid parameters '''
        for s in self.brush_strokes:
            BrushStroke.make_valid(s)


    def cluster_colors(self, n_colors):
        colors = [b.color_transform[:3].detach().cpu().numpy() for b in self.brush_strokes]
        colors = np.stack(colors)[None,:,:]

        from sklearn.cluster import KMeans
        from paint_utils3 import rgb2lab, lab2rgb
        # Cluster in LAB space
        colors = rgb2lab(colors)
        kmeans = KMeans(n_clusters=n_colors, random_state=0)
        kmeans.fit(colors.reshape((colors.shape[0]*colors.shape[1],3)))
        colors = [kmeans.cluster_centers_[i] for i in range(len(kmeans.cluster_centers_))]

        colors = np.array(colors)

        # Back to rgb
        colors = lab2rgb(colors[None,:,:])[0]
        return torch.from_numpy(colors).to(device)# *255., labels
    
    def pop(self):
        ''' Remove and return first stroke in the plan '''
        bs = self.brush_strokes[0]
        # Remove the stroke
        self.brush_strokes = nn.ModuleList([self.brush_strokes[i] for i in range(1,len(self.brush_strokes),1)])
        return bs
    
    def __len__(self):
        return len(self.brush_strokes)
