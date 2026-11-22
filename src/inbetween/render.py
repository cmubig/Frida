"""One rendering interface for the whole pipeline.

``Renderer`` wraps FRIDA's calibrated stroke model (``param2img.pt`` in a
cache dir) and adds an optional *pressure response* applied before rendering:

* ``"model"``: FRIDA's fitted model as-is. Note its width is linear in
  pressure, so geometric interpolation already changes width evenly.
* ``"placeholder_sigmoid"``: an assumed S-shaped pressure -> effective
  pressure curve that mimics a real brush (little change at light pressure,
  a jump in the middle, saturation at the top). PLACEHOLDER ONLY: replace it
  with a response fitted to real calibration strokes before reporting results
  as real-brush behavior.

Swapping in a new learned stroke model later should only touch this file.
"""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace
from typing import Iterable, List, Optional

import torch

from param2stroke import StrokeParametersToImage
from .strokes import Stroke, stroke_path, stroke_tensors

RESPONSES = ("model", "placeholder_sigmoid")


def load_settings(cache_dir: str) -> SimpleNamespace:
    """Settings recorded when the stroke library was painted (canvas size etc.)."""
    fn = os.path.join(cache_dir, "stroke_library", "stroke_settings_during_library.json")
    defaults = dict(CANVAS_WIDTH_M=0.254, CANVAS_HEIGHT_M=0.2032, MIN_STROKE_Z=0.1,
                    MAX_STROKE_LENGTH=0.05, MIN_STROKE_LENGTH=0.0025)
    if os.path.exists(fn):
        with open(fn) as f:
            defaults.update(json.load(f))
    return SimpleNamespace(**defaults)


def sigmoid_response(z: torch.Tensor, steepness: float = 12.0, center: float = 0.5) -> torch.Tensor:
    """Monotone S-curve mapping [0,1] -> [0,1] (placeholder brush response)."""
    s = lambda v: torch.sigmoid(steepness * (v - center))
    lo, hi = s(torch.tensor(0.0)), s(torch.tensor(1.0))
    return (s(z) - lo) / (hi - lo)


class Renderer:
    def __init__(self, cache_dir: str, render_height: int = 256, response: str = "model",
                 device: str = "cpu"):
        if response not in RESPONSES:
            raise ValueError(f"response must be one of {RESPONSES}")
        self.settings = load_settings(cache_dir)
        self.cache_dir = cache_dir
        self.response = response
        self.device = device
        dims = (self.settings.CANVAS_HEIGHT_M, self.settings.CANVAS_WIDTH_M)
        model = StrokeParametersToImage(output_dim_meters=dims, render_height=render_height)
        fn = os.path.join(cache_dir, "param2img.pt")
        if os.path.exists(fn):
            state = torch.load(fn, map_location="cpu")
            if "aspect" not in state:
                state["aspect"] = model.aspect.clone()
            model.load_state_dict(state)
            model.sync_aspect()
            self.calibrated = True
        else:
            print(f"[inbetween] no param2img.pt in {cache_dir}; using uncalibrated defaults")
            self.calibrated = False
        model.eval().requires_grad_(False).to(device)
        self.model = model
        self.H = model.size_y
        self.W = model.size_x

    # -- core ---------------------------------------------------------------
    def apply_response(self, path: torch.Tensor) -> torch.Tensor:
        if self.response == "model":
            return path
        z = sigmoid_response(path[..., 2])
        return torch.cat([path[..., :2], z.unsqueeze(-1)], dim=-1)

    def path(self, p: dict) -> torch.Tensor:
        return stroke_path(p["vertices"], p["radii"], p["z"], self.settings.MIN_STROKE_Z)

    def render_params(self, plist: List[dict], grid: Optional[torch.Tensor] = None,
                      xt=None, yt=None, a=None) -> torch.Tensor:
        """Render a batch of strokes given as tensor dicts. Returns (B, H, W) darkness.

        xt/yt/a override every stroke's pose (used for canonical renders)."""
        paths = torch.stack([self.apply_response(self.path(p)) for p in plist])
        xts = torch.cat([p["xt"].reshape(1) for p in plist]) if xt is None else xt
        yts = torch.cat([p["yt"].reshape(1) for p in plist]) if yt is None else yt
        th = torch.cat([p["a"].reshape(1) for p in plist]) if a is None else a
        return self.model(paths, xts, yts, th, grid=grid)

    # -- convenience ----------------------------------------------------------
    def render_frame(self, strokes: Iterable[Stroke], chunk: int = 8) -> torch.Tensor:
        """Composite strokes over white paper in painting order -> (H, W, 3) in 0..1.
        Strokes are rendered `chunk` at a time, so memory stays flat for frames
        with hundreds of strokes."""
        strokes = sorted(strokes, key=lambda s: (s.order, s.id))
        canvas = torch.ones(3, self.H, self.W, device=self.device)
        if not strokes:
            return canvas.permute(1, 2, 0)
        with torch.no_grad():
            for i in range(0, len(strokes), chunk):
                part = strokes[i:i + chunk]
                maps = self.render_params([stroke_tensors(s, self.device) for s in part])
                for s, m in zip(part, maps):
                    c = torch.tensor(s.color, device=self.device)[:, None, None]
                    canvas = canvas * (1 - m) + m * c
        return canvas.permute(1, 2, 0).clamp(0, 1)

    def to_metric(self, prop_yx: torch.Tensor) -> torch.Tensor:
        """Canvas proportions (y, x) -> isotropic canvas meters (y, x)."""
        return torch.stack([prop_yx[..., 0] / self.model.y_m, prop_yx[..., 1] / self.model.x_m], -1)

    def from_metric(self, m_yx: torch.Tensor) -> torch.Tensor:
        return torch.stack([m_yx[..., 0] * self.model.y_m, m_yx[..., 1] * self.model.x_m], -1)

    def path_canvas(self, p: dict, xt=None, yt=None, a=None) -> torch.Tensor:
        """Stroke path in canvas proportions (32, 2) ordered (y, x)."""
        path = self.path(p)[None]
        return self.model.trajectory_proportions(
            path,
            p["xt"] if xt is None else xt,
            p["yt"] if yt is None else yt,
            p["a"] if a is None else a,
        )[0]

    def pixel_area_m2(self, grid_h_prop: float, grid_w_prop: float, ny: int, nx: int) -> float:
        """Area of one pixel (square meters) for a grid spanning the given proportions."""
        hm = grid_h_prop * self.settings.CANVAS_HEIGHT_M / ny
        wm = grid_w_prop * self.settings.CANVAS_WIDTH_M / nx
        return hm * wm


def save_gif(frames: List[torch.Tensor], path: str, fps: int = 8) -> None:
    import imageio.v2 as imageio
    import numpy as np
    imgs = [(f.cpu().numpy() * 255).astype("uint8") for f in frames]
    imageio.mimsave(path, imgs, duration=1.0 / fps, loop=0)


def save_png(img: torch.Tensor, path: str) -> None:
    import imageio.v2 as imageio
    imageio.imwrite(path, (img.cpu().numpy() * 255).astype("uint8"))
