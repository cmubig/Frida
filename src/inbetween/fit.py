"""Fit a freehand drawn mark to a robot-executable FRIDA stroke.

A drawn mark is a list of canvas points (proportions, x right, y down) with
stylus pressure in 0..1. A FRIDA stroke has 4 corner points, 2 corner
roundness values and 4 pressure values, so the fit finds the stroke whose
path is closest to the drawing.

Pressure is mapped to *visible width*, not straight to the robot's z: light
stylus pressure maps to the lightest pressure that still leaves a mark, so a
light touch never silently paints nothing (the calibrated Sharpie model
paints nothing below z ~ 0.23).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from .render import Renderer, sigmoid_response
from .strokes import Stroke, logit, stroke_path


def _resample(pts: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample a polyline (M, d) to n points evenly spaced by arc length.
    Returns (n, d) points and the (M,) arc-length fractions of the inputs."""
    seg = (pts[1:, :2] - pts[:-1, :2]).norm(dim=1)
    s = torch.cat([torch.zeros(1), seg.cumsum(0)])
    total = float(s[-1])
    frac = s / total if total > 0 else torch.linspace(0, 1, len(pts))
    t = torch.linspace(0, 1, n)
    idx = torch.searchsorted(frac, t).clamp(1, len(pts) - 1)
    f0, f1 = frac[idx - 1], frac[idx]
    w = ((t - f0) / (f1 - f0).clamp_min(1e-9)).unsqueeze(1)
    return pts[idx - 1] * (1 - w) + pts[idx] * w, frac


def visible_z_min(r: Renderer) -> float:
    """Lightest pressure z that still leaves a mark under this stroke model."""
    mult = float(r.model.renderer.thick_mult)
    bias = float(r.model.renderer.thick_bias)

    def thick(z):
        zz = torch.tensor([z])
        if r.response != "model":
            zz = sigmoid_response(zz)
        return float((zz * 2 + 0.5) / 64 * mult + bias)

    lo, hi = r.settings.MIN_STROKE_Z, 0.95
    if thick(lo) > 0:
        return lo
    if thick(hi) <= 0:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if thick(mid) <= 0 else (lo, mid)
    return hi + 0.01


def pressure_to_z(p: torch.Tensor, z_lo: float, z_hi: float = 0.95) -> torch.Tensor:
    """Stylus pressure (0..1) -> robot pressure: 0 stays the lightest visible mark."""
    return z_lo + (z_hi - z_lo) * p.clamp(0, 1)


def fit_stroke(points: Sequence[Sequence[float]], pressure: Sequence[float], r: Renderer,
               iters: int = 80, color=(0.05, 0.05, 0.05)) -> Tuple[Stroke, float]:
    """Return the fitted stroke (id left empty) and the mean fit error in mm."""
    P = torch.tensor(points, dtype=torch.float32).reshape(-1, 2)  # proportions (x, y)
    pr = torch.tensor(pressure, dtype=torch.float32).reshape(-1)
    if len(pr) != len(P):
        pr = torch.full((len(P),), 0.5)
    if len(P) == 1:
        P = torch.cat([P, P + torch.tensor([[1e-3, 0.0]])])
        pr = pr.repeat(2)[:2]

    m = r.model
    x_m, y_m, x_b, y_b = float(m.x_m), float(m.y_m), float(m.x_b), float(m.y_b)
    # Pose: rotation 0, start of the stroke at the first drawn point.
    xt = float(P[0, 0]) - x_b * x_m
    yt = float(P[0, 1]) - y_b * y_m
    # Drawn points -> stroke-local meters (inverse of the renderer's transform).
    local = torch.stack([(P[:, 0] - xt) / x_m - x_b, -((P[:, 1] - yt) / y_m - y_b)], dim=1)

    target, frac = _resample(local, 64)
    length = float((target[1:] - target[:-1]).norm(dim=1).sum())
    if length < 1e-4:  # a dot: make a tiny straight mark
        target = target[0] + torch.stack([torch.linspace(0, 1e-3, 64), torch.zeros(64)], 1)

    # Initial corners at 0, 1/3, 2/3, 1 of the drawn length, then optimize.
    init = _resample(target, 4)[0]
    vertices = init.clone().requires_grad_(True)
    radii = logit(torch.tensor([0.6, 0.6])).requires_grad_(True)
    opt = torch.optim.Adam([{"params": [vertices], "lr": 1.5e-3}, {"params": [radii], "lr": 8e-2}])
    zfix = torch.full((4,), 0.5)
    for _ in range(iters):
        opt.zero_grad()
        path = stroke_path(vertices, radii, zfix, r.settings.MIN_STROKE_Z)[:, :2]
        d = torch.cdist(path, target)
        chamfer = d.min(dim=1).values.mean() + d.min(dim=0).values.mean()
        ends = (path[0] - target[0]).norm() + (path[-1] - target[-1]).norm()
        loss = chamfer + 0.5 * ends
        loss.backward()
        opt.step()
        with torch.no_grad():
            vertices.clamp_(-0.15, 0.15)
            radii.clamp_(-8, 8)

    with torch.no_grad():
        path = stroke_path(vertices, radii, zfix, r.settings.MIN_STROKE_Z)[:, :2]
        err_mm = float(torch.cdist(path, target).min(dim=1).values.mean()) * 1000

    # Pressure: 4 values sampled from the stylus pressure along the stroke.
    pres64 = _resample(torch.cat([local, pr.unsqueeze(1)], 1), 64)[0][:, 2]
    p4 = torch.stack([pres64[int(round(i * 63 / 3))] for i in range(4)])
    z = pressure_to_z(p4, visible_z_min(r))

    s = Stroke(
        id="",
        vertices=vertices.detach().tolist(),
        kappa=torch.sigmoid(radii).detach().tolist(),
        z=[round(float(v), 4) for v in z],
        xt=xt, yt=yt, a=0.0,
        color=list(color),
    )
    return s, err_mm
