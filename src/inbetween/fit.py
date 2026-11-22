"""Fit a freehand drawn mark to a robot-executable FRIDA stroke.

A drawn mark is a list of canvas points (proportions, x right, y down) with
stylus pressure in 0..1. A FRIDA stroke has 4 corner points, 2 corner
roundness values and 4 pressure values, so the fit finds the stroke whose
path is closest to the drawing. A mark too complex for one stroke (a loop, a
zigzag, a long scribble) is split into several strokes by fit_mark.

Pressure is mapped to *visible width*, not straight to the robot's z: the
lightest stylus touch maps to the lightest pressure that paints a clearly
visible line (about 0.4 mm, measured from the stroke model), so a light touch
never silently paints nothing. The calibrated Sharpie model paints nothing
below z ~ 0.23 and only a hairline just above it.
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


_LINE_Z_CACHE = {}


def painted_width_mm(r: Renderer, z: float, height: int = 1024) -> float:
    """Width of a straight stroke at pressure z, as the stroke model paints it (mm)."""
    from .strokes import Stroke
    key = (id(r), "rhi", height)
    if key not in _LINE_Z_CACHE:
        _LINE_Z_CACHE[key] = Renderer(r.cache_dir, render_height=height, response=r.response)
    rh = _LINE_Z_CACHE[key]
    s = Stroke(id="w", vertices=[[0, 0], [0.01, 0], [0.02, 0], [0.03, 0]], kappa=[0.5, 0.5],
               z=[z] * 4, xt=0.4, yt=0.5, a=0.0, order=0, color=[0, 0, 0])
    img = rh.render_frame([s]).mean(-1)
    col = img[:, int(rh.W * 0.4) + int(0.015 / rh.settings.CANVAS_WIDTH_M * rh.W)]
    return float((1 - col).sum()) * rh.settings.CANVAS_HEIGHT_M * 1000 / rh.H


def min_line_z(r: Renderer, width_mm: float = 0.4) -> float:
    """Lightest pressure that paints a clearly visible line (about width_mm wide).
    visible_z_min is where the mark *starts*, which is a line of almost no width;
    stylus pressure 0 maps here instead, so the lightest touch still shows."""
    key = (id(r), "line", width_mm)
    if key not in _LINE_Z_CACHE:
        # Painted width is linear in z above the threshold: two measurements fix the line.
        za, zb = 0.6, 0.95
        wa, wb = painted_width_mm(r, za), painted_width_mm(r, zb)
        if wb <= width_mm or wb <= wa:
            z = zb
        else:
            z = za + (width_mm - wa) * (zb - za) / (wb - wa)
        _LINE_Z_CACHE[key] = round(min(max(z, visible_z_min(r)), zb), 4)
    return _LINE_Z_CACHE[key]


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
    z = pressure_to_z(p4, min_line_z(r))

    s = Stroke(
        id="",
        vertices=vertices.detach().tolist(),
        kappa=torch.sigmoid(radii).detach().tolist(),
        z=[round(float(v), 4) for v in z],
        xt=xt, yt=yt, a=0.0,
        color=list(color),
    )
    return s, err_mm


def _split_index(P: torch.Tensor, r: Renderer, min_frac: float = 0.15) -> int:
    """Where to cut a drawn mark (M, 2 canvas proportions) in two: the sharpest
    turn away from the ends, or the middle of its length if it has no clear corner."""
    m = r.model
    xy = torch.stack([P[:, 0] / float(m.x_m), P[:, 1] / float(m.y_m)], dim=1)  # meters
    seg = (xy[1:] - xy[:-1]).norm(dim=1)
    s = torch.cat([torch.zeros(1), seg.cumsum(0)])
    total = float(s[-1])
    frac = s / total if total > 0 else torch.linspace(0, 1, len(P))
    mid = int(torch.argmin((frac - 0.5).abs()))
    lo, hi = 1, len(P) - 2
    inner = [i for i in range(lo, hi + 1) if min_frac <= float(frac[i]) <= 1 - min_frac]
    if not inner:
        return max(1, min(mid, len(P) - 2))
    # Turning angle over a window of about 3% of the length on each side.
    w = 0.03 * total
    best, best_ang = mid, 0.0
    for i in inner:
        j0 = int(torch.searchsorted(s, s[i] - w).clamp(0, i - 1))
        j1 = int(torch.searchsorted(s, s[i] + w).clamp(i + 1, len(P) - 1))
        a, b = xy[i] - xy[j0], xy[j1] - xy[i]
        na, nb = float(a.norm()), float(b.norm())
        if na < 1e-9 or nb < 1e-9:
            continue
        ang = float(torch.arccos((a @ b / (na * nb)).clamp(-1, 1)))
        if ang > best_ang:
            best, best_ang = i, ang
    return best if best_ang > 0.5 else (mid if mid in inner else inner[len(inner) // 2])  # 0.5 rad ~ 30 deg


def fit_mark(points: Sequence[Sequence[float]], pressure: Sequence[float], r: Renderer,
             max_err_mm: float = 1.0, max_pieces: int = 8, min_piece_mm: float = 4.0,
             iters: int = 80, color=(0.05, 0.05, 0.05)) -> List[Tuple[Stroke, float]]:
    """Fit a drawn mark to one robot stroke, or split it into several when one
    stroke cannot follow it within max_err_mm (a loop, a zigzag, a long scribble).

    Splits happen at the sharpest turn (else mid-length), recursively, until every
    piece fits, a piece would be shorter than min_piece_mm, or max_pieces is reached.
    Neighboring pieces share the cut point, so the painted mark stays connected.
    Returns [(stroke, err_mm), ...] in drawing order."""
    P = [list(p)[:2] for p in points]
    pr = list(pressure) if len(pressure) == len(P) else [0.5] * len(P)
    m = r.model

    def length_mm(pts):
        t = torch.tensor(pts, dtype=torch.float32)
        if len(t) < 2:
            return 0.0
        d = torch.stack([(t[1:, 0] - t[:-1, 0]) / float(m.x_m), (t[1:, 1] - t[:-1, 1]) / float(m.y_m)], 1)
        return float(d.norm(dim=1).sum()) * 1000

    out: List[Tuple[Stroke, float]] = []

    def rec(pts, prs, budget):
        s, err = fit_stroke(pts, prs, r, iters=iters, color=color)
        if err <= max_err_mm or budget <= 1 or len(pts) < 5 or length_mm(pts) < 2 * min_piece_mm:
            out.append((s, err))
            return 1
        k = _split_index(torch.tensor(pts, dtype=torch.float32), r)
        left, right = pts[: k + 1], pts[k:]
        if min(length_mm(left), length_mm(right)) < min_piece_mm:
            out.append((s, err))
            return 1
        used = rec(left, prs[: k + 1], budget - 1)
        return used + rec(right, prs[k:], budget - used)

    rec(P, pr, max_pieces)
    return out
