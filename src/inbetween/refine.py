"""Appearance-space refinement of in-between frames.

Geometric in-betweening changes pressure evenly, but a real brush (or marker)
does not turn even pressure changes into even width changes: there is a dead
zone where nothing is painted, then width grows, then it saturates. So the
painted width "pops". Refinement fixes this through the stroke model:

1. Measure each stroke's *painted width profile* (width at K places along the
   stroke) on every frame, from the rendered image of that stroke in a
   canonical window (its position and rotation removed). Because it is read
   off the rendered image, this works for any stroke model.
2. For every free in-between, the target profile changes evenly, in the
   animator's eased time, between the surrounding fixed frames.
3. Solve for the stroke's pressure values so the rendered profile matches the
   target: a coarse search over a pressure offset (needed because inside the
   dead zone the renderer has no gradient), then gradient descent through the
   differentiable renderer.

Only pressure changes: motion and shape stay exactly as the geometric
in-between made them.

Fixed, never changed: keyframes and strokes the artist edited on an
in-between frame (locked). They are the anchors the targets interpolate
between, so the rest re-flows around artist edits.

Executability: pressure is clamped to the robot's valid range (as in
BrushStroke.make_valid) at every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .interpolate import eased_time
from .render import Renderer
from .strokes import Frame, Sequence, stroke_tensors


@dataclass
class RefineConfig:
    window_px: int = 72  # canonical render resolution (pixels per side)
    margin_m: float = 0.007  # padding around each stroke's canonical window
    profile_points: int = 8  # K places along the stroke where width is measured
    profile_samples: int = 25  # samples across the stroke at each place
    profile_half_m: float = 0.006  # measure +-6 mm either side of the path
    search_steps: int = 41  # coarse search over a pressure offset in [-0.6, 0.6]
    iters: int = 40  # gradient steps per stroke per frame
    lr_z: float = 2e-2
    w_z_prior: float = 1e-3  # small pull toward the geometric in-between's pressure
    verbose: bool = True


# --------------------------------------------------------------------------
# Measuring painted width
# --------------------------------------------------------------------------

def _canonical_setup(r: Renderer, chain: List[dict], cfg: RefineConfig):
    """Fixed canonical pose + square window for one stroke across its frames."""
    with torch.no_grad():
        center, zero = torch.tensor([0.5]), torch.zeros(1)
        offsets, pts = [], []
        for p in chain:
            c = r.path_canvas(p, xt=center, yt=center, a=zero)  # (32, 2) (y, x)
            offsets.append(c.mean(0) - torch.tensor([0.5, 0.5]))  # centroid -> center
            pts.append(c - offsets[-1])
        m = r.to_metric(torch.cat(pts))
        lo, hi = m.min(0).values - cfg.margin_m, m.max(0).values + cfg.margin_m
        half = float((hi - lo).max()) / 2
        mid = (lo + hi) / 2
        lo_p, hi_p = r.from_metric(mid - half), r.from_metric(mid + half)
        n = cfg.window_px
        gy, gx = torch.meshgrid(torch.linspace(float(lo_p[0]), float(hi_p[0]), n),
                                torch.linspace(float(lo_p[1]), float(hi_p[1]), n), indexing="ij")
        grid = torch.stack([gy, gx], -1).view(1, 1, n, n, 2)
    poses = [(torch.tensor([0.5]) - o[1:2], torch.tensor([0.5]) - o[0:1]) for o in offsets]  # (xt, yt)
    return grid, poses


def _render_canonical(r: Renderer, p: dict, grid, pose) -> torch.Tensor:
    xt, yt = pose
    return r.render_params([p], grid=grid, xt=xt, yt=yt, a=torch.zeros(1))[0]


def width_profile(r: Renderer, p: dict, grid, pose, cfg: RefineConfig) -> torch.Tensor:
    """Painted width (meters) at K places along the stroke, read off the render:
    darkness integrated along a line perpendicular to the path."""
    img = _render_canonical(r, p, grid, pose)  # (n, n)
    xt, yt = pose
    m = r.to_metric(r.path_canvas(p, xt=xt, yt=yt, a=torch.zeros(1)))  # (32, 2) metric (y, x)
    idx = torch.linspace(2, m.shape[0] - 3, cfg.profile_points).round().long()
    tang = m[idx + 1] - m[idx - 1]
    tang = tang / tang.norm(dim=1, keepdim=True).clamp_min(1e-9)
    normal = torch.stack([-tang[:, 1], tang[:, 0]], -1)
    off = torch.linspace(-cfg.profile_half_m, cfg.profile_half_m, cfg.profile_samples)
    pts = r.from_metric(m[idx][:, None, :] + off[None, :, None] * normal[:, None, :])  # (K, M, 2)
    lo_y, lo_x = grid[0, 0, 0, 0, 0], grid[0, 0, 0, 0, 1]
    hi_y, hi_x = grid[0, 0, -1, 0, 0], grid[0, 0, 0, -1, 1]
    u = 2 * (pts[..., 1] - lo_x) / (hi_x - lo_x) - 1
    v = 2 * (pts[..., 0] - lo_y) / (hi_y - lo_y) - 1
    samples = torch.nn.functional.grid_sample(img[None, None], torch.stack([u, v], -1)[None],
                                              mode="bilinear", padding_mode="zeros",
                                              align_corners=True)[0, 0]  # (K, M)
    return samples.sum(dim=1) * (2 * cfg.profile_half_m / (cfg.profile_samples - 1))


def _clamp_z(z: torch.Tensor, min_z: float, ink: bool) -> torch.Tensor:
    z = z.clamp(max(0.02, min_z), 0.98)
    if not ink:
        z = torch.cat([z.new_full((1,), 0.1), z[1:]])  # soft brushes: light touchdown (make_valid)
    return z


# --------------------------------------------------------------------------
# Refinement
# --------------------------------------------------------------------------

def _chains(frames: List[Frame]):
    """(stroke id, [frame indices]) for every contiguous run where the stroke exists."""
    ids = sorted({sid for fr in frames for sid in fr.strokes})
    out = []
    for sid in ids:
        run = []
        for fi, fr in enumerate(frames):
            if sid in fr.strokes:
                run.append(fi)
            elif run:
                out.append((sid, run)); run = []
        if run:
            out.append((sid, run))
    return out


def _fit_pressure(r, p, target_mm, grid, pose, cfg, min_z, ink):
    """Adjust p['z'] so the rendered width profile matches target_mm (K,)."""
    z0 = p["z"].detach().clone()
    loss_of = lambda z: (((width_profile(r, {**p, "z": z}, grid, pose, cfg) / 1e-3) - target_mm) ** 2).mean()
    with torch.no_grad():  # coarse: shift all pressure values together
        best = (float("inf"), z0)
        for d in torch.linspace(-0.6, 0.6, cfg.search_steps):
            z = _clamp_z(z0 + d, min_z, ink)
            l = float(loss_of(z) + cfg.w_z_prior * ((z - z0) ** 2).mean())
            if l < best[0]:
                best = (l, z)
    z = best[1].clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=cfg.lr_z)
    for _ in range(cfg.iters):  # fine: per control point, through the renderer
        opt.zero_grad()
        loss = loss_of(z) + cfg.w_z_prior * ((z - z0) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(_clamp_z(z, min_z, ink))
    return z.detach(), float(loss_of(z.detach()))


def refine(seq: Sequence, frames: List[Frame], r: Renderer, cfg: RefineConfig = None,
           ink: bool = True) -> Tuple[List[Frame], Dict[str, float]]:
    """Return refined copies of ``frames`` and per-stroke final fit errors (mm^2)."""
    cfg = cfg or RefineConfig()
    te = eased_time(seq)
    min_z = r.settings.MIN_STROKE_Z
    out = [Frame(frame=fr.frame, kind=fr.kind, strokes={k: s.copy() for k, s in fr.strokes.items()},
                 locked=list(fr.locked)) for fr in frames]
    fixed = lambda fi, sid: frames[fi].kind == "key" or sid in frames[fi].locked
    report = {}

    for sid, run in _chains(frames):
        ps = [stroke_tensors(frames[fi].strokes[sid]) for fi in run]
        grid, poses = _canonical_setup(r, ps, cfg)
        with torch.no_grad():
            W = [width_profile(r, p, grid, pose, cfg) / 1e-3 for p, pose in zip(ps, poses)]  # mm
        anchors = [j for j, fi in enumerate(run) if fixed(fi, sid)]
        errs = []
        for a, b in zip(anchors[:-1], anchors[1:]):
            ta, tb = te[frames[run[a]].frame], te[frames[run[b]].frame]
            for j in range(a + 1, b):
                t = (te[frames[run[j]].frame] - ta) / max(tb - ta, 1e-9)
                target = W[a] + (W[b] - W[a]) * t
                z, err = _fit_pressure(r, ps[j], target, grid, poses[j], cfg, min_z, ink)
                ps[j]["z"] = z
                new = frames[run[j]].strokes[sid].copy()
                new.z = z.tolist()  # only pressure changes; geometry is copied as-is
                out[run[j]].strokes[sid] = new
                errs.append(err)
        # Frames before the first / after the last anchor have no target and keep
        # their geometric in-between (e.g. a stroke that pops in mid-segment).
        if errs:
            report[sid] = sum(errs) / len(errs)
            if cfg.verbose:
                print(f"  refined {sid:>10}: {len(errs)} frames, mean fit error {report[sid]:.4f} mm^2")
    return out, report
