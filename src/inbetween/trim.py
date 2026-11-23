"""Trim paths: a stroke draws on (or off) over a key span instead of popping in (or out)
at its middle, as After Effects' Trim Paths does.

Stored in the keyframe file (version 3) next to "paths", never painted itself:

    "trims": [{"span": "0-10", "ids": ["s7"], "mode": "draw_on"}]

draw_on: the stroke is in keyframe span[1] but not span[0]; on each in-between it is
the first fraction u (by length, at the span's eased time) of its complete shape, u
growing to 1. draw_off: in span[0] but not span[1], u shrinking from 1 to 0 (it retracts
toward where it started, the reverse of drawing it).

Each partial stroke is a new, complete FRIDA stroke, not a mask or a rendering trick:
the complete stroke's painted path (its 32-point polyline) is cut at length fraction u,
and 4 corners + 2 roundness values are fitted to the piece (as fit.py fits a drawn mark),
warm-started from the next longer piece so the frames don't flicker. Its 4 pressure values
are read off the complete stroke's pressure along the path: the first is the pressure at
the start, the last the pressure *at the cut* (so a tapered end doesn't appear on every
frame), the two middle ones a least-squares fit in between. Pose (xt, yt, a) starts as the
complete stroke's. Clamped like BrushStroke.make_valid; a piece shorter than FRIDA's
MIN_STROKE_LENGTH is left out of that frame, so it is never painted.

The complete state is the keyframe where the stroke is complete (span[1] for draw_on,
span[0] for draw_off), or a pin on an in-between: a pin acts as an extra key, so the
stroke draws on toward the pinned state and is interpolated as usual after it.

A trimmed stroke moves with the object it belongs to:
- in a motion-path group for that span (its id is in the path's ids): it shifts with the
  group's anchor, relative to the frame where it is complete;
- otherwise it follows the nearest stroke present in both keys (nearest on the complete
  frame): that stroke's change in position and rotation since the complete frame is
  applied to it (rotation about that stroke's centre).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .render import Renderer
from .strokes import Frame, Sequence, Stroke, Trim, logit, stroke_path, stroke_tensors, wrap_angle

FIT_ITERS = 120       # from scratch
WARM_ITERS = 50       # warm-started from the next longer piece
FIT_POINTS = 64


# --------------------------------------------------------------------------
# Cutting a stroke
# --------------------------------------------------------------------------

def _arc(xy: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros(1), (xy[1:] - xy[:-1]).norm(dim=1).cumsum(0)])


def _interp(s: torch.Tensor, s_knots: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear values (K, d) at knots s_knots (K,), evaluated at s (M,)."""
    k = s_knots.detach()
    idx = torch.searchsorted(k, s.detach().clamp(float(k[0]), float(k[-1]))).clamp(1, len(k) - 1)
    s0, s1 = s_knots[idx - 1], s_knots[idx]
    w = ((s - s0) / (s1 - s0).clamp_min(1e-12)).unsqueeze(-1)
    v = values if values.dim() == 2 else values.unsqueeze(-1)
    out = v[idx - 1] * (1 - w) + v[idx] * w
    return out if values.dim() == 2 else out.squeeze(-1)


def complete_profile(s: Stroke, r: Renderer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The painted path of a stroke in stroke-local meters: (32, 2) xy, (32,) z, (32,) arc length."""
    with torch.no_grad():
        path = r.path(stroke_tensors(s))
    return path[:, :2], path[:, 2], _arc(path[:, :2])


def pressure_at(s: Stroke, r: Renderer, frac: float) -> float:
    """The stroke's pressure at length fraction `frac` along its painted path."""
    _, z, L = complete_profile(s, r)
    return float(_interp(torch.tensor([frac * float(L[-1])]), L, z)[0])


# z of the 32 path points from the 4 pressure values, exactly as stroke_path does it.
_Z_MATRIX = torch.nn.functional.interpolate(torch.eye(4)[:, None, :], size=32, mode="linear")[:, 0, :].T  # (32, 4)


def trim_piece(s: Stroke, u: float, r: Renderer, init: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[Optional[Stroke], float, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """First fraction u of stroke s as a new FRIDA stroke (same id, order, color and pose).

    Returns (stroke or None if shorter than MIN_STROKE_LENGTH, mean fit error in mm,
    fitted (vertices, radii logits) for warm-starting the next piece)."""
    xy, z, L = complete_profile(s, r)
    cut = float(L[-1]) * min(max(u, 0.0), 1.0)
    if cut < r.settings.MIN_STROKE_LENGTH:
        return None, 0.0, init
    if u >= 1.0:
        return s.copy(), 0.0, (torch.tensor(s.vertices), logit(torch.tensor(s.kappa)), float(L[-1]))
    keep = L < cut
    pts = torch.cat([xy[keep], _interp(torch.tensor([cut]), L, xy)])
    target = _interp(torch.linspace(0, cut, FIT_POINTS), _arc(pts), pts)

    if init is None:
        v0 = _interp(torch.linspace(0, cut, 4), _arc(target), target)
        r0 = logit(torch.tensor([0.6, 0.6]))
        iters = FIT_ITERS
    else:  # the next longer piece, shrunk toward the start to this length
        pv, r0, plen = init
        v0 = pv[0] + (pv - pv[0]) * (cut / plen)
        iters = WARM_ITERS
    vertices = v0.clone().requires_grad_(True)
    radii = r0.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [vertices], "lr": 4e-4}, {"params": [radii], "lr": 5e-2}])
    zfix = torch.full((4,), 0.5)
    min_z = r.settings.MIN_STROKE_Z
    s_fit = torch.linspace(0, 1, FIT_POINTS)
    for _ in range(iters):
        opt.zero_grad()
        path = stroke_path(vertices, radii, zfix, min_z)[:, :2]
        # Compare points in order along the stroke (by length): this forbids a piece that
        # folds back on itself, which a nearest-point (chamfer) loss would accept.
        La = _arc(path)
        loss = (_interp(s_fit * La[-1], La, path) - target).norm(dim=1).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            vertices.clamp_(-0.15, 0.15)
            radii.clamp_(-8, 8)

    with torch.no_grad():
        path = stroke_path(vertices, radii, zfix, min_z)[:, :2]
        La = _arc(path)
        err_mm = float((_interp(s_fit * La[-1], La, path) - target).norm(dim=1).mean()) * 1000
        # Pressure: the complete stroke's pressure at the same length along the path.
        want = _interp(_arc(path) / float(_arc(path)[-1].clamp_min(1e-12)) * cut, L, z)  # (32,)
        z0, z3 = float(_interp(torch.tensor([0.0]), L, z)[0]), float(_interp(torch.tensor([cut]), L, z)[0])
        rest = want - _Z_MATRIX[:, 0] * z0 - _Z_MATRIX[:, 3] * z3
        mid = torch.linalg.lstsq(_Z_MATRIX[:, 1:3], rest.unsqueeze(1)).solution[:, 0]
        zz = [z0, float(mid[0]), float(mid[1]), z3]

    piece = s.copy()
    piece.vertices = vertices.detach().tolist()
    piece.kappa = torch.sigmoid(radii).detach().tolist()
    piece.z = [min(max(v, 0.02), 0.98) for v in zz]  # as BrushStroke.make_valid
    return piece, err_mm, (vertices.detach(), radii.detach(), float(La[-1]))


# --------------------------------------------------------------------------
# Moving with the object it belongs to
# --------------------------------------------------------------------------

def _metric_xy(r: Renderer, prop_yx: torch.Tensor) -> torch.Tensor:
    m = r.to_metric(prop_yx)
    return torch.stack([m[..., 1], m[..., 0]], -1)


def _canvas_path_xy(s: Stroke, r: Renderer, **pose) -> torch.Tensor:
    with torch.no_grad():
        p = stroke_tensors(s)
        pose = {k: torch.tensor([float(v)]) for k, v in pose.items()}
        return _metric_xy(r, r.path_canvas(p, **pose))


def _distance(a: Stroke, b: Stroke, r: Renderer) -> float:
    """Mean distance from a's path points to b's path, metres."""
    return float(torch.cdist(_canvas_path_xy(a, r), _canvas_path_xy(b, r)).min(dim=1).values.mean())


def _place(piece: Stroke, start_xy: torch.Tensor, r: Renderer) -> None:
    """Set piece.xt, yt so its path starts at start_xy (metric x, y)."""
    c0 = _canvas_path_xy(piece, r, xt=0.0, yt=0.0)[0]
    target = r.from_metric(torch.stack([start_xy[1], start_xy[0]]))  # (y, x) proportions
    c0p = r.from_metric(torch.stack([c0[1], c0[0]]))
    piece.yt = float(target[0] - c0p[0])
    piece.xt = float(target[1] - c0p[1])


def follow(piece: Stroke, complete: Stroke, n_ref: Stroke, n_now: Stroke, r: Renderer) -> None:
    """Apply stroke N's change in position and rotation (n_ref -> n_now) to the piece, which
    has the complete stroke's pose: rotate about N's centre and move with it."""
    da = wrap_angle(n_now.a - n_ref.a)
    cr, cn = _canvas_path_xy(n_ref, r).mean(0), _canvas_path_xy(n_now, r).mean(0)
    # The canvas rotation that a change da of the pose angle produces (measured on N).
    p0 = _canvas_path_xy(n_ref, r, xt=0.0, yt=0.0)
    p1 = _canvas_path_xy(n_ref, r, xt=0.0, yt=0.0, a=n_ref.a + da)
    ch0, ch1 = p0[-1] - p0[0], p1[-1] - p1[0]
    th = math.atan2(float(ch1[1]), float(ch1[0])) - math.atan2(float(ch0[1]), float(ch0[0]))
    c, s_ = math.cos(th), math.sin(th)
    start = _canvas_path_xy(piece, r)[0] - cr
    start = cn + torch.stack([c * start[0] - s_ * start[1], s_ * start[0] + c * start[1]])
    piece.a = complete.a + da
    _place(piece, start, r)


# --------------------------------------------------------------------------
# Whole sequence
# --------------------------------------------------------------------------

def check_trims(seq: Sequence) -> None:
    key_frames = [k.frame for k in seq.keys]
    neighbors = set(zip(key_frames[:-1], key_frames[1:]))
    seen: Dict[Tuple[int, int], set] = {}
    for t in seq.trims:
        if t.span not in neighbors:
            raise ValueError(f"trim {t.span[0]}-{t.span[1]}: frames {t.span[0]} and {t.span[1]} "
                             "are not neighboring keyframes")
        dup = seen.setdefault(t.span, set()) & set(t.ids)
        if dup:
            raise ValueError(f"trim {t.span[0]}-{t.span[1]}: {sorted(dup)} trimmed twice in the same span")
        seen[t.span] |= set(t.ids)


def valid_ids(seq: Sequence, t: Trim) -> Tuple[List[str], List[str]]:
    """(ids that can draw on / off in this span, ids that can't: wrong keys)."""
    a, b = t.span
    ka = next(k for k in seq.keys if k.frame == a)
    kb = next(k for k in seq.keys if k.frame == b)
    ok = [i for i in t.ids if (i not in ka.strokes and i in kb.strokes) == (t.mode == "draw_on")
          and (i in ka.strokes) != (i in kb.strokes)]
    return ok, [i for i in t.ids if i not in ok]


def trimmed(seq: Sequence) -> Dict[Tuple[str, Tuple[int, int]], str]:
    """{(stroke id, span): mode} for every valid trim."""
    out = {}
    for t in seq.trims:
        for i in valid_ids(seq, t)[0]:
            out[(i, t.span)] = t.mode
    return out


def apply_trims(seq: Sequence, frames: List[Frame], r: Renderer, te: Dict[int, float],
                warnings: Optional[List[str]] = None) -> None:
    """Replace the trimmed strokes on the in-betweens of their span by partial strokes (in place).
    Expects `frames` after interpolation and motion paths."""
    from .motion_path import anchor, members

    check_trims(seq)
    warnings = warnings if warnings is not None else []
    by_frame = {f.frame: f for f in frames}
    for t in seq.trims:
        a, b = t.span
        ids, bad = valid_ids(seq, t)
        if bad:
            what = "in the second keyframe only" if t.mode == "draw_on" else "in the first keyframe only"
            warnings.append(f"{'Draw on' if t.mode == 'draw_on' else 'Draw off'} {a + 1}–{b + 1}: "
                            f"{', '.join(bad)} must be {what}; ignored.")
        group_paths = [p for p in seq.paths if p.span == t.span]
        for sid in ids:
            pins = sorted(e.frame for e in seq.edits if a < e.frame < b and sid in e.strokes)
            # Frames that are partial, each with the frame where the stroke is complete.
            jobs = []  # (u, f, complete frame)
            for f in range(a + 1, b):
                if f in pins:
                    continue
                if t.mode == "draw_on":
                    if any(p < f for p in pins):
                        continue  # after a pin: interpolated as usual
                    ref = min([p for p in pins if p > f] + [b])
                    u = (te[f] - te[a]) / (te[ref] - te[a])
                else:
                    if any(p > f for p in pins):
                        continue
                    ref = max([p for p in pins if p < f] + [a])
                    u = 1 - (te[f] - te[ref]) / (te[b] - te[ref])
                jobs.append((u, f, ref))
            # Fit the longest pieces first, each warm-started from the previous one.
            init: Dict[int, Tuple] = {}
            worst = 0.0
            for u, f, ref in sorted(jobs, reverse=True):
                complete = by_frame[ref].strokes[sid]
                piece, err, init[ref] = trim_piece(complete, u, r, init.get(ref))
                worst = max(worst, err)
                if piece is None:
                    by_frame[f].strokes.pop(sid, None)
                    continue
                path = next((p for p in group_paths if sid in p.ids), None)
                group = members(seq, path)[0] if path else []
                if group:  # moves with its motion-path group's anchor
                    ar = anchor([by_frame[ref].strokes[i] for i in group], r)
                    af = anchor([by_frame[f].strokes[i] for i in group], r)
                    piece.xt += af[0] - ar[0]
                    piece.yt += af[1] - ar[1]
                else:  # moves with the nearest stroke present in both keys
                    ka = next(k for k in seq.keys if k.frame == a)
                    kb = next(k for k in seq.keys if k.frame == b)
                    cands = [i for i in by_frame[ref].strokes if i in ka.strokes and i in kb.strokes]
                    if cands:
                        n = min(cands, key=lambda i: _distance(complete, by_frame[ref].strokes[i], r))
                        follow(piece, complete, by_frame[ref].strokes[n], by_frame[f].strokes[n], r)
                piece.xt, piece.yt = min(max(piece.xt, 0.0), 1.0), min(max(piece.yt, 0.0), 1.0)
                by_frame[f].strokes[sid] = piece
            if worst > 0.5:
                warnings.append(f"Stroke {sid}: drawing {'on' if t.mode == 'draw_on' else 'off'} follows its "
                                f"shape within {worst:.1f} mm.")
