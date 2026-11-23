"""Motion paths: the animator draws the path a group of strokes travels between
two neighboring keyframes (After Effects' motion path, "drawing the arc").

Stored in the keyframe file (version 2) next to "timing", never painted:

    "paths": [{"span": "0-10", "ids": ["s3", "s4"], "points": [[x, y], ...]}]

How it is applied, after the usual linear / arc interpolation:

1. Members: the path's strokes that are in both keys and not pinned on any
   frame of the span. A pinned stroke keeps its pin and does not follow the
   path, and it is left out of the anchor on *every* frame, so the anchor
   never jumps at the pin frame.
2. Anchor: the mean of the members' path points (their centre), in canvas
   proportions. The editor's `center()` is the same per-stroke mean.
3. The drawn polyline is fitted so its ends land exactly on the anchor at the
   two keys (a rotation + uniform scale + shift; for a nearly closed path, or
   ends very close together, a shift blended along the path instead). Editing a
   key later bends the path with it.
4. On an in-between at eased time t, the anchor goes to the point at fraction
   t of the fitted path's length, by shifting every member's xt, yt by the same
   amount. Rotation and shape still come from the interpolation method.
5. Robot limits: like BrushStroke.make_valid, xt and yt stay in [0, 1]. The
   shift is limited so the whole group stays on the paper (it stays rigid);
   a warning says when a path leaves the paper or a shift was limited.

Keys are never changed (only frames strictly between the keys are shifted),
and stroke id / order are untouched.

The editor (editor/app.js, "motion paths") mirrors fit_path / point_at to draw
the path and its per-frame dots.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence as Seq, Tuple

import torch

from .render import Renderer
from .strokes import Frame, MotionPath, Sequence, Stroke, stroke_tensors

# The fit rotates and scales the drawn path only when both its chord and the keys'
# anchor chord are at least this long (in paper heights) and their ratio is within
# SCALE_LIMIT; otherwise it shifts the path, blending the end offsets along it.
MIN_CHORD = 0.01
SCALE_LIMIT = 4.0

Point = Tuple[float, float]


def anchor(strokes: Seq[Stroke], r: Renderer) -> Point:
    """Mean of the strokes' centres (mean of each stroke's path points), canvas proportions (x, y)."""
    with torch.no_grad():
        cs = [r.path_canvas(stroke_tensors(s)).mean(dim=0) for s in strokes]  # (y, x)
        c = torch.stack(cs).mean(dim=0)
    return float(c[1]), float(c[0])


def _cum_lengths(q: List[Point]) -> List[float]:
    out = [0.0]
    for (x0, y0), (x1, y1) in zip(q[:-1], q[1:]):
        out.append(out[-1] + math.hypot(x1 - x0, y1 - y0))
    return out


def fit_path(points: Seq[Seq[float]], a0: Point, a1: Point, aspect: float) -> List[Point]:
    """The drawn polyline, fitted so it starts at a0 and ends at a1 (all in canvas
    proportions). `aspect` = canvas width / height, so the fit is done in isotropic units."""
    P = [(x * aspect, y) for x, y in points]
    A0, A1 = (a0[0] * aspect, a0[1]), (a1[0] * aspect, a1[1])
    cd = (P[-1][0] - P[0][0], P[-1][1] - P[0][1])
    ca = (A1[0] - A0[0], A1[1] - A0[1])
    ld, la = math.hypot(*cd), math.hypot(*ca)
    if ld >= MIN_CHORD and la >= MIN_CHORD and 1 / SCALE_LIMIT <= la / ld <= SCALE_LIMIT:
        # complex multiply by ca / cd: rotation + uniform scale taking the chord onto the keys' chord
        k = complex(*ca) / complex(*cd)
        q = [complex(px - P[0][0], py - P[0][1]) * k for px, py in P]
        out = [(A0[0] + z.real, A0[1] + z.imag) for z in q]
    else:
        L = _cum_lengths(P)
        tot = L[-1] or 1.0
        d0 = (A0[0] - P[0][0], A0[1] - P[0][1])
        d1 = (A1[0] - P[-1][0], A1[1] - P[-1][1])
        out = [(px + d0[0] + (d1[0] - d0[0]) * l / tot, py + d0[1] + (d1[1] - d0[1]) * l / tot)
               for (px, py), l in zip(P, L)]
    out[0], out[-1] = A0, A1  # exact ends
    return [(x / aspect, y) for x, y in out]


def point_at(q: List[Point], u: float, aspect: float) -> Point:
    """Point at fraction u (0..1) of the path's length (measured on the paper, not in proportions)."""
    iso = [(x * aspect, y) for x, y in q]
    L = _cum_lengths(iso)
    if L[-1] <= 0:
        return q[0]
    s = min(max(u, 0.0), 1.0) * L[-1]
    for i in range(1, len(L)):
        if L[i] >= s:
            seg = L[i] - L[i - 1]
            w = (s - L[i - 1]) / seg if seg > 0 else 0.0
            return (q[i - 1][0] + (q[i][0] - q[i - 1][0]) * w, q[i - 1][1] + (q[i][1] - q[i - 1][1]) * w)
    return q[-1]


def members(seq: Sequence, mp: MotionPath) -> Tuple[List[str], List[str], List[str]]:
    """(members, not in both keys, pinned in the span)."""
    a, b = mp.span
    ka = next(k for k in seq.keys if k.frame == a)
    kb = next(k for k in seq.keys if k.frame == b)
    pinned = {sid for e in seq.edits if a < e.frame < b for sid in e.strokes}
    both = [i for i in mp.ids if i in ka.strokes and i in kb.strokes]
    missing = [i for i in mp.ids if i not in both]
    return [i for i in both if i not in pinned], missing, [i for i in both if i in pinned]


def check_paths(seq: Sequence) -> None:
    """Each path spans two neighboring keys, and a stroke is in at most one path per span."""
    key_frames = [k.frame for k in seq.keys]
    neighbors = set(zip(key_frames[:-1], key_frames[1:]))
    seen: Dict[Tuple[int, int], set] = {}
    for mp in seq.paths:
        if mp.span not in neighbors:
            raise ValueError(f"motion path {mp.span[0]}-{mp.span[1]}: frames {mp.span[0]} and "
                             f"{mp.span[1]} are not neighboring keyframes")
        dup = seen.setdefault(mp.span, set()) & set(mp.ids)
        if dup:
            raise ValueError(f"motion path {mp.span[0]}-{mp.span[1]}: {sorted(dup)} in two paths of the same span")
        seen[mp.span] |= set(mp.ids)


def _span_name(mp: MotionPath) -> str:
    return f"keyframes {mp.span[0] + 1}–{mp.span[1] + 1}"  # timeline numbering, as the editor shows it


def apply_paths(seq: Sequence, frames: List[Frame], r: Renderer, te: Dict[int, float],
                warnings: Optional[List[str]] = None) -> None:
    """Move each path's group along its path on the in-betweens of its span (in place)."""
    check_paths(seq)
    warnings = warnings if warnings is not None else []
    aspect = r.settings.CANVAS_WIDTH_M / r.settings.CANVAS_HEIGHT_M
    by_frame = {f.frame: f for f in frames}
    keys = {k.frame: k for k in seq.keys}
    for mp in seq.paths:
        a, b = mp.span
        ids, missing, pinned = members(seq, mp)
        if missing:
            warnings.append(f"Motion path {_span_name(mp)}: {', '.join(missing)} not in both keyframes, so "
                            f"{'it does' if len(missing) == 1 else 'they do'} not follow the path.")
        if pinned:
            warnings.append(f"Motion path {_span_name(mp)}: {', '.join(pinned)} pinned on an in-between, so "
                            f"{'it keeps its pin' if len(pinned) == 1 else 'they keep their pins'} instead.")
        if not ids:
            continue
        q = fit_path(mp.points, anchor([keys[a].strokes[i] for i in ids], r),
                     anchor([keys[b].strokes[i] for i in ids], r), aspect)
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in q):
            warnings.append(f"Motion path {_span_name(mp)} leaves the paper; its strokes are kept on the paper.")
        limited = False
        for f in range(a + 1, b):
            fr = by_frame[f]
            group = [fr.strokes[i] for i in ids]
            t = (te[f] - te[a]) / (te[b] - te[a])
            tx, ty = point_at(q, t, aspect)
            cx, cy = anchor(group, r)
            d = [tx - cx, ty - cy]
            for axis, attr in ((0, "xt"), (1, "yt")):
                vals = [getattr(s, attr) for s in group]
                lo, hi = max(-v for v in vals), min(1 - v for v in vals)
                if lo <= hi and not lo <= d[axis] <= hi:  # keep the whole group on the paper
                    d[axis] = min(max(d[axis], lo), hi)
                    limited = True
            for s in group:
                x, y = s.xt + d[0], s.yt + d[1]
                s.xt, s.yt = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)  # as BrushStroke.make_valid
                limited |= (s.xt, s.yt) != (x, y)
        if limited:
            warnings.append(f"Motion path {_span_name(mp)}: strokes were stopped at the edge of the paper.")
