"""Geometric in-betweening (the baselines that refinement starts from).

Two methods:

* ``linear``: every parameter interpolated linearly (rotation on the circle).
* ``arc``: BetweenIT-style (Whited et al. 2010). The stroke's start point moves
  on a logarithmic spiral (rotation + uniform scale about a fixed center), so
  rotating strokes travel on arcs instead of cutting corners; the stroke's own
  shape is interpolated intrinsically (edge lengths and turning angles,
  Sederberg et al. 1993). Corner roundness and pressure are linear.

Correspondence comes from stable stroke IDs. Artist edits on in-between frames
act as extra per-stroke keys: a stroke locked on frame f is interpolated
key -> f -> key, so neighbors re-flow around the edit.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

from .render import Renderer
from .strokes import Frame, Sequence, Stroke, stroke_tensors, wrap_angle

EASINGS = {
    "linear": lambda t: t,
    "ease_in": lambda t: t * t,
    "ease_out": lambda t: 1 - (1 - t) * (1 - t),
    "ease_in_out": lambda t: 3 * t * t - 2 * t * t * t,
}


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def eased_time(seq: Sequence) -> Dict[int, float]:
    """Global parameter for every frame: key k sits at k, eased within segments."""
    out = {}
    for i, (ka, kb) in enumerate(zip(seq.keys[:-1], seq.keys[1:])):
        ease = EASINGS[seq.timing.get(f"{ka.frame}-{kb.frame}", "linear")]
        span = kb.frame - ka.frame
        for f in range(ka.frame, kb.frame + 1):
            out[f] = i + ease((f - ka.frame) / span)
    return out


# --------------------------------------------------------------------------
# Per-stroke interpolation between two states
# --------------------------------------------------------------------------

def _lerp(a, b, t):
    return [x + (y - x) * t for x, y in zip(a, b)]


def interp_linear(A: Stroke, B: Stroke, t: float) -> Stroke:
    s = A.copy()
    s.vertices = [_lerp(va, vb, t) for va, vb in zip(A.vertices, B.vertices)]
    s.kappa = _lerp(A.kappa, B.kappa, t)
    s.z = _lerp(A.z, B.z, t)
    s.xt = A.xt + (B.xt - A.xt) * t
    s.yt = A.yt + (B.yt - A.yt) * t
    s.a = A.a + wrap_angle(B.a - A.a) * t
    return s


def _intrinsic(vertices) -> Tuple[List[float], float, List[float]]:
    """Edge lengths, angle of the first edge, turning angles at interior corners."""
    v = torch.tensor(vertices)
    e = v[1:] - v[:-1]
    lengths = e.norm(dim=1).tolist()
    angles = torch.atan2(e[:, 1], e[:, 0]).tolist()
    turns = [wrap_angle(angles[i] - angles[i - 1]) for i in range(1, len(angles))]
    return lengths, angles[0], turns


def _rebuild(v0, lengths, theta0, turns):
    pts = [list(v0)]
    theta = theta0
    for i, L in enumerate(lengths):
        if i > 0:
            theta += turns[i - 1]
        x, y = pts[-1]
        pts.append([x + L * math.cos(theta), y + L * math.sin(theta)])
    return pts


def interp_shape_intrinsic(A: Stroke, B: Stroke, t: float) -> List[List[float]]:
    la, tha, tua = _intrinsic(A.vertices)
    lb, thb, tub = _intrinsic(B.vertices)
    lengths = _lerp(la, lb, t)
    theta0 = tha + wrap_angle(thb - tha) * t
    turns = [x + wrap_angle(y - x) * t for x, y in zip(tua, tub)]
    v0 = _lerp(A.vertices[0], B.vertices[0], t)
    return _rebuild(v0, lengths, theta0, turns)


def _spiral(P0: torch.Tensor, P1: torch.Tensor, alpha: float, rho: float, t: float) -> torch.Tensor:
    """Point on the log spiral taking P0 to P1 (2-vectors) with total rotation
    alpha and scale rho, at parameter t. Falls back to a line for tiny rotation."""
    if abs(alpha) < 1e-3 and abs(rho - 1) < 1e-3:
        return P0 + (P1 - P0) * t
    c, s = math.cos(alpha), math.sin(alpha)
    M = torch.tensor([[rho * c, -rho * s], [rho * s, rho * c]], dtype=P0.dtype)
    I = torch.eye(2, dtype=P0.dtype)
    try:
        F = torch.linalg.solve(I - M, P1 - M @ P0)
    except RuntimeError:
        return P0 + (P1 - P0) * t
    ct, st = math.cos(alpha * t), math.sin(alpha * t)
    Mt = (rho ** t) * torch.tensor([[ct, -st], [st, ct]], dtype=P0.dtype)
    return F + Mt @ (P0 - F)


def interp_arc(A: Stroke, B: Stroke, t: float, r: Renderer) -> Stroke:
    s = A.copy()
    s.vertices = interp_shape_intrinsic(A, B, t)
    s.kappa = _lerp(A.kappa, B.kappa, t)
    s.z = _lerp(A.z, B.z, t)
    s.a = A.a + wrap_angle(B.a - A.a) * t

    with torch.no_grad():
        pa, pb = stroke_tensors(A), stroke_tensors(B)
        ca = r.to_metric(r.path_canvas(pa))  # (32, 2) metric (y, x)
        cb = r.to_metric(r.path_canvas(pb))
        # Work in (x, y) with y down; the spiral's rotation is measured from
        # the stroke chord, in the same space as the points.
        xy = lambda c: torch.stack([c[:, 1], c[:, 0]], dim=-1)
        ca, cb = xy(ca), xy(cb)
        chord_a, chord_b = ca[-1] - ca[0], cb[-1] - cb[0]
        alpha = wrap_angle(math.atan2(chord_b[1], chord_b[0]) - math.atan2(chord_a[1], chord_a[0]))
        la, lb = float(chord_a.norm()), float(chord_b.norm())
        rho = lb / la if la > 1e-6 and lb > 1e-6 else 1.0
        start_t = _spiral(ca[0], cb[0], alpha, rho, t)  # desired start point (x, y) metric

        # Place the interpolated stroke so its start lands on the spiral.
        ps = stroke_tensors(s)
        zero = torch.zeros(1)
        c0 = r.path_canvas(ps, xt=zero, yt=zero)[0]  # (y, x) proportions with no offset
        target = r.from_metric(torch.stack([start_t[1], start_t[0]]))
        s.yt = float(target[0] - c0[0])
        s.xt = float(target[1] - c0[1])
    return s


# --------------------------------------------------------------------------
# Whole sequence
# --------------------------------------------------------------------------

def _anchors(seq: Sequence) -> Dict[str, List[Tuple[int, Stroke]]]:
    """For every stroke id: the frames where its state is fixed (keys + artist edits)."""
    anchors: Dict[str, List[Tuple[int, Stroke]]] = {}
    for fr in list(seq.keys) + list(seq.edits):
        for sid, s in fr.strokes.items():
            anchors.setdefault(sid, []).append((fr.frame, s))
    for sid in anchors:
        anchors[sid].sort(key=lambda x: x[0])
    return anchors


def _present(seq: Sequence, sid: str, f: int) -> bool:
    """Is stroke sid drawn on frame f? Strokes missing from a neighboring key pop
    in or out at the middle of that key segment."""
    keys = seq.keys
    if f <= keys[0].frame:
        return sid in keys[0].strokes
    if f >= keys[-1].frame:
        return sid in keys[-1].strokes
    for ka, kb in zip(keys[:-1], keys[1:]):
        if ka.frame <= f <= kb.frame:
            ina, inb = sid in ka.strokes, sid in kb.strokes
            if ina and inb:
                return True
            if not ina and not inb:
                return any(e.frame == f and sid in e.strokes for e in seq.edits)
            mid = (ka.frame + kb.frame) / 2
            return f < mid if ina else f >= mid
    return False


def interpolate(seq: Sequence, method: str = "arc", r: Renderer = None) -> List[Frame]:
    """Produce every frame from the first to the last key."""
    if method not in ("linear", "arc"):
        raise ValueError("method must be 'linear' or 'arc'")
    if method == "arc" and r is None:
        raise ValueError("arc interpolation needs a Renderer (for canvas geometry)")
    te = eased_time(seq)
    anchors = _anchors(seq)
    key_frames = {k.frame: k for k in seq.keys}
    edit_frames = {e.frame: e for e in seq.edits}

    frames = []
    for f in range(seq.keys[0].frame, seq.keys[-1].frame + 1):
        if f in key_frames:
            k = key_frames[f]
            frames.append(Frame(frame=f, kind="key", strokes={i: s.copy() for i, s in k.strokes.items()},
                                locked=list(k.strokes.keys())))
            continue
        strokes, locked = {}, []
        for sid, anc in anchors.items():
            if not _present(seq, sid, f):
                continue
            if f in edit_frames and sid in edit_frames[f].strokes:
                strokes[sid] = edit_frames[f].strokes[sid].copy()
                locked.append(sid)
                continue
            before = [x for x in anc if x[0] < f]
            after = [x for x in anc if x[0] > f]
            if not before or not after:  # only one side exists: hold that state
                strokes[sid] = (before[-1] if before else after[0])[1].copy()
                continue
            (fa, A), (fb, B) = before[-1], after[0]
            t = (te[f] - te[fa]) / (te[fb] - te[fa])
            strokes[sid] = interp_linear(A, B, t) if method == "linear" else interp_arc(A, B, t, r)
        frames.append(Frame(frame=f, kind="inbetween", strokes=strokes, locked=locked))
    return frames
