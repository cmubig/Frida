"""Measure how evenly painted width changes across in-betweens ("popping").

For every stroke and frame, the stroke is rendered in a canonical window and
its painted width is read off the image at several places along the stroke
(the same definition refinement targets, measured at a higher resolution);
we use the mean over those places.
Between two fixed frames (keys or artist-locked strokes) an ideal in-between
changes width evenly in eased time. We report:

  width_dev_max_mm   largest deviation from that even change, in mm
  width_dev_mean_mm  mean deviation, in mm
  roughness_mm       sum of |second difference| of width over frames, in mm

Lower is smoother. Values come from the *stroke model*, so they are only as
real as the model: use a model calibrated on real strokes for claims about
real paintings, and measure photographed frames for the physical result.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .interpolate import eased_time
from .refine import RefineConfig, _canonical_setup, width_profile
from .render import Renderer
from .strokes import Frame, Sequence, stroke_tensors


def stroke_widths(seq: Sequence, frames: List[Frame], r: Renderer, window_px: int = 128) -> Dict[str, dict]:
    cfg = RefineConfig(window_px=window_px, profile_points=12, profile_samples=41)
    out = {}
    ids = sorted({sid for fr in frames for sid in fr.strokes})
    with torch.no_grad():
        for sid in ids:
            run = [fi for fi, fr in enumerate(frames) if sid in fr.strokes]
            ps = [stroke_tensors(frames[fi].strokes[sid]) for fi in run]
            grid, poses = _canonical_setup(r, ps, cfg)
            widths = [float(width_profile(r, p, grid, pose, cfg).mean()) for p, pose in zip(ps, poses)]
            fixed = [frames[fi].kind == "key" or sid in frames[fi].locked for fi in run]
            out[sid] = {"frames": [frames[fi].frame for fi in run], "width_m": widths, "fixed": fixed}
    return out


def popping(seq: Sequence, frames: List[Frame], r: Renderer) -> dict:
    te = eased_time(seq)
    per = {}
    for sid, d in stroke_widths(seq, frames, r).items():
        fr, w, fixed = d["frames"], d["width_m"], d["fixed"]
        anchors = [i for i, f in enumerate(fixed) if f]
        devs = []
        for a, b in zip(anchors[:-1], anchors[1:]):
            for i in range(a + 1, b):
                t = (te[fr[i]] - te[fr[a]]) / max(te[fr[b]] - te[fr[a]], 1e-9)
                devs.append(abs(w[i] - (w[a] + (w[b] - w[a]) * t)))
        rough = sum(abs(w[i - 1] - 2 * w[i] + w[i + 1]) for i in range(1, len(w) - 1))
        per[sid] = {
            "width_mm": [round(x * 1000, 3) for x in w],
            "width_dev_max_mm": round(max(devs) * 1000, 3) if devs else 0.0,
            "width_dev_mean_mm": round(sum(devs) / len(devs) * 1000, 3) if devs else 0.0,
            "roughness_mm": round(rough * 1000, 3),
        }
    keys = ("width_dev_max_mm", "width_dev_mean_mm", "roughness_mm")
    summary = {k: round(sum(v[k] for v in per.values()) / max(len(per), 1), 3) for k in keys}
    return {"summary_mean_over_strokes": summary, "strokes": per}
