"""Write the example keyframe files used for testing and demos.

Run from src/:  python -m inbetween.examples.make_examples

bend.json covers the cases the pipeline must handle:
  stem   - shape change (straight -> bent), constant pressure
  swell  - pressure change 0.15 -> 0.85 while translating (tests width popping)
  swing  - rigid rotation of 70 degrees about a pivot off the stroke (tests arcs)
  leaf   - appears only in the second key (tests strokes entering)
"""

import json
import math
import os

import torch

from inbetween.render import Renderer
from inbetween.strokes import FORMAT_NAME, FORMAT_VERSION, Stroke, stroke_tensors

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "..", "caches", "vae_sharpie_final")
BLACK = [0.05, 0.05, 0.05]


def rotated_about(r: Renderer, A: Stroke, pivot_xy, deg) -> Stroke:
    """Return a copy of A rigidly rotated about a canvas point (proportions x, y)."""
    pa = stroke_tensors(A)
    target = r.to_metric(r.path_canvas(pa))  # (32, 2) metric (y, x)
    piv = r.to_metric(torch.tensor([pivot_xy[1], pivot_xy[0]]))
    best = None
    ang = math.radians(deg)
    c, s = math.cos(ang), math.sin(ang)
    d = target - piv
    rot = torch.stack([piv[0] + s * d[:, 1] + c * d[:, 0], piv[1] + c * d[:, 1] - s * d[:, 0]], -1)
    # FRIDA's pose angle may turn the stroke either way on screen; try both.
    for sign in (1, -1):
        B = A.copy()
        B.a = A.a + sign * ang
        pb = stroke_tensors(B)
        zero = torch.zeros(1)
        c0 = r.path_canvas(pb, xt=zero, yt=zero)[0]
        start = r.from_metric(rot[0])
        B.yt, B.xt = float(start[0] - c0[0]), float(start[1] - c0[1])
        err = float((r.to_metric(r.path_canvas(stroke_tensors(B))) - rot).norm(dim=1).max())
        if best is None or err < best[0]:
            best = (err, B)
    assert best[0] < 1e-4, f"rotation fit error {best[0]}"
    return best[1]


def main():
    r = Renderer(CACHE)
    straight = [[0.0, 0.0], [0.02, 0.0], [0.04, 0.0], [0.06, 0.0]]

    stem_a = Stroke("stem", straight, [0.5, 0.5], [0.6] * 4, xt=0.25, yt=0.75, a=math.pi / 2, order=0, color=BLACK)
    stem_b = stem_a.copy()
    stem_b.vertices = [[0.0, 0.0], [0.02, 0.0], [0.036, 0.012], [0.046, 0.03]]
    stem_b.kappa = [0.85, 0.85]

    swell_a = Stroke("swell", straight, [0.5, 0.5], [0.15] * 4, xt=0.45, yt=0.55, a=0.0, order=1, color=BLACK)
    swell_b = swell_a.copy()
    swell_b.z = [0.85] * 4
    swell_b.xt, swell_b.yt = 0.5, 0.5

    swing_a = Stroke("swing", [[0.0, 0.0], [0.015, 0.0], [0.03, 0.0], [0.045, 0.0]], [0.5, 0.5], [0.6] * 4,
                     xt=0.72, yt=0.3, a=-math.pi / 2, order=2, color=BLACK)
    swing_b = rotated_about(r, swing_a, pivot_xy=(0.80, 0.18), deg=70)

    leaf_b = Stroke("leaf", [[0.0, 0.0], [0.01, -0.006], [0.02, -0.006], [0.03, 0.0]], [0.9, 0.9], [0.5] * 4,
                    xt=0.33, yt=0.45, a=0.4, order=3, color=BLACK)

    seq = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "description": "Test sequence: shape change, pressure change, rotation, entering stroke.",
        "cache_dir": "caches/vae_sharpie_final",
        "keys": [
            {"frame": 0, "strokes": [s.to_dict() for s in (stem_a, swell_a, swing_a)]},
            {"frame": 10, "strokes": [s.to_dict() for s in (stem_b, swell_b, swing_b, leaf_b)]},
        ],
        "timing": {"0-10": "linear"},
    }
    out = os.path.join(HERE, "bend.json")
    with open(out, "w") as f:
        json.dump(seq, f, indent=1)

    # Same sequence plus one artist edit on frame 5 (swell made lighter),
    # to show refinement and interpolation respecting artist fixes.
    edited = json.loads(json.dumps(seq))
    fix = swell_a.copy()
    fix.z = [0.3] * 4
    fix.xt, fix.yt = 0.475, 0.525
    edited["description"] = "bend.json plus an artist fix on frame 5."
    edited["edits"] = [{"frame": 5, "strokes": [fix.to_dict()]}]
    with open(os.path.join(HERE, "bend_with_edit.json"), "w") as f:
        json.dump(edited, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
