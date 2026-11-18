"""The editor's JavaScript stroke geometry must match FRIDA's Python exactly,
or the stage would show strokes somewhere other than where the robot paints.
Needs Node.js; skipped otherwise."""

import json
import math
import os
import shutil
import subprocess

import pytest
import torch

from inbetween.fit import visible_z_min
from inbetween.render import Renderer
from inbetween.strokes import Stroke, stroke_tensors

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.join(HERE, "..", "editor", "geometry.js")
CACHE = os.path.join(HERE, "..", "..", "caches", "vae_sharpie_final")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_js_geometry_matches_python():
    r = Renderer(CACHE)
    m = r.model
    cfg = {"x_m": float(m.x_m), "y_m": float(m.y_m), "x_b": float(m.x_b), "y_b": float(m.y_b),
           "min_stroke_z": r.settings.MIN_STROKE_Z, "thick_mult": float(m.renderer.thick_mult),
           "thick_bias": float(m.renderer.thick_bias), "response": "model"}
    g = torch.Generator().manual_seed(3)
    strokes = []
    for i in range(60):
        v = torch.cumsum(torch.randn(4, 2, generator=g) * 0.015, 0)
        if i % 10 == 0:  # some nearly straight strokes
            v = torch.stack([torch.tensor([0.01 * k, 0.002 * k]) for k in range(4)]) + 1e-5 * torch.randn(4, 2, generator=g)
        strokes.append(Stroke(str(i), v.tolist(), torch.rand(2, generator=g).tolist(),
                              torch.rand(4, generator=g).tolist(), xt=float(torch.rand(1, generator=g)),
                              yt=float(torch.rand(1, generator=g)),
                              a=float(torch.rand(1, generator=g) * 2 * math.pi - math.pi)).to_dict())
    script = f"""
const Geo = require({json.dumps(os.path.abspath(GEO))});
const cfg = {json.dumps(cfg)};
const strokes = {json.dumps(strokes)};
const out = strokes.map(s => {{
  const c = Geo.canvasPath(s, cfg);
  const back = Geo.toLocal(c[5], s, cfg);
  return {{c, back, local5: Geo.localPath(s)[5]}};
}});
console.log(JSON.stringify(out));
"""
    res = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    for sd, js in zip(strokes, res):
        s = Stroke.from_dict(sd)
        p = stroke_tensors(s)
        py = r.path_canvas(p)  # (32, 2) (y, x)
        c = torch.tensor(js["c"])
        assert torch.allclose(c[:, 0], py[:, 1], atol=1e-5), f"stroke {s.id} x"
        assert torch.allclose(c[:, 1], py[:, 0], atol=1e-5), f"stroke {s.id} y"
        assert torch.allclose(torch.tensor(js["back"]), torch.tensor(js["local5"]), atol=1e-6)
        path = r.path(p)
        heights = (path[:, 2] * 2 + 0.5) / 64
        thick = (heights * m.renderer.thick_mult + m.renderer.thick_bias).clamp_min(0)
        assert torch.allclose(c[:, 2], thick.float(), atol=1e-5), f"stroke {s.id} thickness"
