"""Run from src/:  python -m pytest -q inbetween/tests"""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from brush_stroke import BrushStroke
from inbetween.interpolate import interpolate
from inbetween.metrics import popping
from inbetween.refine import RefineConfig, refine
from inbetween.render import Renderer, sigmoid_response
from inbetween.strokes import Stroke, load_sequence, stroke_tensors

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")
CACHE = os.path.join(HERE, "..", "..", "caches", "vae_sharpie_final")


@pytest.fixture(scope="module")
def r():
    return Renderer(CACHE)


@pytest.fixture(scope="module")
def seq():
    return load_sequence(os.path.join(EX, "bend.json"))


def test_path_matches_brushstroke(r):
    s = Stroke("a", [[0, 0], [0.02, 0.01], [0.04, 0.0], [0.06, 0.01]], [0.3, 0.7], [0.2, 0.4, 0.6, 0.8],
               xt=0.4, yt=0.5, a=0.3)
    p = stroke_tensors(s)
    opt = SimpleNamespace(MAX_STROKE_LENGTH=0.05, MIN_STROKE_Z=0.1, max_length_before_new_paint=1, ink=True)
    bs = BrushStroke(opt, vertices=p["vertices"], radii=p["radii"], z=p["z"], xt=p["xt"], yt=p["yt"],
                     a=p["a"], ink=True, device="cpu")
    assert torch.allclose(bs.get_path(), r.path(p), atol=1e-6)


@pytest.mark.parametrize("method", ["linear", "arc"])
def test_keys_reproduced_exactly(seq, r, method):
    frames = interpolate(seq, method, r)
    assert [f.frame for f in frames] == list(range(0, 11))
    for k in seq.keys:
        out = next(f for f in frames if f.frame == k.frame)
        assert out.kind == "key"
        for sid, s in k.strokes.items():
            assert out.strokes[sid].to_dict() == s.to_dict()


def test_entering_stroke_pops_in_at_midpoint(seq, r):
    frames = interpolate(seq, "arc", r)
    present = [("leaf" in f.strokes) for f in frames]
    assert present == [False] * 5 + [True] * 6


def test_arc_keeps_rigid_rotation_rigid(seq, r):
    """The swing stroke only rotates about a pivot: arc in-betweens keep its length."""
    frames = interpolate(seq, "arc", r)
    lengths = []
    for f in frames:
        m = r.to_metric(r.path_canvas(stroke_tensors(f.strokes["swing"])))
        lengths.append(float((m[1:] - m[:-1]).norm(dim=1).sum()))
    assert max(lengths) - min(lengths) < 1e-4


def test_edit_is_respected(r):
    seq = load_sequence(os.path.join(EX, "bend_with_edit.json"))
    edit = seq.edits[0]
    for method in ("linear", "arc"):
        frames = interpolate(seq, method, r)
        f5 = next(f for f in frames if f.frame == edit.frame)
        assert "swell" in f5.locked
        assert f5.strokes["swell"].to_dict() == edit.strokes["swell"].to_dict()
    refined, _ = refine(seq, interpolate(seq, "arc", r), r, RefineConfig(iters=5, verbose=False))
    f5 = next(f for f in refined if f.frame == edit.frame)
    assert f5.strokes["swell"].to_dict() == edit.strokes["swell"].to_dict()


def test_refinement_reduces_popping_and_keeps_motion(seq):
    r = Renderer(CACHE, response="placeholder_sigmoid")
    arc = interpolate(seq, "arc", r)
    refined, _ = refine(seq, arc, r, RefineConfig(verbose=False))
    before = popping(seq, arc, r)["strokes"]["swell"]["width_dev_max_mm"]
    after = popping(seq, refined, r)["strokes"]["swell"]["width_dev_max_mm"]
    assert after < 0.5 * before
    for a, b in zip(arc, refined):  # only pressure may change
        for sid in a.strokes:
            sa, sb = a.strokes[sid], b.strokes[sid]
            assert (sa.vertices, sa.xt, sa.yt, sa.a) == (sb.vertices, sb.xt, sb.yt, sb.a)
            assert all(0.02 <= z <= 0.98 for z in sb.z)
        if a.kind == "key":
            assert all(a.strokes[s].to_dict() == b.strokes[s].to_dict() for s in a.strokes)


def test_sigmoid_response_is_monotone_unit_range():
    z = torch.linspace(0, 1, 101)
    y = sigmoid_response(z)
    assert abs(float(y[0])) < 1e-6 and abs(float(y[-1]) - 1) < 1e-6
    assert bool((y[1:] >= y[:-1]).all())


def test_near_straight_strokes_do_not_blow_up():
    """Regression: nearly straight corners used to produce paths meters away."""
    from inbetween.strokes import logit, stroke_path
    g = torch.Generator().manual_seed(0)
    for _ in range(500):
        L = 0.02 + 0.08 * float(torch.rand(1, generator=g))
        ang = 2 * math.pi * float(torch.rand(1, generator=g))
        d = torch.tensor([math.cos(ang), math.sin(ang)])
        noise = torch.randn(4, 2, generator=g) * 10 ** (-2 - 5 * float(torch.rand(1, generator=g)))
        v = torch.stack([d * L * k / 3 for k in range(4)]) + noise
        p = stroke_path(v.float(), logit(torch.rand(2, generator=g) * 0.98 + 0.01), torch.full((4,), 0.5), 0.1)
        assert torch.isfinite(p).all() and float(p[:, :2].abs().max()) < 0.2


def test_fit_follows_drawing(r):
    from inbetween.fit import fit_stroke
    pts = [[0.3 + 0.25 * t, 0.5 - 0.15 * math.sin(math.pi * t)] for t in [i / 40 for i in range(41)]]
    s, err = fit_stroke(pts, [0.2 + 0.6 * i / 40 for i in range(41)], r)
    assert err < 1.0  # mm
    c = r.path_canvas(stroke_tensors(s))  # (y, x) proportions
    assert abs(float(c[0, 1]) - pts[0][0]) < 2e-3 and abs(float(c[-1, 1]) - pts[-1][0]) < 2e-3
    assert s.z[0] < s.z[-1]  # pressure rose along the drawing


def test_complex_mark_is_split(r):
    from inbetween.fit import fit_mark
    # A simple arc stays one stroke.
    arc = [[0.3 + 0.25 * t, 0.5 - 0.15 * math.sin(math.pi * t)] for t in [i / 40 for i in range(41)]]
    assert len(fit_mark(arc, [0.5] * 41, r)) == 1
    # A zigzag with four sharp turns cannot be one 4-corner stroke: it is split,
    # every piece fits closely, and neighboring pieces meet at the cut.
    zig = []
    for seg in range(5):
        for i in range(12):
            t = i / 12
            zig.append([0.2 + 0.12 * (seg + t), 0.4 + (0.12 * t if seg % 2 == 0 else 0.12 * (1 - t))])
    zig.append([0.8, 0.52])
    pieces = fit_mark(zig, [0.5] * len(zig), r, max_err_mm=1.0)
    assert len(pieces) > 1
    assert max(e for _, e in pieces) < 1.5
    ends = [r.path_canvas(stroke_tensors(s)) for s, _ in pieces]
    for a, b in zip(ends, ends[1:]):
        gap = (a[-1] - b[0]).abs().max()
        assert float(gap) < 0.01  # proportions of the canvas


def test_export_frida_plans(r, seq, tmp_path):
    """Each frame becomes a FRIDA plan (Painting, palette) whose strokes pop in painting order."""
    import pickle
    import subprocess
    import sys
    from inbetween.strokes import save_frames
    frames = interpolate(seq, "arc", r)
    fjson = tmp_path / "frames.json"
    save_frames(str(fjson), frames)
    src = os.path.join(HERE, "..", "..")
    res = subprocess.run([sys.executable, "-m", "inbetween.export_plans", "--frames", str(fjson),
                          "--out", str(tmp_path / "plans"), "--only", "0,5",
                          "--cache_dir", "caches/vae_sharpie_final",
                          "--materials_json", "../materials_vae_sharpie.json", "--ink"],
                         cwd=src, capture_output=True, text=True)
    if res.returncode != 0 and "materials" in res.stderr.lower():
        pytest.skip("FRIDA materials JSON not available")
    assert res.returncode == 0, res.stderr[-2000:]
    painting, palette = pickle.load(open(tmp_path / "plans" / "frame_005.pkl", "rb"))
    planned = next(f for f in frames if f.frame == 5).ordered()
    assert len(painting) == len(planned)
    first = painting.pop()
    assert abs(float(first.xt.detach()) - planned[0].xt) < 1e-5
    assert (tmp_path / "plans" / "frame_000.png").exists()


def test_import_raster_frames(tmp_path):
    """Two drawings sharing a static ground line: the ground becomes one background
    stroke set, each drawing's own line is traced and fitted closely."""
    import json
    from PIL import Image, ImageDraw
    from inbetween.import_frames import main as import_main
    for i, x in enumerate([200, 200, 260]):  # frames 0 and 1 identical (a hold)
        im = Image.new("RGB", (640, 360), "white")
        d = ImageDraw.Draw(im)
        d.line([(20, 300), (620, 300)], fill="black", width=3)              # static ground
        d.line([(x, 280), (x + 40, 200), (x + 80, 280)], fill="black", width=3)  # moving figure
        im.save(tmp_path / f"f_{i:03d}.png")
    out = tmp_path / "out"
    import_main([str(tmp_path / "f_*.png"), "--out", str(out), "--cache_dir", CACHE])
    rep = json.load(open(out / "report.json"))
    assert [d["hold"] for d in rep["drawings"]] == [2, 1]
    assert rep["background"]["strokes"] >= 1
    assert all(1 <= d["strokes"] <= 4 for d in rep["drawings"])
    assert rep["total"]["err_mm_max"] < 1.0
    keys = json.load(open(out / "drawings.json"))["keys"]
    assert [k["frame"] for k in keys] == [0, 2]
    assert (out / "compare" / "drawing_02.png").exists()
