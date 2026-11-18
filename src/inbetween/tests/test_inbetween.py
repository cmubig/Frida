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
