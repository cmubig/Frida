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


# --------------------------------------------------------------------------
# Motion paths
# --------------------------------------------------------------------------

def _bow(a, b, lift, n=21):
    """Polyline from a to b bowed upward by `lift` (canvas proportions)."""
    return [[a[0] + (b[0] - a[0]) * i / (n - 1),
             a[1] + (b[1] - a[1]) * i / (n - 1) - lift * (1 - (2 * i / (n - 1) - 1) ** 2)] for i in range(n)]


def _with_path(name, ids, points, timing="ease_in_out"):
    import json
    from inbetween.strokes import sequence_from_dict
    d = json.load(open(os.path.join(EX, name)))
    d["version"] = 2
    d["timing"] = {"0-10": timing}
    d["paths"] = [{"span": "0-10", "ids": ids, "points": points}]
    return sequence_from_dict(d), d


@pytest.mark.parametrize("method", ["linear", "arc"])
def test_motion_path_keys_unchanged_and_anchor_follows_path(r, method):
    from inbetween.interpolate import eased_time
    from inbetween.motion_path import anchor, fit_path, point_at
    ids = ["swell", "swing"]
    seq, d = _with_path("bend.json", ids, _bow((0.55, 0.45), (0.62, 0.35), 0.2))
    from inbetween.strokes import sequence_from_dict
    plain = interpolate(sequence_from_dict({**d, "paths": []}), method, r)  # same timing, no path
    warnings = []
    frames = interpolate(seq, method, r, warnings)
    assert warnings == []
    for k in seq.keys:  # keys come out exactly as drawn
        out = next(f for f in frames if f.frame == k.frame)
        assert {s: x.to_dict() for s, x in out.strokes.items()} == {s: x.to_dict() for s, x in k.strokes.items()}
    aspect = r.settings.CANVAS_WIDTH_M / r.settings.CANVAS_HEIGHT_M
    ka, kb = seq.keys
    q = fit_path(seq.paths[0].points, anchor([ka.strokes[i] for i in ids], r),
                 anchor([kb.strokes[i] for i in ids], r), aspect)
    te = eased_time(seq)
    for f in frames[1:-1]:
        want = point_at(q, te[f.frame], aspect)  # span 0-10 is the first segment: t = te
        got = anchor([f.strokes[i] for i in ids], r)
        assert math.hypot(want[0] - got[0], want[1] - got[1]) < 1e-5, f.frame
    mid, base = frames[5], plain[5]
    assert anchor([mid.strokes[i] for i in ids], r)[1] < anchor([base.strokes[i] for i in ids], r)[1] - 0.05
    for f, g in zip(frames, plain):  # only the pathed group moves; order, ids and shape are kept
        assert f.strokes["stem"].to_dict() == g.strokes["stem"].to_dict()
        for i in ids:
            a, b = f.strokes[i], g.strokes[i]
            assert (a.id, a.order, a.vertices, a.z, a.a) == (b.id, b.order, b.vertices, b.z, b.a)


def test_motion_path_pinned_stroke_kept_and_anchor_steady(r):
    """swell is pinned on frame 5: it keeps its pin, and the anchor is swing's alone on every frame."""
    from inbetween.interpolate import eased_time
    from inbetween.motion_path import anchor, fit_path, point_at
    seq, _ = _with_path("bend_with_edit.json", ["swell", "swing"], _bow((0.55, 0.45), (0.62, 0.35), 0.2))
    edit = seq.edits[0]
    assert "swell" in edit.strokes
    warnings = []
    frames = interpolate(seq, "arc", r, warnings)
    assert any("pinned" in w for w in warnings)
    f5 = next(f for f in frames if f.frame == edit.frame)
    assert f5.strokes["swell"].to_dict() == edit.strokes["swell"].to_dict()
    aspect = r.settings.CANVAS_WIDTH_M / r.settings.CANVAS_HEIGHT_M
    ka, kb = seq.keys
    q = fit_path(seq.paths[0].points, anchor([ka.strokes["swing"]], r), anchor([kb.strokes["swing"]], r), aspect)
    te = eased_time(seq)
    for f in frames[1:-1]:
        want, got = point_at(q, te[f.frame], aspect), anchor([f.strokes["swing"]], r)
        assert math.hypot(want[0] - got[0], want[1] - got[1]) < 1e-5, f.frame
    refined, _ = refine(seq, frames, r, RefineConfig(iters=5, verbose=False))
    f5 = next(f for f in refined if f.frame == edit.frame)
    assert f5.strokes["swell"].to_dict() == edit.strokes["swell"].to_dict()
    for a, b in zip(frames, refined):  # refinement changes pressure only, so the path motion survives
        assert (a.strokes["swing"].xt, a.strokes["swing"].yt) == (b.strokes["swing"].xt, b.strokes["swing"].yt)


def test_motion_path_off_paper_keeps_strokes_on_paper(r):
    seq, _ = _with_path("bend.json", ["swing"], _bow((0.7, 0.3), (0.69, 0.12), 0.9))
    warnings = []
    frames = interpolate(seq, "arc", r, warnings)
    assert any("leaves the paper" in w for w in warnings)
    assert any("edge of the paper" in w for w in warnings)
    for f in frames:
        s = f.strokes["swing"]
        assert 0 <= s.xt <= 1 and 0 <= s.yt <= 1
    assert min(f.strokes["swing"].yt for f in frames) == 0.0  # stopped at the top edge


def test_motion_path_file_versions(r, tmp_path):
    import json
    from inbetween.strokes import FORMAT_VERSION, sequence_from_dict
    d = json.load(open(os.path.join(EX, "bend.json")))
    assert d["version"] == 1 and "paths" not in d
    seq = sequence_from_dict(d)  # a version 1 file still loads and interpolates as before
    assert seq.paths == []
    assert len(interpolate(seq, "arc", r)) == 11
    assert FORMAT_VERSION == 3
    with pytest.raises(ValueError, match="unsupported version"):
        sequence_from_dict({**d, "version": 4})
    bad = {**d, "version": 2, "paths": [{"span": "0-5", "ids": ["swell"], "points": [[0, 0], [1, 1]]}]}
    with pytest.raises(ValueError, match="not neighboring keyframes"):
        interpolate(sequence_from_dict(bad), "arc", r)


def test_motion_path_through_cli_and_exports(r, tmp_path):
    """The CLI reads the same file; its frames load the way paint_frames.py loads them and
    export as FRIDA plans with the moved positions."""
    import json
    import subprocess
    import sys
    from inbetween.export_plans import build_plans, frida_options
    from inbetween.strokes import load_frames
    _, d = _with_path("bend.json", ["swell", "swing"], _bow((0.55, 0.45), (0.62, 0.35), 0.2))
    kf = tmp_path / "keys.json"
    kf.write_text(json.dumps(d))
    src = os.path.join(HERE, "..", "..")
    res = subprocess.run([sys.executable, "-m", "inbetween.cli", str(kf), "--out", str(tmp_path / "out"),
                          "--method", "arc"], cwd=src, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    frames = load_frames(str(tmp_path / "out" / "frames_arc.json"))
    assert [f.frame for f in frames] == list(range(11))
    expected = interpolate(_with_path("bend.json", ["swell", "swing"], _bow((0.55, 0.45), (0.62, 0.35), 0.2))[0], "arc", r)
    assert abs(frames[5].strokes["swing"].yt - expected[5].strokes["swing"].yt) < 1e-5
    try:
        opt = frida_options("caches/vae_sharpie_final", os.path.join(src, "..", "materials_vae_sharpie.json"), True)
    except Exception as e:
        pytest.skip(f"FRIDA options not available: {e}")
    painting, _ = build_plans([frames[5]], opt)[5]
    planned = frames[5].ordered()
    xts = sorted(float(b.xt.detach()) for b in painting.brush_strokes)
    assert xts == pytest.approx(sorted(min(max(s.xt, 0), 1) for s in planned), abs=1e-6)


# --------------------------------------------------------------------------
# Trim paths (draw on / off)
# --------------------------------------------------------------------------

def _trim_doc(mode="draw_on", path=False):
    """A curved 'body' that moves right and rotates between keys 0 and 8, and a tapered
    'line' beside it that is only in key 8 (draw on) or only in key 0 (draw off)."""
    body = {"id": "body", "order": 0, "color": [0, 0, 0], "vertices": [[0, 0], [0.012, 0.008], [0.026, 0.008], [0.036, 0]],
            "kappa": [0.6, 0.6], "z": [0.6, 0.7, 0.7, 0.6], "xt": 0.3, "yt": 0.5, "a": 0.0}
    line = {"id": "line", "order": 1, "color": [0, 0, 0], "vertices": [[0, 0], [0.01, -0.004], [0.02, -0.004], [0.03, 0.002]],
            "kappa": [0.5, 0.5], "z": [0.55, 0.8, 0.8, 0.3], "xt": 0.31, "yt": 0.42, "a": 0.0}
    moved = lambda s, dx, da: {**s, "xt": s["xt"] + dx, "a": s["a"] + da}
    k0 = [body] + ([line] if mode == "draw_off" else [])
    k8 = [moved(body, 0.25, 0.6)] + ([moved(line, 0.25, 0.6)] if mode == "draw_on" else [])
    if mode == "draw_on":  # put the line beside the moved body: rotate its offset with the body
        from inbetween.strokes import Stroke as S_
        r = Renderer(CACHE)
        b0, b8 = Stroke.from_dict(body), Stroke.from_dict(k8[0])
        l0 = Stroke.from_dict(line)
        from inbetween.trim import follow
        piece = l0.copy()
        follow(piece, l0, b0, b8, r)  # the line placed as if it had moved with the body
        k8[1] = piece.to_dict()
    d = {"format": "frida-inbetween", "version": 3, "keys": [{"frame": 0, "strokes": k0}, {"frame": 8, "strokes": k8}],
         "timing": {"0-8": "ease_in_out"}, "trims": [{"span": "0-8", "ids": ["line"], "mode": mode}]}
    if path:
        d["paths"] = [{"span": "0-8", "ids": ["body", "line"], "points": [[0.32, 0.5], [0.45, 0.3], [0.57, 0.5]]}]
    return d


def _procrustes(P, Q):
    """Rigid transform (R, t) with R @ p + t ~ q for matched point sets (N, 2)."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = torch.linalg.svd((P - pc).T @ (Q - qc))
    R = (U @ Vt).T
    if torch.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return R, qc - R @ pc


def _xy(s, r):
    m = r.to_metric(r.path_canvas(stroke_tensors(s)))
    return torch.stack([m[:, 1], m[:, 0]], 1)


@pytest.mark.parametrize("mode", ["draw_on", "draw_off"])
def test_trim_stays_attached_to_moving_neighbour_and_cut_pressure(r, mode):
    from inbetween.interpolate import eased_time
    from inbetween.strokes import sequence_from_dict
    from inbetween.trim import complete_profile, pressure_at
    seq = sequence_from_dict(_trim_doc(mode))
    warnings = []
    frames = interpolate(seq, "arc", r, warnings)
    assert warnings == []
    for k in seq.keys:  # keys unchanged
        out = next(f for f in frames if f.frame == k.frame)
        assert {i: s.to_dict() for i, s in out.strokes.items()} == {i: s.to_dict() for i, s in k.strokes.items()}
    ref = 8 if mode == "draw_on" else 0
    complete, body_ref = frames[ref].strokes["line"], frames[ref].strokes["body"]
    L = float(complete_profile(complete, r)[2][-1])
    te = eased_time(seq)
    lengths = []
    with torch.no_grad():
        for f in frames[1:-1]:
            u = te[f.frame] if mode == "draw_on" else 1 - te[f.frame]
            # on every in-between (no pop at the middle), except where it is still shorter than
            # FRIDA's minimum stroke length, so it is never painted
            assert ("line" in f.strokes) == (u * L >= r.settings.MIN_STROKE_LENGTH), f.frame
            if "line" not in f.strokes:
                continue
            piece = f.strokes["line"]
            # a valid FRIDA stroke: BrushStroke builds the same path from it
            p = stroke_tensors(piece)
            opt = SimpleNamespace(MAX_STROKE_LENGTH=0.05, MIN_STROKE_Z=0.1, max_length_before_new_paint=1, ink=True)
            bs = BrushStroke(opt, vertices=p["vertices"], radii=p["radii"], z=p["z"], xt=p["xt"], yt=p["yt"], a=p["a"],
                             ink=True, device="cpu")
            assert torch.allclose(bs.get_path(), r.path(p), atol=1e-6)
            assert all(abs(v) <= 0.15 for q in piece.vertices for v in q) and all(0.02 <= z <= 0.98 for z in piece.z)
            # length u of the complete stroke
            n = float(complete_profile(piece, r)[2][-1])
            assert abs(n - u * L) < 0.5e-3, (f.frame, n, u * L)
            lengths.append(n)
            # pressure at the cut is the complete stroke's pressure there, not its tapered end
            assert abs(piece.z[3] - pressure_at(complete, r, u)) < 1e-4
            assert abs(piece.z[0] - complete.z[0]) < 1e-4
            # attached: the body's rigid motion since the complete frame, applied to the complete
            # line, contains the piece
            Rm, t = _procrustes(_xy(body_ref, r), _xy(f.strokes["body"], r))
            moved = _xy(complete, r) @ Rm.T + t
            moved = torch.nn.functional.interpolate(moved.T[None], size=1000, mode="linear", align_corners=True)[0].T
            pts = _xy(piece, r)
            assert (pts[0] - moved[0]).norm() < 0.2e-3, f.frame
            assert torch.cdist(pts, moved).min(dim=1).values.max() < 0.3e-3, f.frame
    assert len(lengths) >= 5 and lengths == sorted(lengths, reverse=(mode == "draw_off"))
    first = next(f for f in (frames if mode == "draw_on" else frames[::-1]) if "line" in f.strokes)
    assert abs(first.strokes["line"].z[3] - complete.z[3]) > 0.1  # the cut is not the taper


def test_trim_follows_motion_path_group(r):
    from inbetween.motion_path import anchor
    from inbetween.strokes import sequence_from_dict
    seq = sequence_from_dict(_trim_doc("draw_off", path=True))
    warnings = []
    frames = interpolate(seq, "arc", r, warnings)
    assert not any("not follow" in w for w in warnings)
    complete = frames[0].strokes["line"]
    a0 = anchor([frames[0].strokes["body"]], r)
    for f in frames[1:-1]:
        if "line" not in f.strokes:
            continue  # shorter than FRIDA's minimum stroke length near the end
        af, piece = anchor([f.strokes["body"]], r), f.strokes["line"]
        assert abs((piece.xt - complete.xt) - (af[0] - a0[0])) < 1e-6
        assert abs((piece.yt - complete.yt) - (af[1] - a0[1])) < 1e-6
    assert anchor([frames[4].strokes["body"]], r)[1] < 0.45  # the group went up the path


def test_trim_pin_refine_versions_and_export(r, tmp_path):
    import json
    from inbetween.export_plans import build_plans, frida_options
    from inbetween.strokes import sequence_from_dict
    from inbetween.trim import complete_profile
    d = _trim_doc("draw_on")
    frames = interpolate(sequence_from_dict(d), "arc", r)
    pin = frames[5].strokes["line"].to_dict()  # pin a partial state, moved a little
    pin["yt"] += 0.02
    d["edits"] = [{"frame": 5, "strokes": [pin]}]
    seq = sequence_from_dict(d)
    frames = interpolate(seq, "arc", r)
    assert frames[5].strokes["line"].to_dict() == Stroke.from_dict(pin).to_dict()  # the pin stays
    Lpin = float(complete_profile(frames[5].strokes["line"], r)[2][-1])
    before = [float(complete_profile(f.strokes["line"], r)[2][-1]) for f in frames[1:5] if "line" in f.strokes]
    assert len(before) >= 2
    assert before == sorted(before) and before[-1] < Lpin  # draws on toward the pinned state
    refined, _ = refine(seq, frames, r, RefineConfig(iters=5, verbose=False))
    for a, b in zip(frames[1:5], refined[1:5]):  # partial pieces keep their pressure and shape
        if "line" in a.strokes:
            assert a.strokes["line"].to_dict() == b.strokes["line"].to_dict()
    assert refined[5].strokes["line"].to_dict() == frames[5].strokes["line"].to_dict()
    # versions: 3 loads, 2 still loads, a bad mode is refused
    assert sequence_from_dict({**d, "version": 2, "trims": []}).trims == []
    with pytest.raises(ValueError, match="mode"):
        sequence_from_dict({**d, "trims": [{"span": "0-8", "ids": ["line"], "mode": "wipe"}]})
    # a trim on a stroke that is in both keys is ignored with a warning
    w = []
    interpolate(sequence_from_dict({**d, "trims": [{"span": "0-8", "ids": ["body"], "mode": "draw_on"}]}), "arc", r, w)
    assert any("ignored" in x for x in w)
    try:
        opt = frida_options("caches/vae_sharpie_final", os.path.join(HERE, "..", "..", "..", "materials_vae_sharpie.json"), True)
    except Exception as e:
        pytest.skip(f"FRIDA options not available: {e}")
    painting, _ = build_plans([frames[3]], opt)[3]
    assert "line" in frames[3].strokes
    assert len(painting.brush_strokes) == 2
