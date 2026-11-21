"""Export in-betweened frames as FRIDA plans: one plan file per frame.

A FRIDA plan is what paint.py saves with --painting_path: a pickle of
(Painting, color_palette). This writes one such file per frame, so each frame
can be opened, rendered or painted with FRIDA's own tools. The editor's
File > Export FRIDA Plans does the same and downloads a .zip.

Run from src/ with the same FRIDA options you paint with, plus:

    python -m inbetween.export_plans --frames ../out/bend/frames_refined.json \\
        --out ../out/bend/plans --cache_dir caches/vae_sharpie_final \\
        --materials_json ../materials_vae_sharpie.json --ink

    --frames   frames_<method>.json from inbetween.cli or the editor (required)
    --out      folder for frame_<NNN>.pkl and preview PNGs (required)
    --only     comma-separated frame numbers to export, e.g. 0,4 (optional)

Load one in Python:

    painting, palette = pickle.load(open("plans/frame_004.pkl", "rb"))

Strokes are stored in painting order (painting.pop() returns them in order),
with the robot limits applied as in paint_frames.py. The palette is the
sequence's palette (the same for every frame), or None with --ink.

paint.py --painting_path <file> loads a plan, but it re-optimizes the
remaining strokes toward its objective between batches, which changes the
frame. To paint frames exactly as planned, use inbetween.paint_frames.
"""

from __future__ import annotations

import io
import os
import pickle
import sys
import zipfile
from typing import Dict, List

from .paint_frames import _pop_arg

README_IN_ZIP = """FRIDA plans exported from the in-betweening editor.

frame_NNN.pkl  one FRIDA plan per frame: pickle of (Painting, color_palette),
               the format paint.py saves with --painting_path. Strokes are in
               painting order. Load from FRIDA's src/ folder:
                   painting, palette = pickle.load(open("frame_004.pkl", "rb"))
frame_NNN.png  preview of the frame rendered by FRIDA's stroke model.

To paint the animation exactly as planned, use the frames JSON with
    python -m inbetween.paint_frames --frames frames_<method>.json ...
paint.py re-optimizes a loaded plan between batches, which changes the frame.
"""


def _cpu_safe_painting():
    """FRIDA's Painting puts its stroke model on the GPU when there is one, but the strokes
    here are built on the CPU (so the pickles load on any machine). Keep the model on the
    CPU too, with or without a GPU, or rendering the preview mixes devices.
    Painting.forward also makes its blank canvas on the module-level `device`.
    (No FRIDA file is changed.)"""
    import torch
    import painting as painting_module
    if not getattr(painting_module, "_inbetween_cpu", False):
        _get = painting_module.get_param2img
        painting_module.get_param2img = lambda o, device="cpu": _get(o, device="cpu")
        painting_module.device = torch.device("cpu")
        painting_module._inbetween_cpu = True
    return painting_module.Painting


def frida_options(cache_dir: str, materials_json: str, ink: bool, extra: List[str] = ()):
    """FRIDA Options as if given on the command line (Options only reads sys.argv)."""
    from options import Options
    argv = ["frida", "--cache_dir", cache_dir, "--materials_json", materials_json, *extra]
    if ink:
        argv.append("--ink")
    saved = sys.argv
    try:
        sys.argv = argv
        opt = Options()
        opt.gather_options()
    finally:
        sys.argv = saved
    return opt


def build_plans(frames, opt) -> Dict[int, tuple]:
    """{frame number: (Painting, palette)} for each frame, strokes in painting order."""
    import torch
    from brush_stroke import BrushStroke
    from .strokes import stroke_tensors

    Painting = _cpu_safe_painting()
    colors = sorted({tuple(round(c, 4) for c in s.color) for f in frames for s in f.strokes.values()})
    palette = None if opt.ink else torch.tensor(colors, dtype=torch.float32)
    plans = {}
    for fr in frames:
        strokes = []
        for s in fr.ordered():
            p = stroke_tensors(s)
            b = BrushStroke(opt, vertices=p["vertices"], radii=p["radii"], z=p["z"],
                            xt=p["xt"], yt=p["yt"], a=p["a"],
                            color=torch.tensor(s.color), ink=opt.ink, device="cpu")
            BrushStroke.make_valid(b)
            b.clear_path_cache()
            strokes.append(b)
        plans[fr.frame] = (Painting(opt, brush_strokes=strokes), palette)
    return plans


def _preview_png(painting, opt, height=512) -> bytes:
    import torch
    from torchvision.utils import save_image
    w = int(round(height * opt.CANVAS_WIDTH_M / opt.CANVAS_HEIGHT_M))
    with torch.no_grad():
        img = painting(height, w, use_alpha=False)
    buf = io.BytesIO()
    save_image(img[:, :3], buf, format="png")
    return buf.getvalue()


def plan_files(frames, opt) -> Dict[str, bytes]:
    """File name -> bytes: frame_NNN.pkl and frame_NNN.png for every frame."""
    out = {}
    for n, (painting, palette) in build_plans(frames, opt).items():
        out[f"frame_{n:03d}.pkl"] = pickle.dumps((painting, palette))
        out[f"frame_{n:03d}.png"] = _preview_png(painting, opt)
    return out


def plans_zip(frames, opt, frames_json: str = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in plan_files(frames, opt).items():
            z.writestr(name, data)
        z.writestr("README.txt", README_IN_ZIP)
        if frames_json:
            z.writestr("frames.json", frames_json)
    return buf.getvalue()


def main():
    frames_path = _pop_arg("--frames")
    out_dir = _pop_arg("--out")
    only = _pop_arg("--only")
    if frames_path is None or out_dir is None:
        sys.exit("need --frames path/to/frames_<method>.json --out path/to/plans")

    from options import Options
    from .strokes import load_frames

    opt = Options()
    opt.gather_options()

    frames = load_frames(frames_path)
    if only:
        wanted = {int(x) for x in only.split(",")}
        frames = [f for f in frames if f.frame in wanted]
    os.makedirs(out_dir, exist_ok=True)
    for name, data in plan_files(frames, opt).items():
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(data)
        if name.endswith(".pkl"):
            print(os.path.join(out_dir, name))
    print(f"Exported {len(frames)} plans to {out_dir}")


if __name__ == "__main__":
    main()
