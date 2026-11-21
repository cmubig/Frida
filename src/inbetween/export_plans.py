"""Export in-betweened frames as FRIDA plans: one plan file per frame.

A FRIDA plan is what paint.py saves with --painting_path: a pickle of
(Painting, color_palette). This writes one such file per frame, so each frame
can be opened, rendered or painted with FRIDA's own tools.

Run from src/ with the same FRIDA options you paint with, plus:

    python -m inbetween.export_plans --frames ../out/bend/frames_refined.json \
        --out ../out/bend/plans --cache_dir caches/vae_sharpie_final \
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

import os
import pickle
import sys

from .paint_frames import _pop_arg


def main():
    frames_path = _pop_arg("--frames")
    out_dir = _pop_arg("--out")
    only = _pop_arg("--only")
    if frames_path is None or out_dir is None:
        sys.exit("need --frames path/to/frames_<method>.json --out path/to/plans")

    import torch
    from torchvision.utils import save_image

    import painting as painting_module
    from brush_stroke import BrushStroke
    from options import Options
    from painting import Painting

    from .strokes import load_frames, stroke_tensors

    if not torch.cuda.is_available():
        # FRIDA's Painting always puts its stroke model on the GPU; keep it on the CPU here.
        _get = painting_module.get_param2img
        painting_module.get_param2img = lambda o, device="cpu": _get(o, device="cpu")

    opt = Options()
    opt.gather_options()

    frames = load_frames(frames_path)
    if only:
        wanted = {int(x) for x in only.split(",")}
        frames = [f for f in frames if f.frame in wanted]
    os.makedirs(out_dir, exist_ok=True)

    colors = sorted({tuple(round(c, 4) for c in s.color) for f in frames for s in f.strokes.values()})
    palette = None if opt.ink else torch.tensor(colors, dtype=torch.float32)
    h = 512
    w = int(round(h * opt.CANVAS_WIDTH_M / opt.CANVAS_HEIGHT_M))

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
        painting = Painting(opt, brush_strokes=strokes)
        fn = os.path.join(out_dir, f"frame_{fr.frame:03d}.pkl")
        with open(fn, "wb") as f:
            pickle.dump((painting, palette), f)
        with torch.no_grad():
            img = painting(h, w, use_alpha=False)
        save_image(img[:, :3], os.path.join(out_dir, f"frame_{fr.frame:03d}.png"))
        print(f"{fn}: {len(strokes)} strokes")

    print(f"Exported {len(frames)} plans to {out_dir}" +
          ("" if palette is None else f" (palette: {len(colors)} colors)"))


if __name__ == "__main__":
    main()
