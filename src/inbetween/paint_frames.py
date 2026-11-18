"""Have the robot paint a generated sequence, one frame per sheet.

Run from src/ with your usual FRIDA robot options, plus:

    python -m inbetween.paint_frames --frames ../out/bend/frames_refined.json \
        --photo_dir ../out/bend/photos --cache_dir caches/vae_sharpie_final \
        --materials_json ../materials_vae_sharpie.json --robot xarm --ink

    --frames      frames_<method>.json written by inbetween.cli (required)
    --photo_dir   where camera photos of each finished frame go (optional)
    --only        comma-separated frame numbers to (re)paint, e.g. 3,4 (optional)

Every other flag goes to FRIDA's Options unchanged (robot, materials, cache,
--simulate, --ink ...). Unlike paint.py, nothing is re-planned while
painting: each frame is executed exactly as planned, so frames stay
consistent with the animation. Strokes are painted in each frame's stroke
order, which is the same in every frame.

Tip: tape an animation peg bar (or corner stops) to the table so every sheet
sits in exactly the same place, and keep camera, lighting and paper position
fixed while photographing.
"""

from __future__ import annotations

import os
import sys


def _pop_arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return val
    return default


def main():
    frames_path = _pop_arg("--frames")
    photo_dir = _pop_arg("--photo_dir")
    only = _pop_arg("--only")
    if frames_path is None:
        sys.exit("need --frames path/to/frames_<method>.json")

    import cv2
    import torch

    from brush_stroke import BrushStroke
    from options import Options
    from paint_utils3 import canvas_to_global_coordinates, nearest_color
    from painter import Painter

    from .strokes import load_frames, stroke_tensors

    import datetime

    from my_tensorboard import TensorBoard

    opt = Options()
    opt.gather_options()
    run_name = "inbetween_" + datetime.datetime.now().strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard(f"{opt.tensorboard_dir}/{run_name}",
                             image_dir=f"{opt.image_log_dir}/{run_name}")  # same as paint.py
    painter = Painter(opt)
    opt = painter.opt

    frames = load_frames(frames_path)
    if only:
        wanted = {int(x) for x in only.split(",")}
        frames = [f for f in frames if f.frame in wanted]
    if photo_dir:
        os.makedirs(photo_dir, exist_ok=True)

    # One palette for the whole sequence (paint setups only).
    colors = sorted({tuple(round(c, 4) for c in s.color) for f in frames for s in f.strokes.values()})
    palette = torch.tensor(colors)
    print(f"{len(frames)} frames to paint. Palette ({len(colors)} colors): {colors}")

    for fr in frames:
        painter.to_neutral()
        if not opt.simulate:
            input(f"\nPlace a fresh sheet for frame {fr.frame} and press Enter...")
        curr_color, consecutive_paints, since_clean = -1, 0, 0
        for i, s in enumerate(fr.ordered()):
            p = stroke_tensors(s)
            stroke = BrushStroke(opt, vertices=p["vertices"], radii=p["radii"], z=p["z"],
                                 xt=p["xt"], yt=p["yt"], a=p["a"],
                                 color=torch.tensor(s.color), ink=opt.ink, device="cpu")
            BrushStroke.make_valid(stroke)
            stroke.clear_path_cache()

            if not opt.ink:  # same paint handling as paint.py
                color_ind, _ = nearest_color(stroke.color_transform.detach().numpy(), palette.numpy())
                new_color = color_ind != curr_color
                if new_color or since_clean > 12:
                    painter.clean_paint_brush()
                    if new_color:
                        painter.clean_paint_brush()
                    since_clean, curr_color, new_color = 0, color_ind, True
                if consecutive_paints >= opt.how_often_to_get_paint or new_color:
                    painter.get_paint(color_ind)
                    consecutive_paints = 0

            # Canvas proportions -> robot coordinates, exactly as in paint.py
            x, y = s.xt, 1 - s.yt
            x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
            x_glob, y_glob, _ = canvas_to_global_coordinates(x, y, None, opt)
            painter.execute_stroke(stroke, x_glob, y_glob, s.a, label=f"f{fr.frame}-{s.id}")
            consecutive_paints += 1
            since_clean += 1

        painter.to_neutral()
        if photo_dir and not opt.simulate:
            img = painter.camera.get_canvas()
            fn = os.path.join(photo_dir, f"frame_{fr.frame:03d}.png")
            cv2.imwrite(fn, img[:, :, ::-1] if img.ndim == 3 else img)
            print(f"saved {fn}")

    if not opt.ink:
        for _ in range(3):
            painter.clean_paint_brush()
    print(f"Done. Strokes skipped after robot failure: {getattr(painter, 'failed_strokes', 0)}")


if __name__ == "__main__":
    main()
