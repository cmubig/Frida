"""Generate in-betweens from a keyframe file.

Run from src/:

    python -m inbetween.cli inbetween/examples/bend.json --out ../out/bend --compare

Outputs (in --out):
    frames_<method>.json   every frame's stroke plan (input for paint_frames.py)
    <method>.gif           animation preview, plus <method>/frame_XXX.png
    compare.gif            methods side by side (with --compare)
    width_plot.png         painted width per stroke over frames (with --compare)
    metrics.json           popping measurements per method
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .interpolate import interpolate
from .metrics import popping
from .refine import RefineConfig, refine
from .render import RESPONSES, Renderer, save_gif, save_png
from .strokes import load_sequence, save_frames


def _label(img: torch.Tensor, text: str) -> torch.Tensor:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    im = Image.fromarray((img.cpu().numpy() * 255).astype("uint8"))
    ImageDraw.Draw(im).text((6, 4), text, fill=(200, 30, 30))
    import numpy as np
    return torch.from_numpy(np.asarray(im).astype("float32") / 255.0)


def _width_plot(metrics: dict, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods = list(metrics)
    # Only strokes whose width really changes (skip measurement noise < 0.1 mm).
    span = lambda w: max(w) - min(w)
    sids = [s for s in metrics[methods[0]]["strokes"]
            if any(span(metrics[m]["strokes"][s]["width_mm"]) > 0.1 for m in methods)]
    if not sids:
        return
    fig, axes = plt.subplots(1, len(sids), figsize=(4 * len(sids), 3.2), squeeze=False)
    for ax, sid in zip(axes[0], sids):
        for m in methods:
            ax.plot(metrics[m]["strokes"][sid]["width_mm"], marker="o", ms=3, label=m)
        ax.set_title(f"stroke '{sid}'")
        ax.set_xlabel("frame")
        ax.set_ylabel("painted width (mm)")
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keyframes", help="keyframe JSON (see inbetween/README.md)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--cache_dir", default=None,
                    help="FRIDA cache dir with param2img.pt (default: the file's cache_dir, "
                         "else caches/vae_sharpie_final)")
    ap.add_argument("--method", choices=["linear", "arc", "refined"], default="refined")
    ap.add_argument("--compare", action="store_true", help="run linear, arc and refined and compare")
    ap.add_argument("--response", choices=RESPONSES, default="model",
                    help="pressure response; placeholder_sigmoid is an ASSUMED curve, not real data")
    ap.add_argument("--render_height", type=int, default=256)
    ap.add_argument("--paint", action="store_true",
                    help="paint setup (not ink): first pressure value pinned for brush touchdown")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args(argv)

    seq = load_sequence(args.keyframes)
    cache_dir = args.cache_dir or seq.meta.get("cache_dir") or "caches/vae_sharpie_final"
    r = Renderer(cache_dir, render_height=args.render_height, response=args.response)
    os.makedirs(args.out, exist_ok=True)
    meta = {"source": os.path.abspath(args.keyframes), "cache_dir": cache_dir, "response": args.response}
    if args.response != "model":
        print("[inbetween] NOTE: using an assumed placeholder pressure response, not a calibrated one.")

    methods = ["linear", "arc", "refined"] if args.compare else [args.method]
    results, metrics = {}, {}
    arc = None
    for m in methods:
        t0 = time.time()
        if m == "linear":
            frames = interpolate(seq, "linear", r)
        else:
            arc = arc or interpolate(seq, "arc", r)
            frames = arc
            if m == "refined":
                print("[inbetween] refining in-betweens through the stroke model...")
                frames, _ = refine(seq, arc, r, RefineConfig(), ink=not args.paint)
        results[m] = frames
        metrics[m] = popping(seq, frames, r)
        save_frames(os.path.join(args.out, f"frames_{m}.json"), frames, {**meta, "method": m})
        imgs = [r.render_frame(fr.strokes.values()) for fr in frames]
        os.makedirs(os.path.join(args.out, m), exist_ok=True)
        for fr, img in zip(frames, imgs):
            save_png(img, os.path.join(args.out, m, f"frame_{fr.frame:03d}.png"))
        save_gif(imgs, os.path.join(args.out, f"{m}.gif"), fps=args.fps)
        s = metrics[m]["summary_mean_over_strokes"]
        print(f"[inbetween] {m:8s} {len(frames)} frames in {time.time() - t0:.1f}s | "
              f"width deviation max {s['width_dev_max_mm']:.3f} mm, mean {s['width_dev_mean_mm']:.3f} mm")

    if len(methods) > 1:
        rows = []
        for fi in range(len(results[methods[0]])):
            tiles = [_label(r.render_frame(results[m][fi].strokes.values()), f"{m}  f{results[m][fi].frame}")
                     for m in methods]
            rows.append(torch.cat(tiles, dim=1))
        save_gif(rows, os.path.join(args.out, "compare.gif"), fps=args.fps)
        _width_plot(metrics, os.path.join(args.out, "width_plot.png"))

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"meta": meta, **metrics}, f, indent=1)
    print(f"[inbetween] wrote results to {args.out}")


if __name__ == "__main__":
    main()
