"""Convert an animator's finished raster frames (PNG) into robot strokes.

Each distinct drawing is traced like a person tracing it by hand: the dark
lines are thinned to their centerlines, the centerlines are followed into
ordered polylines (broken at crossings, straight continuations rejoined), and
each polyline is fitted to robot strokes with the same fitter the editor uses
for a drawn mark (fit.fit_mark, which splits marks one stroke can't follow).
Lines present in every drawing (a static background) are converted once.

Run from src/:

    python -m inbetween.import_frames path/to/Jump/*.png --out ../out/jump \\
        --cache_dir caches/vae_sharpie_final

Writes to --out:
    drawings.json          keyframe file: one key per distinct drawing (opens in the editor)
    background.json        keyframe file with the static background (one key)
    frames_drawings.json   frames file for paint_frames.py (one frame per distinct drawing)
    frames_background.json frames file for the background
    compare/               original vs. predicted painting, per drawing
    contact.png            all drawings: original (top) vs. predicted (bottom)
    report.json            strokes, fit error and holds per drawing

Strokes are not matched between drawings (each drawing gets its own ids).
That's enough to paint the drawings, but in-betweening between two of them
needs their strokes linked first (for example by tracing in the editor).
Filled areas (gray tones) are measured and reported, not converted.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .fit import fit_mark
from .render import Renderer
from .strokes import Frame, Sequence, Stroke, save_frames

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ------------------------------------------------------------- raster -------
def load_gray(path: str) -> np.ndarray:
    """0 = black ink, 1 = white paper (transparent counts as paper)."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return np.asarray(bg.convert("L"), dtype=np.float32) / 255.0


def distinct_drawings(paths: List[str], tol: float = 0.002) -> List[Tuple[int, int]]:
    """[(index of first frame, hold length)] for each run of identical frames."""
    runs, prev = [], None
    for i, p in enumerate(paths):
        small = np.asarray(Image.open(p).convert("L").resize((480, 270)), dtype=np.float32) / 255
        if prev is not None and np.abs(small - prev).mean() < tol:
            runs[-1][1] += 1
        else:
            runs.append([i, 1])
        prev = small
    return [tuple(r) for r in runs]


# ------------------------------------------------------- centerlines --------
def _trace(skel: np.ndarray) -> List[np.ndarray]:
    """Follow a 1-pixel skeleton into polylines of (row, col), split at nodes."""
    pix = set(zip(*np.nonzero(skel)))
    nb = {p: [(p[0] + dy, p[1] + dx) for dy, dx in N8 if (p[0] + dy, p[1] + dx) in pix] for p in pix}
    nodes = {p for p, n in nb.items() if len(n) != 2}
    seen = set()
    paths = []

    def walk(a, b):
        path = [a, b]
        seen.add((a, b)); seen.add((b, a))
        while path[-1] not in nodes:
            cur, prv = path[-1], path[-2]
            nxt = [q for q in nb[cur] if q != prv and (cur, q) not in seen]
            if not nxt:
                break
            seen.add((cur, nxt[0])); seen.add((nxt[0], cur))
            path.append(nxt[0])
            if nxt[0] == path[0]:
                break
        return path

    for a in nodes:
        for b in nb[a]:
            if (a, b) not in seen:
                paths.append(walk(a, b))
    for p in pix:  # closed loops with no node
        if p not in nodes:
            for q in nb[p]:
                if (p, q) not in seen:
                    paths.append(walk(p, q))
    return [np.array(p, dtype=np.float32) for p in paths if len(p) >= 2]


def _length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _dir(p: np.ndarray, at_start: bool, span: float = 8.0) -> np.ndarray:
    """Unit direction pointing out of the path at one end, over about `span` px."""
    q = p if at_start else p[::-1]
    d = np.cumsum(np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1)])
    j = int(np.searchsorted(d, span)) if d[-1] > span else len(q) - 1
    v = q[0] - q[max(j, 1)]
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _clean(paths: List[np.ndarray], spur_px: float, join_deg: float) -> List[np.ndarray]:
    """Drop short spurs and rejoin paths that continue straight through a crossing."""
    key = lambda pt: (int(round(pt[0])), int(round(pt[1])))
    def ends(ps):
        cnt = {}
        for p in ps:
            for e in (key(p[0]), key(p[-1])):
                cnt[e] = cnt.get(e, 0) + 1
        return cnt
    # spurs: short paths with a free end
    for _ in range(2):
        cnt = ends(paths)
        paths = [p for p in paths if not (_length(p) < spur_px and (cnt[key(p[0])] == 1) != (cnt[key(p[-1])] == 1))]
    # rejoin: at each shared end, pair the two paths that continue most straight
    cos_lim = math.cos(math.radians(join_deg))
    changed = True
    while changed:
        changed = False
        at: Dict[tuple, List[Tuple[int, bool]]] = {}
        for i, p in enumerate(paths):
            if key(p[0]) == key(p[-1]):
                continue  # closed loop
            at.setdefault(key(p[0]), []).append((i, True))
            at.setdefault(key(p[-1]), []).append((i, False))
        best = None
        for node, inc in at.items():
            if len(inc) < 2:
                continue
            for a in range(len(inc)):
                for b in range(a + 1, len(inc)):
                    (i, si), (j, sj) = inc[a], inc[b]
                    if i == j:
                        continue
                    c = float(-_dir(paths[i], si) @ _dir(paths[j], sj))  # 1 = perfectly straight through
                    if c > cos_lim and (best is None or c > best[0]):
                        best = (c, i, si, j, sj)
        if best:
            _, i, si, j, sj = best
            pi = paths[i][::-1] if si else paths[i]          # ends at the node
            pj = paths[j] if sj else paths[j][::-1]          # starts at the node
            paths = [p for k, p in enumerate(paths) if k not in (i, j)] + [np.vstack([pi, pj[1:]])]
            changed = True
    return paths


def _simplify(p: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker."""
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    d = (np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / n) if n > 0 else np.linalg.norm(p - a, axis=1)
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.vstack([_simplify(p[: i + 1], eps)[:-1], _simplify(p[i:], eps)])
    return np.vstack([a, b])


def centerlines(ink: np.ndarray, min_len_px: float = 6, spur_px: float = 8, join_deg: float = 40,
                eps_px: float = 0.8) -> List[np.ndarray]:
    from skimage.morphology import skeletonize
    skel = skeletonize(ink)
    paths = _clean(_trace(skel), spur_px, join_deg)
    out = []
    for p in paths:
        if _length(p) < min_len_px:
            continue
        out.append(_simplify(p, eps_px))
    out.sort(key=lambda p: (p[:, 0].min(), p[:, 1].min()))  # top to bottom: a stable painting order
    return out


def dots(ink: np.ndarray, max_px: int = 60) -> List[np.ndarray]:
    """Small blobs (eyes, specks) that thin to almost nothing: a tiny polyline across each."""
    from skimage.measure import label, regionprops
    out = []
    for rg in regionprops(label(ink)):
        if rg.area <= max_px:
            r0, c0, r1, c1 = rg.bbox
            cy, cx = rg.centroid
            if c1 - c0 >= r1 - r0:
                out.append(np.array([[cy, c0], [cy, c1 - 1]], dtype=np.float32))
            else:
                out.append(np.array([[r0, cx], [r1 - 1, cx]], dtype=np.float32))
    return out


# ------------------------------------------------------------- fitting ------
class Placement:
    """Image pixels -> canvas proportions, the image fitted inside the canvas."""
    def __init__(self, w_px: int, h_px: int, canvas_w: float, canvas_h: float, margin: float = 0.0):
        s = min(canvas_w * (1 - 2 * margin) / w_px, canvas_h * (1 - 2 * margin) / h_px)  # meters per px
        self.m_per_px = s
        self.ox = (canvas_w - w_px * s) / 2
        self.oy = (canvas_h - h_px * s) / 2
        self.cw, self.ch = canvas_w, canvas_h

    def __call__(self, rc: np.ndarray) -> List[List[float]]:
        x = (self.ox + rc[:, 1] * self.m_per_px) / self.cw
        y = (self.oy + rc[:, 0] * self.m_per_px) / self.ch
        return np.stack([x, y], 1).tolist()


def _densify(p: np.ndarray, step: float = 3.0) -> np.ndarray:
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.linalg.norm(b - a) // step))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def fit_polylines(polys: List[np.ndarray], place: Placement, r: Renderer, pressure: float,
                  max_err_mm: float, prefix: str, color) -> Tuple[List[Stroke], List[float]]:
    strokes, errs = [], []
    for p in polys:
        pts = place(_densify(p))
        for s, e in fit_mark(pts, [pressure] * len(pts), r, max_err_mm=max_err_mm, color=color):
            s.id = f"{prefix}{len(strokes) + 1}"
            s.order = len(strokes)
            strokes.append(s)
            errs.append(e)
    return strokes, errs


# ------------------------------------------------------------- output -------
def _render(r: Renderer, strokes: List[Stroke]) -> np.ndarray:
    return (r.render_frame(strokes).cpu().numpy() * 255).astype(np.uint8)


def _original_on_canvas(gray: np.ndarray, place: Placement, H: int, W: int) -> np.ndarray:
    canvas = Image.new("L", (W, H), 255)
    h, w = gray.shape
    sw, sh = round(w * place.m_per_px / place.cw * W), round(h * place.m_per_px / place.ch * H)
    im = Image.fromarray((gray * 255).astype(np.uint8)).resize((sw, sh), Image.LANCZOS)
    canvas.paste(im, (round(place.ox / place.cw * W), round(place.oy / place.ch * H)))
    return np.asarray(canvas.convert("RGB"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="+", help="PNG frames in order (a glob is fine)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default="caches/vae_sharpie_final")
    ap.add_argument("--ink_threshold", type=float, default=0.5, help="gray below this is a line (0 black, 1 white)")
    ap.add_argument("--fill_threshold", type=float, default=0.85, help="gray between the two thresholds is a filled area")
    ap.add_argument("--pressure", type=float, default=0.0, help="stylus-style pressure 0..1 (0 = thinnest visible mark)")
    ap.add_argument("--max_err_mm", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.0, help="empty border as a fraction of the canvas")
    ap.add_argument("--no_background", action="store_true", help="don't separate a static background")
    ap.add_argument("--only", default=None, help="comma-separated drawing numbers (1-based) to convert")
    args = ap.parse_args(argv)

    paths = sorted(p for pat in args.frames for p in (glob.glob(pat) or [pat]))
    os.makedirs(os.path.join(args.out, "compare"), exist_ok=True)
    r = Renderer(args.cache_dir, render_height=512)
    cw, ch = r.settings.CANVAS_WIDTH_M, r.settings.CANVAS_HEIGHT_M
    color = [0.05, 0.05, 0.05]

    runs = distinct_drawings(paths)
    grays = [load_gray(paths[i]) for i, _ in runs]
    inks = [g < args.ink_threshold for g in grays]
    h, w = grays[0].shape
    place = Placement(w, h, cw, ch, args.margin)
    print(f"{len(paths)} frames, {len(runs)} distinct drawings; 1 px = {place.m_per_px * 1000:.3f} mm on the canvas")

    static = np.logical_and.reduce(inks) if (len(inks) > 1 and not args.no_background) else np.zeros_like(inks[0])
    if static.any():
        from scipy.ndimage import binary_dilation
        static_zone = binary_dilation(static, iterations=2)
    else:
        static_zone = static

    report = {"source_frames": len(paths), "mm_per_px": place.m_per_px * 1000, "drawings": []}
    bg_strokes = []
    if static.any():
        polys = centerlines(static)
        bg_strokes, errs = fit_polylines(polys, place, r, args.pressure, args.max_err_mm, "bg", color)
        report["background"] = {"strokes": len(bg_strokes), "err_mm_median": float(np.median(errs)),
                                "err_mm_max": float(max(errs))}
        print(f"background: {len(bg_strokes)} strokes, fit error median {np.median(errs):.2f} mm, max {max(errs):.2f} mm")

    only = {int(x) for x in args.only.split(",")} if args.only else None
    keys, frames = [], []
    H = 512
    W = int(round(H * cw / ch))
    tiles = []
    for n, ((first, hold), g, ink) in enumerate(zip(runs, grays, inks), start=1):
        if only and n not in only:
            continue
        own = ink & ~static_zone
        polys = centerlines(own) + dots(own)
        strokes, errs = fit_polylines(polys, place, r, args.pressure, args.max_err_mm, f"d{n:02d}_s", color)
        fill = (g >= args.ink_threshold) & (g < args.fill_threshold) & ~static_zone
        fill_cm2 = float(fill.sum()) * (place.m_per_px * 100) ** 2
        info = {"drawing": n, "frame": first, "hold": hold, "strokes": len(strokes),
                "err_mm_median": float(np.median(errs)) if errs else 0.0,
                "err_mm_max": float(max(errs)) if errs else 0.0, "fill_cm2_not_converted": round(fill_cm2, 2)}
        report["drawings"].append(info)
        print(f"drawing {n:2d} (frame {first:3d}, held {hold}): {len(strokes):3d} strokes, "
              f"fit error median {info['err_mm_median']:.2f} mm, max {info['err_mm_max']:.2f} mm"
              + (f", fill {fill_cm2:.1f} cm2 not converted" if fill_cm2 > 0.05 else ""))
        keys.append(Frame(frame=first, strokes={s.id: s for s in strokes}, kind="key"))
        frames.append(Frame(frame=first, strokes={s.id: s for s in strokes}, kind="key", locked=list({s.id for s in strokes})))
        orig = _original_on_canvas(g, place, H, W)
        pred = _render(r, bg_strokes + strokes)
        pair = np.concatenate([orig, np.full((H, 8, 3), 255, np.uint8), pred], 1)
        Image.fromarray(pair).save(os.path.join(args.out, "compare", f"drawing_{n:02d}.png"))
        tiles.append((orig, pred))

    def keyfile(ks):
        return {"format": "frida-inbetween", "version": 1, "cache_dir": args.cache_dir,
                "keys": [{"frame": k.frame, "strokes": [s.to_dict() for s in k.ordered()]} for k in ks],
                "edits": [], "timing": {}}
    with open(os.path.join(args.out, "drawings.json"), "w") as f:
        json.dump(keyfile(keys), f, indent=1)
    save_frames(os.path.join(args.out, "frames_drawings.json"), frames, {"source": "import_frames"})
    if bg_strokes:
        bgk = Frame(frame=0, strokes={s.id: s for s in bg_strokes}, kind="key")
        with open(os.path.join(args.out, "background.json"), "w") as f:
            json.dump(keyfile([bgk]), f, indent=1)
        save_frames(os.path.join(args.out, "frames_background.json"),
                    [Frame(frame=0, strokes=bgk.strokes, kind="key", locked=list(bgk.strokes))], {"source": "import_frames"})

    if tiles:  # contact sheet: originals on top, predictions below
        th = 180
        tw = int(round(th * W / H))
        sheet = Image.new("RGB", (tw * len(tiles) + 4 * (len(tiles) - 1), th * 2 + 4), (200, 200, 200))
        for i, (o, p) in enumerate(tiles):
            sheet.paste(Image.fromarray(o).resize((tw, th), Image.LANCZOS), (i * (tw + 4), 0))
            sheet.paste(Image.fromarray(p).resize((tw, th), Image.LANCZOS), (i * (tw + 4), th + 4))
        sheet.save(os.path.join(args.out, "contact.png"))
    d = report["drawings"]
    report["total"] = {"drawing_strokes": sum(x["strokes"] for x in d), "background_strokes": len(bg_strokes),
                       "err_mm_max": max([x["err_mm_max"] for x in d] + [report.get("background", {}).get("err_mm_max", 0)])}
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"total: {report['total']['drawing_strokes']} drawing strokes + {len(bg_strokes)} background strokes -> {args.out}")


if __name__ == "__main__":
    main()
