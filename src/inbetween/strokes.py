"""Stroke data: the JSON keyframe format and conversion to/from FRIDA tensors.

A stroke uses FRIDA's BrushStroke parameters directly, so every frame we
produce can be executed by the robot code unchanged:

    id        stable string ID; the same id in two keyframes = the same stroke
    order     painting order within a frame (lower paints first)
    color     [r, g, b] in 0..1 (ignored by ink/marker setups)
    vertices  4 x [x, y] corner points, meters, stroke-local frame
    kappa     2 corner roundness values in 0..1 (0 = sharp, 1 = max round)
    z         4 pressure values in 0..1 (0 = lightest touch, 1 = deepest),
              spread along the stroke. This is what the robot commands; the
              visible width is predicted by the stroke model, never stored.
    xt, yt    stroke placement on the canvas, as proportions 0..1
              (same convention FRIDA's renderer and paint.py use)
    a         stroke rotation, radians

Optional per-stroke fields for strokes that exist in only some keyframes are
not needed: a stroke missing from a neighboring keyframe simply pops in/out
at the midpoint of that segment (see interpolate.py).
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from filleted_stroke import radii_from_kappa, sample_filleted_polyline

N_VERTICES = 4
N_PATH_POINTS = 32
FORMAT_NAME = "frida-inbetween"
# Keyframe file version. 2 added "paths" (motion paths); version 1 files still load.
FORMAT_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
# The frames file (what paint_frames.py reads) did not change; it keeps its own version.
FRAMES_VERSION = 1


@dataclass
class Stroke:
    id: str
    vertices: List[List[float]]
    kappa: List[float]
    z: List[float]
    xt: float
    yt: float
    a: float
    order: int = 0
    color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @staticmethod
    def from_dict(d: dict) -> "Stroke":
        s = Stroke(
            id=str(d["id"]),
            vertices=[[float(x), float(y)] for x, y in d["vertices"]],
            kappa=[float(k) for k in d["kappa"]],
            z=[float(v) for v in d["z"]],
            xt=float(d["xt"]),
            yt=float(d["yt"]),
            a=float(d["a"]),
            order=int(d.get("order", 0)),
            color=[float(c) for c in d.get("color", [0.0, 0.0, 0.0])],
        )
        s.validate()
        return s

    def to_dict(self) -> dict:
        r = lambda v: round(float(v), 6)
        return {
            "id": self.id,
            "order": self.order,
            "color": [r(c) for c in self.color],
            "vertices": [[r(x), r(y)] for x, y in self.vertices],
            "kappa": [r(k) for k in self.kappa],
            "z": [r(v) for v in self.z],
            "xt": r(self.xt),
            "yt": r(self.yt),
            "a": r(self.a),
        }

    def validate(self) -> None:
        if len(self.vertices) != N_VERTICES:
            raise ValueError(f"stroke {self.id}: need {N_VERTICES} vertices")
        if len(self.kappa) != N_VERTICES - 2:
            raise ValueError(f"stroke {self.id}: need {N_VERTICES - 2} kappa values")
        if len(self.z) != 4:
            raise ValueError(f"stroke {self.id}: need 4 z values")

    def copy(self) -> "Stroke":
        return copy.deepcopy(self)


@dataclass
class Frame:
    frame: int
    strokes: Dict[str, Stroke]
    kind: str = "key"  # "key" | "inbetween"
    locked: List[str] = field(default_factory=list)  # stroke ids fixed by the artist

    def ordered(self) -> List[Stroke]:
        return sorted(self.strokes.values(), key=lambda s: (s.order, s.id))


@dataclass
class MotionPath:
    """A path drawn by the animator: the strokes in `ids` travel along it, as a group,
    from keyframe span[0] to keyframe span[1] (neighboring keys). `points` is the drawn
    polyline in canvas proportions [x, y]. It is never painted; see motion_path.py."""
    span: Tuple[int, int]
    ids: List[str]
    points: List[List[float]]


@dataclass
class Sequence:
    keys: List[Frame]
    edits: List[Frame] = field(default_factory=list)  # artist fixes on in-between frames
    timing: Dict[str, str] = field(default_factory=dict)  # "fromFrame-toFrame" -> easing
    meta: dict = field(default_factory=dict)
    paths: List[MotionPath] = field(default_factory=list)  # motion paths (version 2)


def load_sequence(path: str) -> Sequence:
    with open(path) as f:
        return sequence_from_dict(json.load(f), name=path)


def sequence_from_dict(d: dict, name: str = "document") -> Sequence:
    if d.get("format") != FORMAT_NAME:
        raise ValueError(f"{name}: not a {FORMAT_NAME} file")
    if int(d.get("version", 0)) not in SUPPORTED_VERSIONS:
        raise ValueError(f"{name}: unsupported version {d.get('version')}")

    def frames(items, kind):
        out = []
        for item in items:
            strokes = {}
            for sd in item["strokes"]:
                s = Stroke.from_dict(sd)
                if s.id in strokes:
                    raise ValueError(f"frame {item['frame']}: duplicate stroke id {s.id}")
                strokes[s.id] = s
            out.append(Frame(frame=int(item["frame"]), strokes=strokes, kind=kind,
                             locked=list(strokes.keys()) if kind == "edit" else []))
        return sorted(out, key=lambda fr: fr.frame)

    keys = frames(d["keys"], "key")
    if len(keys) < 2:
        raise ValueError("need at least two keyframes")
    if len({k.frame for k in keys}) != len(keys):
        raise ValueError("two keyframes share a frame number")
    edits = frames(d.get("edits", []), "edit")
    for e in edits:
        if any(e.frame == k.frame for k in keys):
            raise ValueError(f"edit on frame {e.frame} collides with a keyframe; edit the key instead")
        if not keys[0].frame < e.frame < keys[-1].frame:
            raise ValueError(f"edit on frame {e.frame} is outside the keyframe range")
    paths = [_motion_path(p, name) for p in d.get("paths", [])]
    meta = {k: v for k, v in d.items() if k not in ("keys", "edits", "timing", "format", "version", "paths")}
    return Sequence(keys=keys, edits=edits, timing=d.get("timing", {}), meta=meta, paths=paths)


def _motion_path(p: dict, name: str) -> MotionPath:
    try:
        a, b = (int(x) for x in str(p["span"]).split("-"))
        points = [[float(x), float(y)] for x, y in p["points"]]
        ids = [str(i) for i in p["ids"]]
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"{name}: bad motion path {p.get('span')!r}: {e}") from None
    if not a < b:
        raise ValueError(f"{name}: motion path {a}-{b} must go forward in time")
    if len(points) < 2:
        raise ValueError(f"{name}: motion path {a}-{b} needs at least 2 points")
    return MotionPath(span=(a, b), ids=ids, points=points)


def frames_to_dict(frames: List[Frame], meta: Optional[dict] = None) -> dict:
    return {
        "format": FORMAT_NAME + "-frames",
        "version": FRAMES_VERSION,
        **(meta or {}),
        "frames": [
            {
                "frame": fr.frame,
                "kind": fr.kind,
                "locked": fr.locked,
                "strokes": [s.to_dict() for s in fr.ordered()],
            }
            for fr in frames
        ],
    }


def save_frames(path: str, frames: List[Frame], meta: Optional[dict] = None) -> None:
    out = frames_to_dict(frames, meta)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def load_frames(path: str) -> List[Frame]:
    with open(path) as f:
        return frames_from_dict(json.load(f))


def frames_from_dict(d: dict) -> List[Frame]:
    return [
        Frame(
            frame=int(item["frame"]),
            kind=item.get("kind", "inbetween"),
            locked=item.get("locked", []),
            strokes={sd["id"]: Stroke.from_dict(sd) for sd in item["strokes"]},
        )
        for item in d["frames"]
    ]


# --------------------------------------------------------------------------
# Tensor helpers (match BrushStroke.get_path exactly, but as a pure function so
# optimization never hits BrushStroke's path cache)
# --------------------------------------------------------------------------

def logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def stroke_path(vertices: torch.Tensor, radii_logits: torch.Tensor, z: torch.Tensor,
                min_z: float) -> torch.Tensor:
    """(4,2) vertices, (2,) kappa logits, (4,) z -> (32, 3) path of (x, y, z)."""
    radii_m = radii_from_kappa(vertices, torch.sigmoid(radii_logits))
    xy = sample_filleted_polyline(vertices, radii_m, n_points=N_PATH_POINTS)
    zz = torch.nn.functional.interpolate(z[None, None], size=N_PATH_POINTS, mode="linear")[0, 0]
    zz = zz.clamp(min_z, 0.95)
    return torch.cat([xy, zz.unsqueeze(-1)], dim=-1)


def stroke_tensors(s: Stroke, device="cpu") -> dict:
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
    return {
        "vertices": t(s.vertices),
        "radii": logit(t(s.kappa)),
        "z": t(s.z),
        "xt": t([s.xt]),
        "yt": t([s.yt]),
        "a": t([s.a]),
    }


def stroke_from_tensors(template: Stroke, p: dict) -> Stroke:
    s = template.copy()
    s.vertices = p["vertices"].detach().cpu().tolist()
    s.kappa = torch.sigmoid(p["radii"]).detach().cpu().tolist()
    s.z = p["z"].detach().cpu().tolist()
    s.xt = float(p["xt"].reshape(-1)[0])
    s.yt = float(p["yt"].reshape(-1)[0])
    s.a = float(p["a"].reshape(-1)[0])
    return s


def wrap_angle(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi
