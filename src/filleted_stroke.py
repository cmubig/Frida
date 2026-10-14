"""Filleted polyline (vertices + corner radii) → dense XY path.

Vendored/minimal from frida_new_brush_representation/pytorch_stroke/stroke_nurbs:
expand_stroke + sample_quadratic_pieces only (no geometry init / fit / CEM).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm().clamp_min(eps)


def _perp(v: torch.Tensor) -> torch.Tensor:
    return torch.stack([-v[1], v[0]])


def _line_intersect(
    p: torch.Tensor, d: torch.Tensor, q: torch.Tensor, e: torch.Tensor
) -> torch.Tensor | None:
    den = d[0] * e[1] - d[1] * e[0]
    if den.abs() < 1e-10:
        return None
    w = q - p
    a = (w[0] * e[1] - w[1] * e[0]) / den
    return p + a * d


def expand_stroke(
    vertices: torch.Tensor,
    radii: torch.Tensor,
    eps: float = 1e-4,
) -> dict[str, Any]:
    """
    Expand authoring polyline into piecewise rational quadratic Beziers.

    Args:
        vertices: (N, 2) corner points
        radii: (N-2,) fillet radii for interior vertices (non-negative)
    """
    if vertices.ndim != 2 or vertices.shape[-1] != 2:
        raise ValueError("vertices must be (N, 2)")
    n = vertices.shape[0]
    device = vertices.device
    dtype = vertices.dtype

    if n < 2:
        empty = vertices.new_zeros((0, 2))
        return {
            "points": empty,
            "weights": vertices.new_zeros((0,)),
            "pieces": [],
            "piece_count": 0,
        }

    if n == 2:
        a, b = vertices[0], vertices[1]
        mid = 0.5 * (a + b)
        one = a.new_ones(())
        pieces = [
            {"p0": a, "p1": mid, "p2": b, "w0": one, "w1": one, "w2": one, "kind": "line"}
        ]
        return {
            "points": torch.stack([a, mid, b]),
            "weights": torch.ones(3, device=device, dtype=dtype),
            "pieces": pieces,
            "piece_count": 1,
        }

    if radii.numel() == n:
        r_int = radii[1:-1]
    elif radii.numel() == n - 2:
        r_int = radii
    else:
        raise ValueError(f"radii must have length N-2={n-2} or N={n}, got {radii.numel()}")

    r_req = torch.relu(r_int)

    zero = vertices.new_zeros(())
    trim_in = [zero] * n
    trim_out = [zero] * n
    metas = [None] * n

    for i in range(1, n - 1):
        rr = r_req[i - 1]
        P = vertices[i]
        A = vertices[i - 1]
        B = vertices[i + 1]
        ua = _norm(A - P)
        ub = _norm(B - P)
        dot = torch.clamp((ua * ub).sum(), -1.0 + 1e-6, 1.0 - 1e-6)
        alpha = torch.acos(dot)
        near_flat = (alpha < 1e-3) | ((torch.pi - alpha) < 1e-3) | (rr < eps)
        if bool(near_flat.detach()):
            metas[i] = {"r": rr * 0, "alpha": alpha, "ua": ua, "ub": ub, "skip": True}
            continue
        t = rr / torch.tan(alpha * 0.5).clamp_min(1e-6)
        trim_in[i] = t
        trim_out[i] = t
        metas[i] = {"r": rr, "alpha": alpha, "ua": ua, "ub": ub, "skip": False, "t": t}

    for e in range(n - 1):
        L = (vertices[e + 1] - vertices[e]).norm().clamp_min(eps)
        max_share = (L * 0.5 - eps).clamp_min(0.0)
        a = trim_out[e]
        b = trim_in[e + 1]
        need = a + b
        if float(need.detach()) > float((L - 2 * eps).detach()) and float(need.detach()) > 0:
            scale = (L - 2 * eps) / need.clamp_min(eps)
            a = a * scale
            b = b * scale
        trim_out[e] = torch.minimum(a, max_share)
        trim_in[e + 1] = torch.minimum(b, max_share)

    for i in range(1, n - 1):
        meta = metas[i]
        if meta is None or meta["skip"]:
            continue
        t = torch.minimum(trim_in[i], trim_out[i])
        if float(t.detach()) < eps:
            meta["skip"] = True
            meta["r"] = t * 0
            trim_in[i] = t * 0
            trim_out[i] = t * 0
            continue
        trim_in[i] = t
        trim_out[i] = t
        meta["t"] = t
        meta["r"] = t * torch.tan(meta["alpha"] * 0.5)

    pieces: list[dict] = []

    def add_line(a: torch.Tensor, b: torch.Tensor) -> None:
        if float((a - b).norm().detach()) < eps:
            return
        mid = 0.5 * (a + b)
        one = a.new_ones(())
        pieces.append(
            {"p0": a, "p1": mid, "p2": b, "w0": one, "w1": one, "w2": one, "kind": "line"}
        )

    cursor = vertices[0]
    for i in range(1, n - 1):
        P = vertices[i]
        meta = metas[i]
        if meta is None or meta["skip"] or float(meta["r"].detach()) < eps:
            add_line(cursor, P)
            cursor = P
            continue

        t = meta["t"]
        Tin = P + meta["ua"] * t
        Tout = P + meta["ub"] * t
        add_line(cursor, Tin)

        bis = _norm(meta["ua"] + meta["ub"])
        sin_half = torch.sin(meta["alpha"] * 0.5).clamp_min(1e-6)
        center = P + bis * (meta["r"] / sin_half)
        rad_in = Tin - center
        rad_out = Tout - center
        tan_in = _norm(_perp(rad_in))
        tan_out = _norm(_perp(rad_out))
        to_out = Tout - Tin
        if float((tan_in * to_out).sum().detach()) < 0:
            tan_in = -tan_in
        if float((tan_out * to_out).sum().detach()) < 0:
            tan_out = -tan_out
        Q = _line_intersect(Tin, tan_in, Tout, tan_out)
        mid_cp = Q if Q is not None else 0.5 * (Tin + Tout)
        cdot = torch.clamp(
            (_norm(rad_in) * _norm(rad_out)).sum(), -1.0 + 1e-6, 1.0 - 1e-6
        )
        theta = torch.acos(cdot)
        w = torch.cos(theta * 0.5).clamp_min(1e-4)
        one = Tin.new_ones(())
        pieces.append(
            {
                "p0": Tin,
                "p1": mid_cp,
                "p2": Tout,
                "w0": one,
                "w1": w,
                "w2": one,
                "kind": "arc",
            }
        )
        cursor = Tout

    add_line(cursor, vertices[n - 1])

    if not pieces:
        a, b = vertices[0], vertices[n - 1]
        mid = 0.5 * (a + b)
        one = a.new_ones(())
        pieces = [
            {"p0": a, "p1": mid, "p2": b, "w0": one, "w1": one, "w2": one, "kind": "line"}
        ]

    return {
        "pieces": pieces,
        "piece_count": len(pieces),
    }


def sample_quadratic_pieces(
    pieces: list[dict],
    samples_per_piece: int = 24,
) -> torch.Tensor:
    """Evaluate piecewise rational quadratic Beziers and concatenate. Returns (S, 2)."""
    if not pieces:
        return torch.zeros((0, 2))

    all_pts = []
    for seg in pieces:
        p0, p1, p2 = seg["p0"], seg["p1"], seg["p2"]
        w0, w1, w2 = seg["w0"], seg["w1"], seg["w2"]
        t = torch.linspace(0.0, 1.0, samples_per_piece, device=p0.device, dtype=p0.dtype)
        omt = 1.0 - t
        b0 = omt * omt
        b1 = 2.0 * omt * t
        b2 = t * t
        num = (
            (b0 * w0).unsqueeze(-1) * p0
            + (b1 * w1).unsqueeze(-1) * p1
            + (b2 * w2).unsqueeze(-1) * p2
        )
        den = (b0 * w0 + b1 * w1 + b2 * w2).unsqueeze(-1).clamp_min(1e-12)
        pts = num / den
        if all_pts:
            pts = pts[1:]
        all_pts.append(pts)
    return torch.cat(all_pts, dim=0)


def _resample_polyline(pts: torch.Tensor, n_points: int) -> torch.Tensor:
    """Resample (S, 2) to exactly (n_points, 2) via linear interpolation."""
    if pts.shape[0] == 0:
        return pts.new_zeros((n_points, 2))
    if pts.shape[0] == 1:
        return pts.expand(n_points, -1).contiguous()
    if pts.shape[0] == n_points:
        return pts
    # (1, 2, S) -> interpolate along last dim
    x = pts.T.unsqueeze(0)
    x = F.interpolate(x, size=n_points, mode="linear", align_corners=True)
    return x[0].T.contiguous()


def project_radii(vertices: torch.Tensor, radii: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Clamp each interior radius so trims fit on adjacent edges."""
    n = vertices.shape[0]
    if n < 3 or radii.numel() == 0:
        return radii
    out = []
    for k in range(n - 2):
        i = k + 1
        P = vertices[i]
        A = vertices[i - 1]
        B = vertices[i + 1]
        L0 = (P - A).norm().clamp_min(eps)
        L1 = (B - P).norm().clamp_min(eps)
        ua = _norm(A - P)
        ub = _norm(B - P)
        dot = torch.clamp((ua * ub).sum(), -1.0 + 1e-6, 1.0 - 1e-6)
        alpha = torch.acos(dot)
        if float(alpha.detach()) < 1e-3 or float((torch.pi - alpha).detach()) < 1e-3:
            out.append(radii.new_zeros(()))
            continue
        t_max = torch.minimum(L0, L1) * 0.45
        r_max = t_max * torch.tan(alpha * 0.5)
        out.append(torch.minimum(torch.relu(radii[k]), r_max))
    return torch.stack(out)


def sample_filleted_polyline(
    vertices: torch.Tensor,
    radii: torch.Tensor,
    n_points: int = 32,
) -> torch.Tensor:
    """
    Densify filleted polyline to exactly n_points XY samples.

    Args:
        vertices: (N, 2) in meters
        radii: (N-2,) non-negative fillet radii in meters
        n_points: output length (Frida renderer expects 32)

    Returns:
        (n_points, 2)
    """
    out = expand_stroke(vertices, radii)
    spp = max(8, n_points // max(1, out["piece_count"]))
    dense = sample_quadratic_pieces(out["pieces"], samples_per_piece=spp)
    return _resample_polyline(dense, n_points)
