// Stroke geometry for the editor: a line-by-line port of FRIDA's
// filleted_stroke.py + BrushStroke.get_path + StrokeParametersToImage
// .trajectory_proportions, so the stage shows strokes exactly where the robot
// paints them. tests/test_editor_geometry.py checks it against Python.
"use strict";

const Geo = (() => {
  const N_PATH = 32;
  const EDGE_TRIM_FRAC = 1.45;
  const FLAT_ANGLE = 2e-2;
  const EPS = 1e-4;

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const mul = (a, s) => [a[0] * s, a[1] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const len = (a) => Math.hypot(a[0], a[1]);
  const norm = (v) => mul(v, 1 / Math.max(len(v), 1e-8));
  const perp = (v) => [-v[1], v[0]];
  const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
  const relu = (x) => Math.max(x, 0);

  function lineIntersect(p, d, q, e) {
    const den = d[0] * e[1] - d[1] * e[0];
    if (Math.abs(den) < 1e-10) return null;
    const w = sub(q, p);
    const a = (w[0] * e[1] - w[1] * e[0]) / den;
    return add(p, mul(d, a));
  }

  function maxFilletRadii(V) {
    const out = [];
    for (let k = 0; k < V.length - 2; k++) {
      const i = k + 1, P = V[i], A = V[i - 1], B = V[i + 1];
      const L0 = Math.max(len(sub(P, A)), EPS), L1 = Math.max(len(sub(B, P)), EPS);
      const ua = norm(sub(A, P)), ub = norm(sub(B, P));
      const alpha = Math.acos(clamp(dot(ua, ub), -1 + 1e-6, 1 - 1e-6));
      if (alpha < 1e-3 || Math.PI - alpha < FLAT_ANGLE) { out.push(0); continue; }
      out.push(Math.min(L0, L1) * EDGE_TRIM_FRAC * Math.tan(alpha * 0.5));
    }
    return out;
  }

  const radiiFromKappa = (V, kappa) => maxFilletRadii(V).map((r, i) => clamp(kappa[i], 0, 1) * r);

  function expandStroke(V, radii) {
    const n = V.length;
    const line = (a, b) => ({ p0: a, p1: mul(add(a, b), 0.5), p2: b, w1: 1 });
    if (n === 2) return [line(V[0], V[1])];
    const rReq = radii.map(relu);
    const trimIn = new Array(n).fill(0), trimOut = new Array(n).fill(0), metas = new Array(n).fill(null);
    for (let i = 1; i < n - 1; i++) {
      const rr = rReq[i - 1], P = V[i], A = V[i - 1], B = V[i + 1];
      const ua = norm(sub(A, P)), ub = norm(sub(B, P));
      const alpha = Math.acos(clamp(dot(ua, ub), -1 + 1e-6, 1 - 1e-6));
      if (alpha < 1e-3 || Math.PI - alpha < FLAT_ANGLE || rr < EPS) {
        metas[i] = { r: 0, alpha, ua, ub, skip: true }; continue;
      }
      const t = rr / Math.max(Math.tan(alpha * 0.5), 1e-6);
      trimIn[i] = t; trimOut[i] = t;
      metas[i] = { r: rr, alpha, ua, ub, skip: false, t };
    }
    for (let e = 0; e < n - 1; e++) {
      const L = Math.max(len(sub(V[e + 1], V[e])), EPS);
      const maxShare = Math.max(L * 0.5 - EPS, 0);
      let a = trimOut[e], b = trimIn[e + 1];
      const need = a + b;
      if (need > L - 2 * EPS && need > 0) { const s = (L - 2 * EPS) / Math.max(need, EPS); a *= s; b *= s; }
      trimOut[e] = Math.min(a, maxShare);
      trimIn[e + 1] = Math.min(b, maxShare);
    }
    for (let i = 1; i < n - 1; i++) {
      const m = metas[i];
      if (!m || m.skip) continue;
      const t = Math.min(trimIn[i], trimOut[i]);
      if (t < EPS) { m.skip = true; m.r = 0; trimIn[i] = 0; trimOut[i] = 0; continue; }
      trimIn[i] = t; trimOut[i] = t; m.t = t; m.r = t * Math.tan(m.alpha * 0.5);
    }
    const pieces = [];
    const addLine = (a, b) => { if (len(sub(a, b)) >= EPS) pieces.push(line(a, b)); };
    let cursor = V[0];
    for (let i = 1; i < n - 1; i++) {
      const P = V[i], m = metas[i];
      if (!m || m.skip || m.r < EPS) { addLine(cursor, P); cursor = P; continue; }
      const Tin = add(P, mul(m.ua, m.t)), Tout = add(P, mul(m.ub, m.t));
      addLine(cursor, Tin);
      const bis = norm(add(m.ua, m.ub));
      const center = add(P, mul(bis, m.r / Math.max(Math.sin(m.alpha * 0.5), 1e-6)));
      const radIn = sub(Tin, center), radOut = sub(Tout, center);
      let tanIn = norm(perp(radIn)), tanOut = norm(perp(radOut));
      const toOut = sub(Tout, Tin);
      if (dot(tanIn, toOut) < 0) tanIn = mul(tanIn, -1);
      if (dot(tanOut, toOut) < 0) tanOut = mul(tanOut, -1);
      const Q = lineIntersect(Tin, tanIn, Tout, tanOut);
      const theta = Math.acos(clamp(dot(norm(radIn), norm(radOut)), -1 + 1e-6, 1 - 1e-6));
      pieces.push({ p0: Tin, p1: Q || mul(add(Tin, Tout), 0.5), p2: Tout, w1: Math.max(Math.cos(theta * 0.5), 1e-4) });
      cursor = Tout;
    }
    addLine(cursor, V[n - 1]);
    if (!pieces.length) pieces.push(line(V[0], V[n - 1]));
    return pieces;
  }

  function samplePieces(pieces, spp) {
    const out = [];
    pieces.forEach((s, k) => {
      for (let j = k === 0 ? 0 : 1; j < spp; j++) {
        const t = j / (spp - 1), o = 1 - t;
        const b0 = o * o, b1 = 2 * o * t * s.w1, b2 = t * t;
        const den = Math.max(b0 + b1 + b2, 1e-12);
        out.push([(b0 * s.p0[0] + b1 * s.p1[0] + b2 * s.p2[0]) / den,
                  (b0 * s.p0[1] + b1 * s.p1[1] + b2 * s.p2[1]) / den]);
      }
    });
    return out;
  }

  // torch F.interpolate(mode="linear", align_corners=True)
  function resampleAligned(pts, n) {
    if (pts.length === n) return pts;
    if (pts.length === 1) return Array.from({ length: n }, () => pts[0]);
    const out = [];
    for (let i = 0; i < n; i++) {
      const x = (i * (pts.length - 1)) / (n - 1), i0 = Math.min(Math.floor(x), pts.length - 2), w = x - i0;
      out.push([pts[i0][0] * (1 - w) + pts[i0 + 1][0] * w, pts[i0][1] * (1 - w) + pts[i0 + 1][1] * w]);
    }
    return out;
  }

  // torch F.interpolate(mode="linear", align_corners=False)
  function interpUnaligned(vals, n) {
    const m = vals.length, out = [];
    for (let i = 0; i < n; i++) {
      let x = ((i + 0.5) * m) / n - 0.5;
      if (x < 0) x = 0;
      const i0 = Math.min(Math.floor(x), m - 1), i1 = Math.min(i0 + 1, m - 1), w = x - i0;
      out.push(vals[i0] * (1 - w) + vals[i1] * w);
    }
    return out;
  }

  function localPath(s) {
    const V = s.vertices;
    const pieces = expandStroke(V, radiiFromKappa(V, s.kappa));
    const spp = Math.max(8, Math.floor(N_PATH / Math.max(1, pieces.length)));
    return resampleAligned(samplePieces(pieces, spp), N_PATH);
  }

  function pathZ(s, cfg) {
    return interpUnaligned(s.z, N_PATH).map((z) => clamp(z, cfg.min_stroke_z, 0.95));
  }

  // stroke-local meters -> canvas proportions [x, y] (y down)
  function toCanvas(local, s, cfg) {
    const c = Math.cos(s.a), sn = Math.sin(s.a);
    return local.map(([lx, ly]) => {
      const y = -ly + cfg.y_b, x = lx + cfg.x_b;
      const xr = x * c + y * sn, yr = -x * sn + y * c;
      return [xr * cfg.x_m + s.xt, yr * cfg.y_m + s.yt];
    });
  }

  // canvas proportions -> stroke-local meters (inverse of toCanvas)
  function toLocal(p, s, cfg) {
    const c = Math.cos(s.a), sn = Math.sin(s.a);
    const xr = (p[0] - s.xt) / cfg.x_m, yr = (p[1] - s.yt) / cfg.y_m;
    const x = xr * c - yr * sn, y = xr * sn + yr * c;
    return [x - cfg.x_b, -(y - cfg.y_b)];
  }

  function sigmoidResponse(z) {
    const f = (v) => 1 / (1 + Math.exp(-12 * (v - 0.5)));
    return (f(z) - f(0)) / (f(1) - f(0));
  }

  // Painted width in canvas-height units (the renderer's soft profile has the
  // same ink as a solid line this wide).
  function thickness(z, cfg) {
    const zz = cfg.response === "placeholder_sigmoid" ? sigmoidResponse(z) : z;
    return Math.max(((zz * 2 + 0.5) / 64) * cfg.thick_mult + cfg.thick_bias, 0);
  }

  function canvasPath(s, cfg) {
    const pts = toCanvas(localPath(s), s, cfg);
    const z = pathZ(s, cfg);
    return pts.map((p, i) => [p[0], p[1], thickness(z[i], cfg)]);
  }

  return { canvasPath, localPath, pathZ, toCanvas, toLocal, thickness, N_PATH };
})();

if (typeof module !== "undefined") module.exports = Geo;
