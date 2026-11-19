// Flipbook editor: Animate/Flare-style authoring for robot-painted animation.
//
// Two ways to make the next keyframe, as in Animate:
//   F6  Insert Keyframe: copies the current drawing; reshape its strokes
//       (Selection / Subselection tools). Strokes keep their identity.
//   F7  Insert Blank Keyframe: empty frame with onion skin; redraw strokes
//       over their ghosts. Each traced stroke replaces the ghost it was drawn
//       over (Alt+click a ghost first to choose it explicitly).
// Then Create In-betweens fills the frames between keyframes using the Python
// pipeline. Changing a stroke on an in-between pins it for that frame.
"use strict";

// ---------------------------------------------------------------- state ---
const S = {
  cfg: null,
  doc: null,
  gen: { frames: {}, stale: true, method: null, metrics: null },
  ui: {
    frame: 0, tool: "brush", sel: new Set(), onion: true, preview: false,
    playing: false, color: "#0d0d0d", picking: null, linkTarget: null,
    suggest: null, spaceHand: false, busy: false,
  },
  view: { scale: 1, ox: 0, oy: 0, fit: true },
  hist: { undo: [], redo: [] },
  previewCache: new Map(),
};
const $ = (id) => document.getElementById(id);
const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
const clone = (o) => JSON.parse(JSON.stringify(o));
const EASES = [["linear", "Linear"], ["ease_in", "Ease In"], ["ease_out", "Ease Out"], ["ease_in_out", "Ease In-Out"]];

// ------------------------------------------------------------------ api ---
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

// ------------------------------------------------------------- document ---
function newDoc() {
  return {
    format: "frida-inbetween", version: 1, cache_dir: S.cfg.cache_dir,
    keys: [{ frame: 0, strokes: [] }], edits: [], timing: {},
    editor: { fps: 8, length: 24, next_id: 1, name: "Untitled" },
  };
}
const keys = () => S.doc.keys;
const keyAt = (f) => keys().find((k) => k.frame === f) || null;
const prevKey = (f) => keys().filter((k) => k.frame < f).sort((a, b) => b.frame - a.frame)[0] || null;
const nextKey = (f) => keys().filter((k) => k.frame > f).sort((a, b) => a.frame - b.frame)[0] || null;
const keyAtOrBefore = (f) => keyAt(f) || prevKey(f);
const editAt = (f) => S.doc.edits.find((e) => e.frame === f) || null;
const lastFrame = () => Math.max(S.doc.editor.length - 1, ...keys().map((k) => k.frame));
const isBetween = (f) => !keyAt(f) && !!prevKey(f) && !!nextKey(f);
function sortKeys() { keys().sort((a, b) => a.frame - b.frame); }
function newId() { return "s" + S.doc.editor.next_id++; }

// ------------------------------------------------------------ clipboard ---
// Copy / cut / paste strokes like Animate. Regular paste and duplicate make
// new strokes (new IDs, they pop in). Paste in Place keeps a stroke's ID when
// this keyframe doesn't have that stroke yet, so the in-betweens animate it
// from where it was copied (handy for a stroke you forgot to trace).
let clip = [];
const selected = () => content(S.ui.frame).strokes.filter((s) => S.ui.sel.has(s.id));
function pasteStrokes(strokes, { inPlace = false } = {}) {
  const f = S.ui.frame, k = keyAt(f);
  if (!strokes.length) return toast("Nothing to paste.");
  if (!k) return toast("Paste onto a keyframe (F6 / F7 makes one here).");
  pushUndo();
  const have = new Set(k.strokes.map((s) => s.id));
  const top = Math.max(-1, ...k.strokes.map((s) => s.order), ...(prevKey(f)?.strokes || []).map((s) => s.order));
  const dx = inPlace ? 0 : 10 / paperW(), dy = inPlace ? 0 : 10 / paperH();
  const sorted = [...strokes].sort((a, b) => a.order - b.order || a.id.localeCompare(b.id));
  const ids = [];
  let next = top;
  for (const src of sorted) {
    const s = clone(src);
    s.xt += dx; s.yt += dy;
    if (inPlace && !have.has(s.id)) {
      const g = prevKey(f)?.strokes.find((x) => x.id === s.id);
      if (g) s.order = g.order; else s.order = ++next;
    } else { s.id = newId(); s.order = ++next; }
    have.add(s.id); k.strokes.push(s); ids.push(s.id);
  }
  S.ui.sel = new Set(ids);
  changed(); renderAll();
  return ids;
}

// What is on frame f, and how it can be edited.
function content(f) {
  const k = keyAt(f);
  if (k) return { strokes: k.strokes, mode: "key" };
  const g = S.gen.frames[f];
  if (isBetween(f) && g) {
    const e = editAt(f);
    const strokes = g.strokes.map((s) => (e && e.strokes.find((x) => x.id === s.id)) || s);
    return { strokes, mode: "between", pinned: new Set(e ? e.strokes.map((s) => s.id) : []) };
  }
  const h = keyAtOrBefore(f);
  return { strokes: h ? h.strokes : [], mode: "held" };
}

function pushUndo() {
  S.hist.undo.push(JSON.stringify(S.doc));
  if (S.hist.undo.length > 200) S.hist.undo.shift();
  S.hist.redo = [];
}
function markStale() { if (Object.keys(S.gen.frames).length) S.gen.stale = true; }
function changed() { markStale(); render(); }

// Apply fn to a stroke on the current frame (keys directly; in-betweens pin it).
function mutateStroke(id, fn) {
  const f = S.ui.frame, c = content(f);
  if (c.mode === "key") {
    const s = c.strokes.find((x) => x.id === id);
    if (s) fn(s);
  } else if (c.mode === "between") {
    let e = editAt(f);
    if (!e) { e = { frame: f, strokes: [] }; S.doc.edits.push(e); }
    let s = e.strokes.find((x) => x.id === id);
    if (!s) { s = clone(c.strokes.find((x) => x.id === id)); e.strokes.push(s); }
    fn(s);
  }
}
function canEdit(quiet) {
  const m = content(S.ui.frame).mode;
  if (m !== "held") return true;
  if (!quiet) toast(isBetween(S.ui.frame)
    ? "Create in-betweens first, or insert a keyframe here (F6 / F7)."
    : "Insert a keyframe here (F6 copies the drawing, F7 starts blank) to change this frame.");
  return false;
}

// --------------------------------------------------------------- stage ---
const cv = $("canvas"), ctx = cv.getContext("2d");
const aspect = () => S.cfg.canvas_width_m / S.cfg.canvas_height_m;
const paperH = () => S.view.scale;
const paperW = () => S.view.scale * aspect();
const toScreen = (p) => [S.view.ox + p[0] * paperW(), S.view.oy + p[1] * paperH()];
const toProp = (x, y) => [(x - S.view.ox) / paperW(), (y - S.view.oy) / paperH()];

function resizeStage() {
  const r = $("stage").getBoundingClientRect(), d = devicePixelRatio || 1;
  cv.width = r.width * d; cv.height = r.height * d;
  ctx.setTransform(d, 0, 0, d, 0, 0);
  if (S.view.fit) zoomFit();
  render();
}
function zoomFit() {
  const r = $("stage").getBoundingClientRect();
  const s = Math.max(50, Math.min((r.width - 60) / aspect(), r.height - 60));
  S.view = { scale: s, ox: (r.width - s * aspect()) / 2, oy: (r.height - s) / 2, fit: true };
  $("zoomsel").value = "fit";
}
function zoomAt(factor, cx, cy) {
  const r = $("stage").getBoundingClientRect();
  if (cx === undefined) { cx = r.width / 2; cy = r.height / 2; }
  const p = toProp(cx, cy);
  S.view.scale = clamp(S.view.scale * factor, 40, 20000);
  S.view.ox = cx - p[0] * paperW(); S.view.oy = cy - p[1] * paperH();
  S.view.fit = false;
  const actual = S.view.scale / (S.cfg.canvas_height_m / 0.0254 * 96);
  $("zoomsel").value = ["0.5", "1", "2", "4"].find((v) => Math.abs(+v - actual) < 0.01) || "";
  render();
}
function zoomTo(level) {
  if (level === "fit") { zoomFit(); render(); return; }
  const target = (+level) * (S.cfg.canvas_height_m / 0.0254 * 96); // 100% = actual size at 96 dpi
  zoomAt(target / S.view.scale);
}

function strokeScreenPath(s) {
  return Geo.canvasPath(s, S.cfg).map((p) => [...toScreen(p), p[2] * paperH()]);
}
// Strokes are drawn opaque (segments overlap at the joints, so drawing them
// with transparency would show beads); faint ghosts go through an offscreen
// layer that is composited once with the requested alpha.
const layer = document.createElement("canvas"), lctx = layer.getContext("2d");
function drawStroke(s, opts = {}) {
  const alpha = opts.alpha ?? 1;
  if (alpha < 1) {
    if (layer.width !== cv.width || layer.height !== cv.height) { layer.width = cv.width; layer.height = cv.height; }
    const d = devicePixelRatio || 1;
    lctx.setTransform(1, 0, 0, 1, 0, 0); lctx.clearRect(0, 0, layer.width, layer.height); lctx.setTransform(d, 0, 0, d, 0, 0);
    const pts = strokeOnto(lctx, s, opts);
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = alpha; ctx.drawImage(layer, 0, 0); ctx.restore();
    return pts;
  }
  return strokeOnto(ctx, s, opts);
}
function strokeOnto(g, s, opts) {
  const pts = strokeScreenPath(s), color = opts.color || cssColor(s.color);
  const visible = pts.some((p) => p[2] > 0.05);
  g.save();
  g.lineCap = "round"; g.lineJoin = "round";
  if (!visible) { // too light to leave a mark: dashed outline so it can still be found
    g.setLineDash([4, 3]); g.strokeStyle = opts.color || "rgba(120,120,120,.9)"; g.lineWidth = 1;
    g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.stroke();
  } else {
    g.strokeStyle = color;
    for (let i = 0; i < pts.length - 1; i++) {
      const w = (pts[i][2] + pts[i + 1][2]) / 2;
      if (w <= 0.05) continue;
      g.lineWidth = Math.max(w, opts.minWidth || 0);
      g.beginPath(); g.moveTo(pts[i][0], pts[i][1]); g.lineTo(pts[i + 1][0], pts[i + 1][1]); g.stroke();
    }
  }
  g.restore();
  return pts;
}
function cssColor(c) {
  if (S.cfg.ink) return "#111";
  return `rgb(${c.map((v) => Math.round(v * 255)).join(",")})`;
}
function hexToRgb(h) { return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255); }
function rgbToHex(c) { return "#" + c.map((v) => Math.round(clamp(v, 0, 1) * 255).toString(16).padStart(2, "0")).join(""); }

function ghostSets(f) {
  const out = [];
  if (!S.ui.onion) return out;
  const p = prevKey(f), n = nextKey(f);
  if (p) out.push({ key: p, rgb: "64,140,255" });
  if (n) out.push({ key: n, rgb: "70,190,110" });
  return out;
}
// Strokes of the previous keyframe not yet traced in this keyframe.
function traceCandidates(f) {
  const k = keyAt(f), p = prevKey(f);
  if (!k || !p) return [];
  const have = new Set(k.strokes.map((s) => s.id));
  return p.strokes.filter((s) => !have.has(s.id));
}

function render() {
  if (!S.cfg) return;
  const r = $("stage").getBoundingClientRect();
  ctx.fillStyle = "#3a3a3a"; ctx.fillRect(0, 0, r.width, r.height);
  ctx.save(); ctx.shadowColor = "rgba(0,0,0,.45)"; ctx.shadowBlur = 14;
  ctx.fillStyle = "#fff"; ctx.fillRect(S.view.ox, S.view.oy, paperW(), paperH()); ctx.restore();

  const f = S.ui.frame, c = content(f);
  // onion skin ghosts
  const candidates = new Set(traceCandidates(f).map((s) => s.id));
  for (const g of ghostSets(f)) {
    for (const s of g.key.strokes) {
      const target = S.ui.linkTarget === s.id || (S.ui.suggest === s.id);
      drawStroke(s, { color: target ? "rgb(255,150,40)" : `rgb(${g.rgb})`,
                      alpha: target ? 0.8 : (candidates.has(s.id) ? 0.45 : 0.22), minWidth: 1.5 });
    }
  }
  // drawing or paint preview
  const pv = S.ui.preview && previewFor(c.strokes);
  if (pv) ctx.drawImage(pv, S.view.ox, S.view.oy, paperW(), paperH());
  else for (const s of c.strokes) drawStroke(s);
  if (c.mode === "held") { // dim held frames like a static frame in Animate
    ctx.fillStyle = "rgba(255,255,255,.35)"; ctx.fillRect(S.view.ox, S.view.oy, paperW(), paperH());
  }
  // pinned strokes on in-betweens
  if (c.mode === "between") for (const s of c.strokes) if (c.pinned.has(s.id)) {
    const p = Geo.canvasPath(s, S.cfg); const [x, y] = toScreen(p[0]);
    ctx.fillStyle = "#f0a030"; diamond(x, y - 8, 4);
  }
  // selection
  for (const s of c.strokes) if (S.ui.sel.has(s.id)) drawSelection(s);
  if (S.ui.sel.size && (S.ui.tool === "selection")) drawRotateHandle(c.strokes.filter((s) => S.ui.sel.has(s.id)));
  // selection rectangle
  if (drag && drag.kind === "marquee") {
    const x = Math.min(drag.x0, drag.x1), y = Math.min(drag.y0, drag.y1);
    const w = Math.abs(drag.x1 - drag.x0), h = Math.abs(drag.y1 - drag.y0);
    ctx.save(); ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.strokeRect(x + 0.5, y + 0.5, w, h);
    ctx.setLineDash([4, 3]); ctx.strokeStyle = "#222"; ctx.strokeRect(x + 0.5, y + 0.5, w, h);
    ctx.restore();
  }
  // live brush
  if (drag && drag.kind === "brush") drawLive(drag.pts);
  if (pendingFit) drawLive(pendingFit, true);
  updateModeHint();
}
function diamond(x, y, r) { ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.fill(); }

function drawSelection(s) {
  const pts = strokeScreenPath(s);
  ctx.save(); ctx.strokeStyle = "#2d8ceb"; ctx.lineWidth = 1.2;
  ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.stroke();
  if (S.ui.tool === "subselection") {
    const V = Geo.toCanvas(s.vertices, s, S.cfg).map(toScreen);
    ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(45,140,235,.6)";
    ctx.beginPath(); V.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.stroke();
    ctx.setLineDash([]);
    V.forEach((p) => { ctx.fillStyle = "#fff"; ctx.fillRect(p[0] - 4, p[1] - 4, 8, 8); ctx.strokeStyle = "#2d8ceb"; ctx.strokeRect(p[0] - 4, p[1] - 4, 8, 8); });
  } else {
    const b = bbox(pts);
    ctx.setLineDash([4, 3]); ctx.strokeRect(b[0] - 4, b[1] - 4, b[2] - b[0] + 8, b[3] - b[1] + 8);
  }
  ctx.restore();
}
function bbox(pts) {
  return [Math.min(...pts.map((p) => p[0])), Math.min(...pts.map((p) => p[1])),
          Math.max(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[1]))];
}
function selBox(strokes) { return bbox(strokes.flatMap((s) => strokeScreenPath(s))); }
function rotateHandlePos(strokes) { const b = selBox(strokes); return [(b[0] + b[2]) / 2, b[1] - 26]; }
function drawRotateHandle(strokes) {
  if (!strokes.length) return;
  const b = selBox(strokes), h = rotateHandlePos(strokes);
  ctx.save(); ctx.strokeStyle = "#2d8ceb"; ctx.beginPath(); ctx.moveTo(h[0], b[1] - 4); ctx.lineTo(h[0], h[1] + 5); ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(h[0], h[1], 5, 0, 7); ctx.fill(); ctx.stroke(); ctx.restore();
}
function drawLive(pts, pending) {
  if (pts.length < 1) return;
  ctx.save(); ctx.lineCap = "round"; ctx.strokeStyle = pending ? "rgba(17,17,17,.4)" : "#111";
  for (let i = 0; i < pts.length - 1; i++) {
    const a = toScreen(pts[i]), b = toScreen(pts[i + 1]);
    ctx.lineWidth = Math.max(1, Geo.thickness(pressureToZ(pts[i][2]), S.cfg) * paperH());
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  ctx.restore();
}
const pressureToZ = (p) => S.cfg.visible_z_min + (0.95 - S.cfg.visible_z_min) * clamp(p, 0, 1);

// Exact stroke-model render from the server, cached per drawing.
let previewTimer = null;
function previewFor(strokes) {
  const key = JSON.stringify(strokes);
  const hit = S.previewCache.get(key);
  if (hit instanceof Image && hit.complete) return hit;
  if (!hit) {
    S.previewCache.set(key, "pending");
    clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
      try {
        const { png } = await api("/api/render", { strokes, height: Math.round(Math.min(768, paperH() * 1.5)) });
        const img = new Image(); img.onload = render; img.src = png;
        if (S.previewCache.size > 200) S.previewCache.clear();
        S.previewCache.set(key, img);
      } catch (e) { S.previewCache.delete(key); toast("Preview failed: " + e.message); }
    }, 150);
  }
  return null;
}

// ------------------------------------------------------------ hit tests ---
function distToPath(x, y, pts) {
  let best = Infinity;
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = pts[i], [bx, by] = pts[i + 1];
    const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy || 1e-9;
    const t = clamp(((x - ax) * dx + (y - ay) * dy) / L, 0, 1);
    best = Math.min(best, Math.hypot(x - ax - t * dx, y - ay - t * dy) - (pts[i][2] || 0) / 2);
  }
  return best;
}
function hitStroke(x, y, strokes) {
  let best = null, bd = 7;
  for (const s of strokes) { const d = distToPath(x, y, strokeScreenPath(s)); if (d < bd) { bd = d; best = s; } }
  return best;
}
function hitGhost(x, y) {
  const p = prevKey(S.ui.frame);
  return p ? hitStroke(x, y, p.strokes) : null;
}

// ----------------------------------------------------------- pointer ---
let drag = null, pendingFit = null;
function tool() { return S.ui.spaceHand ? "hand" : S.ui.tool; }

cv.addEventListener("pointerdown", (e) => {
  if (S.ui.playing) stop();
  cv.setPointerCapture(e.pointerId);
  const x = e.offsetX, y = e.offsetY, t = tool(), f = S.ui.frame, c = content(f);
  if (e.button === 1 || t === "hand") { drag = { kind: "pan", x, y, ox: S.view.ox, oy: S.view.oy }; return; }
  if (t === "zoom") { zoomAt(e.altKey ? 0.5 : 2, x, y); return; }

  if (S.ui.picking) { pickGhost(x, y); return; }

  if (t === "brush") {
    if (e.altKey) { // choose which ghost the next stroke traces
      const g = traceCandidates(f).length ? hitStroke(x, y, traceCandidates(f)) : null;
      if (g) { S.ui.linkTarget = g.id; toast(`Next stroke will replace ghost ${g.id}.`); render(); }
      else toast("Alt+click a faint ghost stroke from the previous keyframe to trace it.");
      return;
    }
    if (c.mode !== "key") {
      toast(c.mode === "between" ? "Draw on keyframes. On in-betweens, adjust strokes with the Selection tools."
                                 : "Insert a keyframe first (F6 copies the drawing, F7 starts blank).");
      return;
    }
    drag = { kind: "brush", pts: [] }; addBrushPoints(e); render(); return;
  }

  // selection / subselection
  if (t === "subselection" && S.ui.sel.size) {
    for (const s of c.strokes) if (S.ui.sel.has(s.id)) {
      const V = Geo.toCanvas(s.vertices, s, S.cfg).map(toScreen);
      const i = V.findIndex((p) => Math.abs(p[0] - x) < 7 && Math.abs(p[1] - y) < 7);
      if (i >= 0) {
        if (!canEdit()) return;
        pushUndo(); drag = { kind: "vertex", id: s.id, i }; return;
      }
    }
  }
  if (t === "selection" && S.ui.sel.size) {
    const sel = c.strokes.filter((s) => S.ui.sel.has(s.id));
    const h = rotateHandlePos(sel);
    if (Math.hypot(x - h[0], y - h[1]) < 9) {
      if (!canEdit()) return;
      const b = selBox(sel);
      pushUndo();
      drag = { kind: "rotate", cx: (b[0] + b[2]) / 2, cy: (b[1] + b[3]) / 2, start: Math.atan2(y - (b[1] + b[3]) / 2, x - (b[0] + b[2]) / 2),
               orig: sel.map((s) => ({ id: s.id, a: s.a, xt: s.xt, yt: s.yt, center: center(s) })) };
      return;
    }
  }
  const hit = hitStroke(x, y, c.strokes);
  if (hit) {
    if (e.shiftKey) { S.ui.sel.has(hit.id) ? S.ui.sel.delete(hit.id) : S.ui.sel.add(hit.id); }
    else if (!S.ui.sel.has(hit.id)) { S.ui.sel = new Set([hit.id]); }
    if (t === "selection" && c.mode !== "held") {
      drag = { kind: "move", start: toProp(x, y), moved: false,
               orig: c.strokes.filter((s) => S.ui.sel.has(s.id)).map((s) => ({ id: s.id, xt: s.xt, yt: s.yt })) };
    }
  } else {
    // empty spot: drag a dashed rectangle to select every stroke fully inside it
    // (Shift+drag adds to the selection), as with Animate's Selection tool
    if (!e.shiftKey) S.ui.sel.clear();
    drag = { kind: "marquee", x0: x, y0: y, x1: x, y1: y, base: new Set(S.ui.sel) };
  }
  renderAll();
});

cv.addEventListener("pointermove", (e) => {
  const x = e.offsetX, y = e.offsetY;
  if (!drag) {
    cv.style.cursor = { hand: "grab", zoom: "zoom-in", brush: "crosshair" }[tool()] || "default";
    return;
  }
  if (drag.kind === "pan") { S.view.ox = drag.ox + x - drag.x; S.view.oy = drag.oy + y - drag.y; S.view.fit = false; render(); return; }
  if (drag.kind === "brush") {
    addBrushPoints(e);
    const cand = traceCandidates(S.ui.frame);
    S.ui.suggest = S.ui.linkTarget ? null : (S.ui.onion && cand.length ? nearestGhost(drag.pts, cand) : null);
    render(); return;
  }
  if (drag.kind === "marquee") {
    drag.x1 = x; drag.y1 = y;
    S.ui.sel = new Set([...drag.base, ...strokesInRect(content(S.ui.frame).strokes, drag)]);
    render(); return;
  }
  if (drag.kind === "move") {
    const p = toProp(x, y), dx = p[0] - drag.start[0], dy = p[1] - drag.start[1];
    if (!drag.moved) { if (Math.hypot(dx * paperW(), dy * paperH()) < 2) return; if (!canEdit()) { drag = null; return; } pushUndo(); drag.moved = true; }
    for (const o of drag.orig) mutateStroke(o.id, (s) => { s.xt = o.xt + dx; s.yt = o.yt + dy; });
    changed(); return;
  }
  if (drag.kind === "vertex") {
    const s0 = content(S.ui.frame).strokes.find((s) => s.id === drag.id);
    const local = Geo.toLocal(toProp(x, y), s0, S.cfg);
    mutateStroke(drag.id, (s) => { s.vertices[drag.i] = local.map((v) => clamp(v, -0.15, 0.15)); });
    changed(); return;
  }
  if (drag.kind === "rotate") {
    const delta = Math.atan2(y - drag.cy, x - drag.cx) - drag.start;
    for (const o of drag.orig) {
      // rotate the stroke's center about the selection center (screen space)
      const [sx, sy] = toScreen(o.center), c = Math.cos(delta), sn = Math.sin(delta);
      const want = toProp(drag.cx + (sx - drag.cx) * c - (sy - drag.cy) * sn, drag.cy + (sx - drag.cx) * sn + (sy - drag.cy) * c);
      mutateStroke(o.id, (s) => {
        s.a = o.a - delta; s.xt = o.xt; s.yt = o.yt;
        const now = center(s); s.xt += want[0] - now[0]; s.yt += want[1] - now[1];
      });
    }
    changed();
  }
});

cv.addEventListener("pointerup", async () => {
  const d = drag; drag = null;
  if (!d) return;
  if (d.kind === "brush") await finishBrush(d.pts);
  renderAll();
});

cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.ctrlKey || e.metaKey) zoomAt(Math.exp(-e.deltaY * 0.002), e.offsetX, e.offsetY);
  else { S.view.ox -= e.deltaX; S.view.oy -= e.deltaY; S.view.fit = false; render(); }
}, { passive: false });

function strokesInRect(strokes, r) {
  const x0 = Math.min(r.x0, r.x1), x1 = Math.max(r.x0, r.x1), y0 = Math.min(r.y0, r.y1), y1 = Math.max(r.y0, r.y1);
  if (x1 - x0 < 3 && y1 - y0 < 3) return [];
  return strokes.filter((s) => strokeScreenPath(s).every((p) => p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1))
                .map((s) => s.id);
}
function center(s) {
  const p = Geo.canvasPath(s, S.cfg);
  return [p.reduce((a, q) => a + q[0], 0) / p.length, p.reduce((a, q) => a + q[1], 0) / p.length];
}
function addBrushPoints(e) {
  const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  for (const ev of (evs.length ? evs : [e])) {
    const r = cv.getBoundingClientRect();
    const p = toProp(ev.clientX - r.left, ev.clientY - r.top);
    const pr = ev.pointerType === "pen" ? ev.pressure : 0.5;
    const last = drag.pts[drag.pts.length - 1];
    if (!last || Math.hypot((p[0] - last[0]) * paperW(), (p[1] - last[1]) * paperH()) > 1.5) drag.pts.push([p[0], p[1], pr]);
  }
}
// Mean distance (in paper heights) from a drawn mark to a ghost stroke.
function markDistance(pts, s) {
  const path = Geo.canvasPath(s, S.cfg).map((p) => [p[0] * aspect(), p[1]]);
  let tot = 0;
  for (const q of pts) {
    const x = q[0] * aspect(), y = q[1];
    let best = Infinity;
    for (const p of path) best = Math.min(best, Math.hypot(x - p[0], y - p[1]));
    tot += best;
  }
  return tot / Math.max(pts.length, 1);
}
function nearestGhost(pts, cand) {
  let best = null, bd = 0.06; // within 6% of the paper height
  for (const s of cand) { const d = markDistance(pts, s); if (d < bd) { bd = d; best = s.id; } }
  return best;
}

async function finishBrush(pts) {
  const f = S.ui.frame, k = keyAt(f);
  if (!k || !pts.length) return;
  const target = S.ui.linkTarget || S.ui.suggest;
  S.ui.suggest = null; S.ui.linkTarget = null;
  pendingFit = pts; render();
  try {
    const color = S.cfg.ink ? [0.05, 0.05, 0.05] : hexToRgb(S.ui.color);
    const res = await api("/api/fit", { points: pts.map((p) => [p[0], p[1]]), pressure: pts.map((p) => p[2]), color });
    // A mark too complex for one robot stroke comes back split into pieces, in drawing order.
    const pieces = (res.strokes || [{ stroke: res.stroke, error_mm: res.error_mm }]).map((x) => x.stroke);
    const worst = res.error_mm;
    const cand = traceCandidates(f);
    const ghost = target && cand.find((g) => g.id === target);
    const piecePts = (s) => Geo.canvasPath(s, S.cfg).map((p) => [p[0], p[1]]);
    // Link pieces to ghosts: the chosen ghost goes to the piece nearest it; when tracing
    // with onion skin, each other piece takes its nearest unused ghost.
    const link = new Map();
    if (ghost) {
      let bi = 0, bd = Infinity;
      pieces.forEach((s, i) => { const d = markDistance(piecePts(s), ghost); if (d < bd) { bd = d; bi = i; } });
      link.set(bi, ghost);
    }
    if (pieces.length > 1 && S.ui.onion && cand.length) {
      const used = new Set([...link.values()].map((g) => g.id));
      pieces.forEach((s, i) => {
        if (link.has(i)) return;
        const id = nearestGhost(piecePts(s), cand.filter((g) => !used.has(g.id)));
        if (id) { const g = cand.find((x) => x.id === id); link.set(i, g); used.add(id); }
      });
    }
    pushUndo();
    let next = Math.max(-1, ...k.strokes.map((x) => x.order), ...(prevKey(f)?.strokes || []).map((x) => x.order)) + 1;
    pieces.forEach((s, i) => {
      const g = link.get(i);
      if (g) { s.id = g.id; s.order = g.order; } else { s.id = newId(); s.order = next++; }
      k.strokes.push(s);
    });
    S.ui.sel = new Set(pieces.map((s) => s.id));
    const linked = [...link.values()].map((g) => g.id);
    if (pieces.length > 1) {
      toast(`Too complex for one robot stroke: split into ${pieces.length} strokes` +
            (linked.length ? `, ${linked.length} traced onto ghosts (${linked.join(", ")}).` : ".") +
            (worst > 1.5 ? ` Still off by up to ${worst.toFixed(1)} mm; try shorter marks.` : ""));
    } else if (ghost) toast(`Traced: replaces ghost ${ghost.id}. Wrong one? Use "Pick another ghost" in Properties.`);
    else if (worst > 1.5) toast(`This mark is complex; the robot stroke follows it within ${worst.toFixed(1)} mm. Try shorter marks.`);
    changed();
  } catch (e) {
    toast("Could not fit stroke: " + e.message);
  } finally { pendingFit = null; renderAll(); }
}

function pickGhost(x, y) {
  const id = S.ui.picking; S.ui.picking = null;
  const f = S.ui.frame, k = keyAt(f), p = prevKey(f);
  const g = p && hitStroke(x, y, p.strokes);
  if (!k || !g) { toast("No ghost there. Linking unchanged."); renderAll(); return; }
  pushUndo();
  const other = k.strokes.find((s) => s.id === g.id);
  const mine = k.strokes.find((s) => s.id === id);
  if (other && other !== mine) other.id = newId();
  mine.id = g.id; mine.order = g.order;
  S.ui.sel = new Set([g.id]);
  toast(`Now replaces ghost ${g.id}.`);
  changed(); renderAll();
}

// ------------------------------------------------------------- commands ---
const cmds = {
  new() { if (!confirm("Start a new document? Unsaved changes are lost.")) return; resetDoc(newDoc()); },
  open() { $("fileinput").click(); },
  save() {
    download((S.doc.editor.name || "animation") + ".json", JSON.stringify(S.doc, null, 1));
    toast("Saved keyframes (open again with File > Open, or use with inbetween.cli).");
  },
  exportFrames() {
    const ks = keys();
    if (ks.length < 2 || !Object.keys(S.gen.frames).length) return toast("Create in-betweens first.");
    if (S.gen.stale) return toast("In-betweens are out of date. Create In-betweens again before exporting.");
    const frames = [];
    for (let f = ks[0].frame; f <= ks[ks.length - 1].frame; f++) {
      const c = content(f), e = editAt(f);
      frames.push({ frame: f, kind: c.mode === "key" ? "key" : "inbetween",
                    locked: c.mode === "key" ? c.strokes.map((s) => s.id) : (e ? e.strokes.map((s) => s.id) : []),
                    strokes: [...c.strokes].sort((a, b) => a.order - b.order || a.id.localeCompare(b.id)) });
    }
    const out = { format: "frida-inbetween-frames", version: 1, method: S.gen.method, cache_dir: S.cfg.cache_dir, frames };
    download(`frames_${S.gen.method}.json`, JSON.stringify(out, null, 1));
    toast("Exported. Paint with: python -m inbetween.paint_frames --frames <file> …");
  },
  undo() { if (!S.hist.undo.length) return; S.hist.redo.push(JSON.stringify(S.doc)); S.doc = JSON.parse(S.hist.undo.pop()); afterHistory(); },
  redo() { if (!S.hist.redo.length) return; S.hist.undo.push(JSON.stringify(S.doc)); S.doc = JSON.parse(S.hist.redo.pop()); afterHistory(); },
  selectAll() { S.ui.sel = new Set(content(S.ui.frame).strokes.map((s) => s.id)); renderAll(); },
  copy() {
    const sel = selected();
    if (!sel.length) return toast("Select strokes to copy.");
    clip = clone(sel); toast(`Copied ${sel.length} stroke${sel.length > 1 ? "s" : ""}.`);
  },
  cut() {
    const sel = selected();
    if (!sel.length) return toast("Select strokes to cut.");
    if (content(S.ui.frame).mode !== "key") return toast("Cut strokes on a keyframe; in-betweens follow the keyframes.");
    clip = clone(sel); cmds.delete();
  },
  paste() { pasteStrokes(clip); },
  pasteInPlace() {
    const ids = pasteStrokes(clip, { inPlace: true });
    if (ids && ids.some((id) => clip.find((s) => s.id === id)))
      toast("Pasted in place. Strokes kept their identity, so in-betweens animate them from the copied keyframe.");
  },
  duplicate() {
    const sel = selected();
    if (!sel.length) return toast("Select strokes to duplicate.");
    pasteStrokes(sel); // the clipboard is left as it was, as in Animate
  },
  deselect() { S.ui.sel.clear(); S.ui.picking = null; S.ui.linkTarget = null; renderAll(); },
  delete() {
    const f = S.ui.frame, c = content(f);
    if (!S.ui.sel.size) return;
    if (c.mode === "between") {
      const e = editAt(f);
      if (e && [...S.ui.sel].some((id) => e.strokes.find((s) => s.id === id))) {
        pushUndo(); e.strokes = e.strokes.filter((s) => !S.ui.sel.has(s.id));
        if (!e.strokes.length) S.doc.edits = S.doc.edits.filter((x) => x !== e);
        toast("Unpinned: these strokes follow the in-betweening again."); S.ui.sel.clear(); changed(); renderAll(); return;
      }
      return toast("Delete strokes on a keyframe; in-betweens follow the keyframes.");
    }
    if (c.mode !== "key") return canEdit();
    pushUndo(); const k = keyAt(f); k.strokes = k.strokes.filter((s) => !S.ui.sel.has(s.id)); S.ui.sel.clear(); changed(); renderAll();
  },
  zoomIn() { zoomAt(1.25); }, zoomOut() { zoomAt(0.8); }, zoomFit() { zoomFit(); render(); },
  toggleOnion() { S.ui.onion = !S.ui.onion; renderAll(); },
  togglePreview() { S.ui.preview = !S.ui.preview; renderAll(); },
  insertFrame() { // F5: add a frame here, pushing later frames right (adjusts timing)
    pushUndo(); const f = S.ui.frame;
    for (const k of keys()) if (k.frame > f) k.frame++;
    for (const e of S.doc.edits) if (e.frame > f) e.frame++;
    S.doc.timing = shiftTiming(S.doc.timing, f, +1);
    S.doc.editor.length++; S.gen.frames = {}; S.gen.stale = true; renderAll();
  },
  removeFrame() { // Shift+F5
    const f = S.ui.frame;
    if (keyAt(f)) return toast("This frame is a keyframe. Clear it first (Shift+F6).");
    if (S.doc.editor.length <= 1) return;
    pushUndo();
    S.doc.edits = S.doc.edits.filter((e) => e.frame !== f);
    for (const k of keys()) if (k.frame > f) k.frame--;
    for (const e of S.doc.edits) if (e.frame > f) e.frame--;
    S.doc.timing = shiftTiming(S.doc.timing, f, -1);
    S.doc.editor.length--; S.gen.frames = {}; S.gen.stale = true; S.ui.frame = Math.min(f, lastFrame()); renderAll();
  },
  insertKeyframe() { // F6: copy what is on this frame (in-between included), ids preserved
    const f = S.ui.frame; if (keyAt(f)) return toast("Already a keyframe.");
    pushUndo(); const c = content(f);
    S.doc.keys.push({ frame: f, strokes: clone(c.strokes) });
    S.doc.edits = S.doc.edits.filter((e) => e.frame !== f);
    sortKeys(); grow(f); changed(); renderAll();
    toast("Keyframe: reshape strokes with the Selection (V) and Subselection (A) tools.");
  },
  insertBlankKeyframe() { // F7: empty keyframe, trace over the onion skin
    const f = S.ui.frame; if (keyAt(f)) return toast("Already a keyframe.");
    pushUndo(); S.doc.keys.push({ frame: f, strokes: [] });
    S.doc.edits = S.doc.edits.filter((e) => e.frame !== f);
    sortKeys(); grow(f); S.ui.onion = true; S.ui.tool = "brush"; changed(); renderAll();
    toast("Blank keyframe: redraw strokes over their ghosts. Each traced stroke replaces its ghost.");
  },
  clearKeyframe() {
    const f = S.ui.frame, k = keyAt(f);
    if (!k) return; if (keys().length <= 1) return toast("Keep at least one keyframe.");
    pushUndo(); S.doc.keys = keys().filter((x) => x !== k); changed(); renderAll();
  },
  generate, play: () => (S.ui.playing ? stop() : play()),
  prevFrame() { goto(S.ui.frame - 1); }, nextFrame() { goto(S.ui.frame + 1); },
  prevKey() { const k = prevKey(S.ui.frame); if (k) goto(k.frame); },
  nextKey() { const k = nextKey(S.ui.frame); if (k) goto(k.frame); },
  shortcuts() { showShortcuts(); },
};
function shiftTiming(timing, f, d) {
  const out = {};
  for (const [span, ease] of Object.entries(timing)) {
    let [a, b] = span.split("-").map(Number);
    if (a > f) a += d; if (b > f) b += d;
    out[`${a}-${b}`] = ease;
  }
  return out;
}
function grow(f) { S.doc.editor.length = Math.max(S.doc.editor.length, f + 1); }
function afterHistory() { S.ui.sel.clear(); markStale(); S.ui.frame = Math.min(S.ui.frame, lastFrame()); renderAll(); }
function resetDoc(doc) {
  S.doc = doc; S.gen = { frames: {}, stale: true, method: null, metrics: null };
  S.hist = { undo: [], redo: [] }; S.ui.sel.clear(); S.ui.frame = keys()[0]?.frame || 0;
  $("tl-fps").value = S.doc.editor.fps; renderAll();
}
function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$("fileinput").addEventListener("change", async (e) => {
  const file = e.target.files[0]; e.target.value = "";
  if (!file) return;
  try {
    const d = JSON.parse(await file.text());
    if (d.format !== "frida-inbetween") throw new Error("not a keyframe file");
    d.edits = d.edits || []; d.timing = d.timing || {};
    let maxId = 0;
    for (const k of d.keys) for (const s of k.strokes) { const m = /^s(\d+)$/.exec(s.id); if (m) maxId = Math.max(maxId, +m[1]); }
    d.editor = Object.assign({ fps: 8, length: 1, next_id: maxId + 1, name: file.name.replace(/\.json$/, "") }, d.editor || {});
    d.editor.next_id = Math.max(d.editor.next_id, maxId + 1);
    d.editor.length = Math.max(d.editor.length, ...d.keys.map((k) => k.frame + 1));
    d.keys.sort((a, b) => a.frame - b.frame);
    resetDoc(d); toast(`Opened ${file.name}`);
  } catch (err) { toast("Could not open: " + err.message); }
});

async function generate() {
  const ks = keys();
  if (ks.length < 2) return toast("Add a second keyframe first (F6 or F7).");
  if (S.ui.busy) return;
  const method = $("tl-method").value;
  const doc = clone(S.doc);
  doc.edits = doc.edits.filter((e) => e.frame > ks[0].frame && e.frame < ks[ks.length - 1].frame && !keyAt(e.frame) && e.strokes.length);
  S.ui.busy = true; $("tl-generate").disabled = true;
  status(method === "refined" ? "Creating in-betweens and refining for the painted look…" : "Creating in-betweens…");
  try {
    const res = await api("/api/inbetween", { doc, method });
    const frames = {};
    for (const fr of res.frames) if (fr.kind !== "key") frames[fr.frame] = fr;
    S.gen = { frames, stale: false, method, metrics: res.metrics };
    status(`${res.frames.length} frames · ${method} · width deviation max ${res.metrics.width_dev_max_mm.toFixed(2)} mm, mean ${res.metrics.width_dev_mean_mm.toFixed(2)} mm`);
  } catch (e) { status(""); toast("Could not create in-betweens: " + e.message); }
  finally { S.ui.busy = false; $("tl-generate").disabled = false; renderAll(); }
}

// ------------------------------------------------------------ playback ---
let playTimer = null;
function goto(f) { S.ui.frame = clamp(f, 0, lastFrame()); S.ui.sel.clear(); S.ui.picking = null; renderAll(); scrollTimeline(); }
function play() {
  S.ui.playing = true; $("tl-play").textContent = "■";
  const first = keys()[0]?.frame || 0;
  playTimer = setInterval(() => {
    const next = S.ui.frame + 1 > lastFrame() ? first : S.ui.frame + 1;
    S.ui.frame = next; render(); renderTimeline(); $("tl-frame").textContent = next + 1;
  }, 1000 / clamp(+S.doc.editor.fps || 8, 1, 60));
}
function stop() { S.ui.playing = false; clearInterval(playTimer); $("tl-play").textContent = "▶"; renderAll(); }

// ------------------------------------------------------------ timeline ---
const tl = $("tlcanvas"), tctx = tl.getContext("2d");
const CW = 14, HEAD = 22, ROW = 26;
function renderTimeline() {
  const n = Math.max(lastFrame() + 1, 0), cols = Math.max(n + 40, Math.ceil($("tlframes").clientWidth / CW) + 1);
  const d = devicePixelRatio || 1;
  tl.width = cols * CW * d; tl.height = (HEAD + ROW + 2) * d; tl.style.width = cols * CW + "px"; tl.style.height = HEAD + ROW + 2 + "px";
  tctx.setTransform(d, 0, 0, d, 0, 0);
  tctx.fillStyle = "#282828"; tctx.fillRect(0, 0, cols * CW, HEAD + ROW + 2);
  tctx.font = "10px sans-serif"; tctx.textAlign = "center";
  for (let i = 0; i < cols; i++) {
    const x = i * CW;
    if ((i + 1) % 5 === 0 || i === 0) { tctx.fillStyle = "#9a9a9a"; tctx.fillText(String(i + 1), x + CW / 2, 14); }
    tctx.fillStyle = i < n ? ((i + 1) % 5 === 0 ? "#3d3d3d" : "#353535") : "#2b2b2b";
    tctx.fillRect(x, HEAD, CW, ROW);
    tctx.strokeStyle = "#232323"; tctx.beginPath(); tctx.moveTo(x + 0.5, HEAD); tctx.lineTo(x + 0.5, HEAD + ROW); tctx.stroke();
  }
  // in-between spans
  const ks = keys();
  for (let j = 0; j < ks.length - 1; j++) {
    const a = ks[j].frame, b = ks[j + 1].frame;
    const has = Object.keys(S.gen.frames).some((f) => f > a && f < b);
    if (b - a < 2) continue;
    if (has) {
      tctx.fillStyle = S.gen.stale ? "#3e3a52" : "#4b4270"; tctx.fillRect(a * CW + 1, HEAD + 1, (b - a) * CW - 1, ROW - 2);
      tctx.strokeStyle = "#a99be8"; tctx.setLineDash(S.gen.stale ? [3, 3] : []);
      const y = HEAD + ROW / 2;
      tctx.beginPath(); tctx.moveTo(a * CW + CW / 2 + 4, y); tctx.lineTo(b * CW + CW / 2 - 6, y); tctx.stroke();
      tctx.setLineDash([]);
      if (!S.gen.stale) { tctx.beginPath(); tctx.moveTo(b * CW + CW / 2 - 6, y - 3); tctx.lineTo(b * CW + CW / 2 - 3, y); tctx.lineTo(b * CW + CW / 2 - 6, y + 3); tctx.stroke(); }
    }
  }
  // keyframes
  for (const k of ks) {
    const x = k.frame * CW + CW / 2, y = HEAD + ROW / 2;
    tctx.beginPath(); tctx.arc(x, y, 3.6, 0, 7);
    if (k.strokes.length) { tctx.fillStyle = "#e9e9e9"; tctx.fill(); }
    else { tctx.strokeStyle = "#e9e9e9"; tctx.lineWidth = 1.2; tctx.stroke(); tctx.lineWidth = 1; }
  }
  // end-of-sequence marker
  if (n > 0) { tctx.strokeStyle = "#cfcfcf"; tctx.strokeRect((n - 1) * CW + 3.5, HEAD + ROW - 11.5, CW - 7, 7); }
  // artist pins on in-betweens
  tctx.fillStyle = "#f0a030";
  for (const e of S.doc.edits) if (e.strokes.length) { const x = e.frame * CW + CW / 2, y = HEAD + ROW - 5; tctx.beginPath(); tctx.moveTo(x, y - 3); tctx.lineTo(x + 3, y); tctx.lineTo(x, y + 3); tctx.lineTo(x - 3, y); tctx.fill(); }
  // playhead
  const px = S.ui.frame * CW;
  tctx.fillStyle = "rgba(232,71,60,.25)"; tctx.fillRect(px, HEAD, CW, ROW);
  tctx.fillStyle = "#e8473c"; tctx.fillRect(px, 3, CW, 16); tctx.fillRect(px + CW / 2 - 0.5, HEAD, 1, ROW);
  tctx.fillStyle = "#fff"; tctx.font = "bold 9px sans-serif"; tctx.fillText(String(S.ui.frame + 1), px + CW / 2, 14);
}
function frameAt(e) { return Math.max(0, Math.floor(e.offsetX / CW)); }
let tlDrag = false;
tl.addEventListener("pointerdown", (e) => { if (e.button !== 0) return; tlDrag = true; tl.setPointerCapture(e.pointerId); if (S.ui.playing) stop(); goto(frameAt(e)); });
tl.addEventListener("pointermove", (e) => { if (tlDrag) { const f = frameAt(e); if (f !== S.ui.frame) goto(f); } });
tl.addEventListener("pointerup", () => { tlDrag = false; });
tl.addEventListener("contextmenu", (e) => {
  e.preventDefault(); goto(frameAt(e));
  const f = S.ui.frame, k = keyAt(f);
  const items = [
    ["Insert Frame", "insertFrame", "F5"], ["Remove Frame", "removeFrame", "Shift+F5", !!k],
    null,
    ["Insert Keyframe", "insertKeyframe", "F6", !!k], ["Insert Blank Keyframe", "insertBlankKeyframe", "F7", !!k],
    ["Clear Keyframe", "clearKeyframe", "Shift+F6", !k],
    null,
    ["Create In-betweens", "generate", "Ctrl+Enter", keys().length < 2],
  ];
  showContext(e.clientX, e.clientY, items);
});
function scrollTimeline() {
  const box = $("tlframes"), x = S.ui.frame * CW;
  if (x < box.scrollLeft || x > box.scrollLeft + box.clientWidth - CW) box.scrollLeft = x - box.clientWidth / 2;
}

// ------------------------------------------------------------ menus ---
document.querySelectorAll(".menu").forEach((m) => {
  m.addEventListener("mousedown", (e) => {
    if (e.target.closest(".dropdown")) return;
    const open = m.classList.contains("open");
    closeMenus(); if (!open) m.classList.add("open");
  });
  m.addEventListener("mouseenter", () => { if (document.querySelector(".menu.open") && !m.classList.contains("open")) { closeMenus(); m.classList.add("open"); } });
});
document.querySelectorAll("[data-cmd]").forEach((b) => b.addEventListener("click", () => { closeMenus(); cmds[b.dataset.cmd](); }));
document.addEventListener("mousedown", (e) => { if (!e.target.closest(".menu") && !e.target.closest("#ctxmenu")) closeMenus(); });
function closeMenus() { document.querySelectorAll(".menu.open").forEach((m) => m.classList.remove("open")); $("ctxmenu").classList.remove("show"); }
function showContext(x, y, items) {
  const m = $("ctxmenu"); m.innerHTML = "";
  for (const it of items) {
    if (!it) { m.appendChild(document.createElement("hr")); continue; }
    const b = document.createElement("button"); b.innerHTML = `${it[0]}<kbd>${it[2]}</kbd>`; b.disabled = !!it[3];
    b.onclick = () => { closeMenus(); cmds[it[1]](); }; m.appendChild(b);
  }
  m.style.left = x + "px"; m.style.top = y + "px"; m.classList.add("show");
}

// ------------------------------------------------------------ keyboard ---
const TOOLKEYS = { v: "selection", a: "subselection", b: "brush", h: "hand", z: "zoom" };
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea")) return;
  const k = e.key, mod = e.ctrlKey || e.metaKey;
  const run = (c) => { e.preventDefault(); cmds[c](); };
  if (mod) {
    if (k === "z" || k === "Z") return run(e.shiftKey ? "redo" : "undo");
    if (k === "y") return run("redo");
    if (k === "s") return run("save");
    if (k === "o") return run("open");
    if (k === "n") return run("new");
    if (k === "a") return run("selectAll");
    if (k === "c") return run("copy");
    if (k === "x") return run("cut");
    if (k === "v" || k === "V") return run(e.shiftKey ? "pasteInPlace" : "paste");
    if (k === "d") return run("duplicate");
    if (k === "Enter") return run("generate");
    if (k === "=" || k === "+") return run("zoomIn");
    if (k === "-") return run("zoomOut");
    if (k === "0") return run("zoomFit");
    return;
  }
  if (k === "F5") return run(e.shiftKey ? "removeFrame" : "insertFrame");
  if (k === "F6") return run(e.shiftKey ? "clearKeyframe" : "insertKeyframe");
  if (k === "F7") return run("insertBlankKeyframe");
  if (k === "Enter") return run("play");
  if (k === "," || k === "<") return run(e.shiftKey ? "prevKey" : "prevFrame");
  if (k === "." || k === ">") return run(e.shiftKey ? "nextKey" : "nextFrame");
  if (k === "Delete" || k === "Backspace") return run("delete");
  if (k === "Escape") return run("deselect");
  if (k === "o" || k === "O") return run("toggleOnion");
  if (k === "p" || k === "P") return run("togglePreview");
  if (k === "?") return run("shortcuts");
  if (k === " ") { e.preventDefault(); if (!S.ui.spaceHand) { S.ui.spaceHand = true; cv.style.cursor = "grab"; } return; }
  const t = TOOLKEYS[k.toLowerCase()];
  if (t) { setTool(t); e.preventDefault(); }
});
document.addEventListener("keyup", (e) => { if (e.key === " ") S.ui.spaceHand = false; });
function setTool(t) { S.ui.tool = t; S.ui.picking = null; renderAll(); }
document.querySelectorAll(".tool").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));

// ------------------------------------------------------------ properties ---
function renderProps() {
  const body = $("propbody"), f = S.ui.frame, c = content(f);
  const sel = c.strokes.filter((s) => S.ui.sel.has(s.id));
  let h = "";
  if (sel.length === 1) h = strokeProps(sel[0], c);
  else if (sel.length > 1) h = `<div class="psec"><h4>Selection</h4><div class="pinfo"><b>${sel.length}</b> strokes selected. Drag to move together, or use the round handle to rotate.</div>
    <button class="pbtn" data-act="delete">Delete</button></div>`;
  else h = frameProps(c);
  body.innerHTML = h;
  bindProps(sel[0], c);
}
function slider(name, label, v, min, max, step, fmt) {
  return `<div class="prow"><label>${label}</label><input type="range" data-p="${name}" min="${min}" max="${max}" step="${step}" value="${v}"><span class="val" data-v="${name}">${fmt(v)}</span></div>`;
}
function strokeProps(s, c) {
  const pct = (v) => Math.round(v * 100) + "%";
  const mm = (v) => (v * 1000).toFixed(0);
  const W = S.cfg.canvas_width_m, H = S.cfg.canvas_height_m;
  const vis = ((S.cfg.visible_z_min - 0.02) / 0.96) * 100;
  let state = "";
  if (c.mode === "key") state = `On keyframe ${f1()}.`;
  else if (c.mode === "between") state = c.pinned.has(s.id)
    ? `<span class="warn">Pinned on this in-between</span>: the in-betweening keeps your change. <button class="pbtn" data-act="unpin">Unpin</button>`
    : `Generated in-between. Changing it pins it for this frame.`;
  else state = "Held from the previous keyframe (read-only).";
  let link = "";
  const pk = prevKey(S.ui.frame);
  if (c.mode === "key" && pk) {
    const g = pk.strokes.find((x) => x.id === s.id);
    link = `<div class="psec"><h4>Correspondence</h4><div class="pinfo">${g
      ? `In-betweens move this stroke from <b>${s.id}</b> in keyframe ${pk.frame + 1}.`
      : `New stroke: it appears halfway between keyframe ${pk.frame + 1} and this one.`}</div>
      <button class="pbtn ${S.ui.picking ? "on" : ""}" data-act="pick">Pick another ghost</button>
      ${g ? `<button class="pbtn" data-act="unlink">Make it a new stroke</button>` : ""}</div>`;
  }
  return `<div class="psec"><h4>Brush Stroke</h4><div class="pinfo">ID <b>${s.id}</b> · paint order ${s.order}<br>${state}</div></div>
  <div class="psec"><h4>Pressure along the stroke</h4>
    ${["Start", "⅓", "⅔", "End"].map((l, i) => slider("z" + i, l, s.z[i], 0.02, 0.98, 0.01, pct)).join("")}
    <div class="pinfo">Below ${pct(S.cfg.visible_z_min)} this ${S.cfg.ink ? "marker" : "brush"} leaves no mark (dashed outline on stage).</div></div>
  <div class="psec"><h4>Shape</h4>
    ${slider("k0", "Corner 1 round", s.kappa[0], 0, 1, 0.01, pct)}${slider("k1", "Corner 2 round", s.kappa[1], 0, 1, 0.01, pct)}
    <div class="pinfo">Drag corner points with the Subselection tool (A).</div></div>
  <div class="psec"><h4>Position and Rotation</h4>
    ${slider("x", "X (mm)", s.xt, 0, 1, 0.001, (v) => mm(v * W))}${slider("y", "Y (mm)", s.yt, 0, 1, 0.001, (v) => mm(v * H))}
    ${slider("a", "Rotation", s.a, -Math.PI, Math.PI, 0.01, (v) => Math.round((-v * 180) / Math.PI) + "°")}</div>
  ${S.cfg.ink ? "" : `<div class="psec"><h4>Color</h4><div class="prow two"><label>Paint</label><input type="color" data-p="color" value="${rgbToHex(s.color)}"></div></div>`}
  ${link}`;
}
function f1() { return S.ui.frame + 1; }
function frameProps(c) {
  const f = S.ui.frame, pk = prevKey(f), nk = nextKey(f), k = keyAt(f);
  let h = "";
  if (k) {
    const missing = traceCandidates(f);
    h += `<div class="psec"><h4>Keyframe ${f + 1}</h4><div class="pinfo"><b>${k.strokes.length}</b> strokes.</div>
      ${missing.length ? `<div class="pinfo warn">${missing.length} stroke${missing.length > 1 ? "s" : ""} from keyframe ${pk.frame + 1} ${missing.length > 1 ? "are" : "is"} not in this keyframe yet, so ${missing.length > 1 ? "they disappear" : "it disappears"} halfway there. Trace ${missing.length > 1 ? "them" : "it"} with the Brush, or:</div>
      <button class="pbtn" data-act="carry">Bring back unchanged</button>` : ""}</div>`;
  } else if (pk && nk) {
    const span = `${pk.frame}-${nk.frame}`, ease = S.doc.timing[span] || "linear";
    h += `<div class="psec"><h4>In-betweens ${pk.frame + 1}–${nk.frame + 1}</h4>
      <div class="prow two"><label>Ease</label><select data-p="ease">${EASES.map(([v, l]) => `<option value="${v}" ${v === ease ? "selected" : ""}>${l}</option>`).join("")}</select></div>
      <div class="pinfo">${nk.frame - pk.frame - 1} in-between frames. Add or remove frames (F5 / Shift+F5) to change timing.</div>
      <div class="pinfo">${S.gen.frames[f] ? (S.gen.stale ? `<span class="warn">Out of date.</span> Create In-betweens again.` : `Created (${S.gen.method}). Adjust a stroke here to pin it.`) : "Not created yet: Create In-betweens (Ctrl+Enter)."}</div></div>`;
  } else {
    h += `<div class="psec"><h4>Frame ${f + 1}</h4><div class="pinfo">${pk ? `Holds keyframe ${pk.frame + 1}.` : "Empty."} Press F6 to copy the drawing into a new keyframe, or F7 for a blank keyframe to trace over.</div></div>`;
  }
  h += `<div class="psec"><h4>Document</h4>
    <div class="pinfo">Paper <b>${(S.cfg.canvas_width_m * 1000).toFixed(0)} × ${(S.cfg.canvas_height_m * 1000).toFixed(0)} mm</b><br>
    Stroke model <b>${S.cfg.cache_dir.split("/").pop()}</b> ${S.cfg.calibrated ? "(calibrated on real strokes)" : '<span class="warn">(not calibrated)</span>'}<br>
    ${S.cfg.response !== "model" ? '<span class="warn">Using an assumed pressure curve, not real data.</span><br>' : ""}
    ${S.cfg.ink ? "Ink / marker setup" : "Paint setup"}</div>
    ${S.cfg.ink ? "" : `<div class="prow two"><label>Brush color</label><input type="color" data-p="brushcolor" value="${S.ui.color}"></div>`}</div>
  <div class="psec"><h4>Workflow</h4><div class="pinfo">
    <b>F6</b> Keyframe: copy the drawing and reshape strokes (V, A).<br>
    <b>F7</b> Blank Keyframe: redraw strokes over the onion skin; each traced stroke replaces its ghost (Alt+click a ghost to choose it).<br>
    <b>Ctrl+Enter</b> Create In-betweens · <b>Enter</b> Play · <b>?</b> all shortcuts</div></div>`;
  return h;
}
function bindProps(s, c) {
  const body = $("propbody");
  body.querySelectorAll("input[type=range]").forEach((inp) => {
    let started = false;
    inp.addEventListener("input", () => {
      if (!canEdit()) { renderProps(); return; }
      if (!started) { pushUndo(); started = true; }
      const v = +inp.value, p = inp.dataset.p;
      mutateStroke(s.id, (x) => {
        if (p[0] === "z") x.z[+p[1]] = v; else if (p[0] === "k") x.kappa[+p[1]] = v;
        else if (p === "x") x.xt = v; else if (p === "y") x.yt = v; else if (p === "a") x.a = v;
      });
      const lab = body.querySelector(`[data-v="${p}"]`);
      if (lab) {
        const W = S.cfg.canvas_width_m, H = S.cfg.canvas_height_m;
        lab.textContent = p[0] === "z" || p[0] === "k" ? Math.round(v * 100) + "%" : p === "x" ? (v * W * 1000).toFixed(0) : p === "y" ? (v * H * 1000).toFixed(0) : Math.round((-v * 180) / Math.PI) + "°";
      }
      markStale(); render(); renderTimeline();
    });
    inp.addEventListener("change", () => { started = false; renderAll(); });
  });
  const col = body.querySelector('[data-p="color"]');
  if (col) col.addEventListener("change", () => { if (!canEdit()) return; pushUndo(); mutateStroke(s.id, (x) => { x.color = hexToRgb(col.value); }); changed(); renderAll(); });
  const bc = body.querySelector('[data-p="brushcolor"]');
  if (bc) bc.addEventListener("change", () => { S.ui.color = bc.value; });
  const ease = body.querySelector('[data-p="ease"]');
  if (ease) ease.addEventListener("change", () => {
    const pk = prevKey(S.ui.frame), nk = nextKey(S.ui.frame);
    pushUndo(); S.doc.timing[`${pk.frame}-${nk.frame}`] = ease.value; changed(); renderAll();
  });
  body.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => {
    const act = b.dataset.act, f = S.ui.frame;
    if (act === "delete") cmds.delete();
    if (act === "pick") { S.ui.picking = s.id; toast("Click the ghost stroke this stroke should replace."); renderAll(); }
    if (act === "unlink") { pushUndo(); const st = keyAt(f).strokes.find((x) => x.id === s.id); st.id = newId(); S.ui.sel = new Set([st.id]); changed(); renderAll(); }
    if (act === "unpin") { pushUndo(); const e = editAt(f); e.strokes = e.strokes.filter((x) => x.id !== s.id); if (!e.strokes.length) S.doc.edits = S.doc.edits.filter((x) => x !== e); changed(); renderAll(); }
    if (act === "carry") { pushUndo(); keyAt(f).strokes.push(...clone(traceCandidates(f))); changed(); renderAll(); }
  }));
}

// ------------------------------------------------------------ misc UI ---
function updateModeHint() {
  const f = S.ui.frame, c = content(f);
  let t = "";
  if (S.ui.picking) t = "Click the ghost stroke to link to";
  else if (c.mode === "key" && traceCandidates(f).length && S.ui.onion) t = "Keyframe · tracing over onion skin";
  else if (c.mode === "key") t = "Keyframe";
  else if (c.mode === "between") t = S.gen.stale ? "In-between (out of date)" : "In-between · adjust a stroke to pin it";
  else t = isBetween(f) ? "Not created yet · Ctrl+Enter" : "Held frame · F6 / F7 to add a keyframe";
  $("modehint").textContent = t;
}
let toastTimer = null;
function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), 3200); }
function status(msg) { $("tl-status").textContent = msg; }
function showShortcuts() {
  const rows = [["V", "Selection tool: click or drag a rectangle to select (Shift adds), drag to move; round handle rotates"], ["A", "Subselection tool: drag corner points"], ["B", "Brush tool (stylus pressure sets width)"],
    ["Alt+click ghost", "Brush: choose which ghost the next stroke traces"], ["H / Space", "Hand tool / temporary pan"], ["Z", "Zoom tool (Alt+click zooms out)"],
    ["F5 / Shift+F5", "Insert / remove frame (timing)"], ["F6", "Insert keyframe (copy the drawing)"], ["F7", "Insert blank keyframe (trace over onion skin)"],
    ["Shift+F6", "Clear keyframe"], ["Ctrl+Enter", "Create in-betweens"], ["Enter", "Play / stop"], [", / .", "Previous / next frame"],
    ["Shift+, / Shift+.", "Previous / next keyframe"], ["O", "Onion skin"], ["P", "Paint preview (stroke model)"], ["Del", "Delete stroke, or unpin on an in-between"],
    ["Ctrl+Z / Ctrl+Shift+Z", "Undo / redo"], ["Ctrl+C / Ctrl+X / Ctrl+V", "Copy / cut / paste (pasted strokes are new)"],
    ["Ctrl+Shift+V", "Paste in place (keeps stroke identity if the keyframe doesn't have it)"], ["Ctrl+D", "Duplicate"], ["Ctrl+S / Ctrl+O", "Save / open keyframes"], ["Ctrl+= / Ctrl+- / Ctrl+0", "Zoom in / out / fit"]];
  $("modalbody").innerHTML = `<h3>Keyboard Shortcuts</h3><table>${rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("")}</table>`;
  $("modal").hidden = false;
}
$("modalclose").onclick = () => { $("modal").hidden = true; };

function renderAll() {
  document.querySelectorAll(".tool").forEach((b) => b.classList.toggle("active", b.dataset.tool === S.ui.tool));
  $("tl-onion").classList.toggle("on", S.ui.onion);
  $("tl-preview").classList.toggle("on", S.ui.preview);
  $("tl-frame").textContent = S.ui.frame + 1;
  $("docname").textContent = (S.doc.editor.name || "Untitled") + (S.gen.stale && Object.keys(S.gen.frames).length ? " · in-betweens out of date" : "");
  render(); renderTimeline(); renderProps();
}

// timeline bar
$("tl-play").onclick = () => cmds.play();
$("tl-prev").onclick = () => cmds.prevFrame();
$("tl-next").onclick = () => cmds.nextFrame();
$("tl-prevkey").onclick = () => cmds.prevKey();
$("tl-nextkey").onclick = () => cmds.nextKey();
$("tl-onion").onclick = () => cmds.toggleOnion();
$("tl-preview").onclick = () => cmds.togglePreview();
$("tl-generate").onclick = () => generate();
$("tl-fps").onchange = (e) => { S.doc.editor.fps = clamp(+e.target.value || 8, 1, 60); if (S.ui.playing) { stop(); play(); } };
$("zoomsel").onchange = (e) => { if (e.target.value) zoomTo(e.target.value); };
window.addEventListener("resize", resizeStage);

// ------------------------------------------------------------ boot ---
(async function boot() {
  try {
    S.cfg = await api("/api/config");
  } catch (e) {
    document.body.innerHTML = `<p style="padding:30px">Could not reach the editor server. Start it from src/ with <code>python -m inbetween.app</code>.<br>${e.message}</p>`;
    return;
  }
  S.doc = newDoc();
  resizeStage();
  renderAll();
  status(`Stroke model: ${S.cfg.cache_dir}`);
})();
