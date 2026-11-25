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
    suggest: null, spaceHand: false, busy: false, linkFirst: null,
  },
  view: { scale: 1, ox: 0, oy: 0, fit: true },
  hist: { undo: [], redo: [] },
  previewCache: new Map(),
  // Reference layer: the animator's own frames (PNG sequence or video), shown under the
  // drawing for tracing. Never painted, exported or saved in the keyframe file.
  ref: { images: [], name: "", visible: true, opacity: 0.35, offset: 0 },
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
    format: "frida-inbetween", version: 3, cache_dir: S.cfg.cache_dir,
    keys: [{ frame: 0, strokes: [] }], edits: [], timing: {}, paths: [], trims: [],
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
  drawReference(f);
  // onion skin ghosts
  const candidates = new Set(traceCandidates(f).map((s) => s.id));
  const links = linkState(f); // on a keyframe with onion skin: what is not linked to the previous key
  for (const g of ghostSets(f)) {
    for (const s of g.key.strokes) {
      const target = S.ui.linkTarget === s.id || (S.ui.suggest === s.id);
      const lonely = links && g.key === links.pk && links.lonely.has(s.id);
      drawStroke(s, { color: target ? "rgb(255,150,40)" : lonely ? WARN : `rgb(${g.rgb})`,
                      alpha: target ? 0.8 : lonely ? 0.55 : (candidates.has(s.id) ? 0.45 : 0.22), minWidth: 1.5 });
    }
  }
  // drawing or paint preview
  const pv = S.ui.preview && previewFor(c.strokes);
  if (pv) ctx.drawImage(pv, S.view.ox, S.view.oy, paperW(), paperH());
  else for (const s of c.strokes) drawStroke(s, links && links.unlinked.has(s.id) ? { color: WARN } : {});
  if (S.ui.tool === "link" && links) drawLinks(links);
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
  drawMotionPaths(f);
  // selection rectangle
  if (drag && drag.kind === "marquee") {
    const x = Math.min(drag.x0, drag.x1), y = Math.min(drag.y0, drag.y1);
    const w = Math.abs(drag.x1 - drag.x0), h = Math.abs(drag.y1 - drag.y0);
    ctx.save(); ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.strokeRect(x + 0.5, y + 0.5, w, h);
    ctx.setLineDash([4, 3]); ctx.strokeStyle = "#222"; ctx.strokeRect(x + 0.5, y + 0.5, w, h);
    ctx.restore();
  }
  drawBez();
  // live brush
  if (drag && drag.kind === "brush") drawLive(drag.pts);
  if (drag && drag.kind === "mpath") {
    ctx.save(); ctx.strokeStyle = "#c440d6"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); drag.pts.forEach((pt, i) => { const [x, y] = toScreen(pt); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.restore();
  }
  if (pendingFit) drawLive(pendingFit, true);
  updateModeHint();
}
// --------------------------------------------------------- reference ---
const refImage = (f) => S.ref.images[f - S.ref.offset] || null;
// Fit the frame inside the paper, centered: the same placement inbetween.import_frames uses,
// so imported strokes line up with their reference frames.
function drawReference(f) {
  const im = S.ref.visible && refImage(f);
  if (!im) return;
  const s = Math.min(paperW() / im.width, paperH() / im.height);
  const w = im.width * s, h = im.height * s;
  ctx.save(); ctx.globalAlpha = S.ref.opacity;
  ctx.beginPath(); ctx.rect(S.view.ox, S.view.oy, paperW(), paperH()); ctx.clip();
  ctx.drawImage(im, S.view.ox + (paperW() - w) / 2, S.view.oy + (paperH() - h) / 2, w, h);
  ctx.restore();
}
const MAX_REF_W = 1600; // keep memory reasonable for long sequences
async function bitmap(src) {
  const w0 = src.videoWidth || src.naturalWidth || src.width, h0 = src.videoHeight || src.naturalHeight || src.height;
  const k = Math.min(1, MAX_REF_W / w0);
  return createImageBitmap(src, { resizeWidth: Math.round(w0 * k), resizeHeight: Math.round(h0 * k), resizeQuality: "high" });
}
async function imagesFromFiles(files) {
  const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const out = [];
  for (const [i, file] of sorted.entries()) {
    status(`Loading reference ${i + 1} / ${sorted.length}…`);
    out.push(await bitmap(await createImageBitmap(file)));
  }
  return out;
}
async function imagesFromVideo(file, fps) {
  const v = document.createElement("video");
  v.muted = true; v.preload = "auto"; v.src = URL.createObjectURL(file);
  await new Promise((ok, bad) => { v.onloadeddata = ok; v.onerror = () => bad(new Error("this browser can't decode that video; export it as PNG frames instead")); });
  const n = Math.max(1, Math.round(v.duration * fps));
  const out = [];
  for (let i = 0; i < n; i++) {
    status(`Reading video frame ${i + 1} / ${n}…`);
    await new Promise((ok) => { v.onseeked = ok; v.currentTime = Math.min((i + 0.5) / fps, v.duration - 1e-3); });
    out.push(await bitmap(v));
  }
  URL.revokeObjectURL(v.src);
  return out;
}
async function importReference(files) {
  if (!files.length) return;
  const video = [...files].find((f) => f.type.startsWith("video/"));
  try {
    S.ui.busy = true;
    let images;
    if (video) {
      const ans = prompt("Frames per second of this video? (One drawing per timeline frame.)", "24");
      if (ans === null) return;
      images = await imagesFromVideo(video, clamp(+ans || 24, 1, 120));
    } else {
      images = await imagesFromFiles([...files].filter((f) => f.type.startsWith("image/")));
    }
    if (!images.length) throw new Error("no image or video files selected");
    S.ref.images.forEach((b) => b.close && b.close());
    Object.assign(S.ref, { images, name: video ? video.name : `${images.length} images`, visible: true, offset: 0 });
    grow(images.length - 1);
    toast(`Reference: ${images.length} frames on frames 1–${images.length}. It is never painted or exported.`);
  } catch (err) {
    toast("Could not load reference: " + err.message);
  } finally { S.ui.busy = false; status(""); renderAll(); }
}

// ------------------------------------------------------------- linking ---
// A stroke on a keyframe is linked when the previous keyframe has a stroke with its id
// (that is what in-betweening moves it from). Link mode (L): click a stroke, then its
// ghost; each link is one undo step. Unlinked strokes and ghosts without a partner are
// drawn in WARN while onion skin is on. Strokes set to Draw on are new on purpose.
const WARN = "rgb(226,62,87)";
function linkState(f, always = false) { // null unless onion skin shows the ghosts (or always)
  const k = keyAt(f), pk = prevKey(f);
  if (!k || !pk || (!always && (!S.ui.onion || S.ui.preview))) return null;
  const span = `${pk.frame}-${f}`;
  const prevIds = new Set(pk.strokes.map((s) => s.id)), mine = new Set(k.strokes.map((s) => s.id));
  const intended = new Set(k.strokes.filter((s) => trimMode(span, s.id) === "draw_on").map((s) => s.id));
  const unlinked = new Set(k.strokes.filter((s) => !prevIds.has(s.id) && !intended.has(s.id)).map((s) => s.id));
  const lonely = new Set(pk.strokes.filter((s) => !mine.has(s.id)).map((s) => s.id));
  const m = k.strokes.length - intended.size;
  return { k, pk, unlinked, lonely, linked: m - unlinked.size, m, drawOn: intended.size };
}
function linkSummary(f) {
  const k = keyAt(f), pk = prevKey(f);
  if (!k || !pk) return "";
  const st = linkState(f, true);
  const done = st.linked === st.m && !st.lonely.size;
  return `<div class="psec"><h4>Links to keyframe ${pk.frame + 1}</h4>
    <div class="pinfo ${done ? "" : "warn"}" data-links><b>${st.linked} of ${st.m}</b> strokes linked${st.drawOn ? ` (+${st.drawOn} drawing on)` : ""};
    ${st.lonely.size} ghost${st.lonely.size === 1 ? "" : "s"} without a partner.</div>
    <div class="pinfo">${S.ui.tool === "link" ? (S.ui.linkFirst ? "Now click the ghost it continues." : "Click a stroke, then its ghost. Esc leaves Link mode.")
      : `Unlinked strokes and ghosts show in red (onion skin on). <b>L</b> links them: click a stroke, then its ghost.`}</div></div>`;
}
function linkClick(x, y) {
  const f = S.ui.frame, k = keyAt(f), pk = prevKey(f);
  if (!k || !pk) return toast("Link mode works on a keyframe that has a keyframe before it.");
  if (S.ui.linkFirst && k.strokes.some((s) => s.id === S.ui.linkFirst)) {
    const g = hitStroke(x, y, pk.strokes);
    if (g) {
      pushUndo();
      const mine = k.strokes.find((s) => s.id === S.ui.linkFirst), other = k.strokes.find((s) => s.id === g.id);
      if (other && other !== mine) other.id = newId(); // that stroke loses the link
      mine.id = g.id; mine.order = g.order;
      S.ui.linkFirst = null; S.ui.sel.clear();
      const st = linkState(f);
      toast(other && other !== mine ? `Linked to ghost ${g.id}; the stroke that had it is now unlinked.`
                                    : `Linked${st ? `: ${st.linked} of ${st.m}` : ""}.`);
      changed(); renderAll(); return;
    }
  }
  const hit = hitStroke(x, y, k.strokes); // (re)choose the stroke
  S.ui.linkFirst = hit ? hit.id : null;
  S.ui.sel = new Set(hit ? [hit.id] : []);
  if (!hit) toast("Click a stroke on this keyframe, then the ghost it continues.");
  renderAll();
}
function drawLinks(st) { // thin lines from each linked stroke to its ghost
  ctx.save(); ctx.strokeStyle = "rgba(40,120,220,.55)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  for (const s of st.k.strokes) {
    const g = st.pk.strokes.find((x) => x.id === s.id); if (!g) continue;
    const a = toScreen(center(s)), b = toScreen(center(g));
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  ctx.restore();
}

// --------------------------------------------------------- motion paths ---
// The animator draws the path a group of strokes travels between two neighboring keys
// (Motion Path tool, M). Stored in S.doc.paths as {span: "a-b", ids, points: [[x, y], ...]}.
// fitPath / pointAt / pathMembers mirror inbetween/motion_path.py, so the dots drawn on the
// path are where Create In-betweens puts the group's anchor on each frame.
const MP_MIN_CHORD = 0.01, MP_SCALE_LIMIT = 4;
const EASE_FN = { linear: (t) => t, ease_in: (t) => t * t, ease_out: (t) => 1 - (1 - t) * (1 - t),
                  ease_in_out: (t) => 3 * t * t - 2 * t * t * t };
const spanOf = (p) => p.span.split("-").map(Number);
const easedT = (a, b, f) => (EASE_FN[S.doc.timing[`${a}-${b}`]] || EASE_FN.linear)((f - a) / (b - a));
function groupAnchor(strokes) {
  const cs = strokes.map(center);
  return [cs.reduce((t, c) => t + c[0], 0) / cs.length, cs.reduce((t, c) => t + c[1], 0) / cs.length];
}
function cumLen(q) { const L = [0]; for (let i = 1; i < q.length; i++) L.push(L[i - 1] + Math.hypot(q[i][0] - q[i - 1][0], q[i][1] - q[i - 1][1])); return L; }
function fitPath(points, a0, a1) {
  const k = aspect(), P = points.map((p) => [p[0] * k, p[1]]), A0 = [a0[0] * k, a0[1]], A1 = [a1[0] * k, a1[1]];
  const n = P.length - 1, cd = [P[n][0] - P[0][0], P[n][1] - P[0][1]], ca = [A1[0] - A0[0], A1[1] - A0[1]];
  const ld = Math.hypot(...cd), la = Math.hypot(...ca);
  let out;
  if (ld >= MP_MIN_CHORD && la >= MP_MIN_CHORD && la / ld >= 1 / MP_SCALE_LIMIT && la / ld <= MP_SCALE_LIMIT) {
    // rotation + uniform scale taking the drawn chord onto the keys' chord (complex ca / cd)
    const den = cd[0] * cd[0] + cd[1] * cd[1];
    const re = (ca[0] * cd[0] + ca[1] * cd[1]) / den, im = (ca[1] * cd[0] - ca[0] * cd[1]) / den;
    out = P.map(([x, y]) => { const dx = x - P[0][0], dy = y - P[0][1]; return [A0[0] + dx * re - dy * im, A0[1] + dx * im + dy * re]; });
  } else { // shift, blending the two end offsets along the path
    const L = cumLen(P), tot = L[n] || 1, d0 = [A0[0] - P[0][0], A0[1] - P[0][1]], d1 = [A1[0] - P[n][0], A1[1] - P[n][1]];
    out = P.map(([x, y], i) => [x + d0[0] + (d1[0] - d0[0]) * L[i] / tot, y + d0[1] + (d1[1] - d0[1]) * L[i] / tot]);
  }
  out[0] = A0; out[n] = A1;
  return out.map(([x, y]) => [x / k, y]);
}
function pathLocate(q, u) { // segment index and blend weight at fraction u of the length on paper
  const k = aspect(), L = cumLen(q.map((p) => [p[0] * k, p[1]])), n = q.length - 1;
  if (L[n] <= 0) return [1, 0];
  const s = clamp(u, 0, 1) * L[n];
  for (let i = 1; i <= n; i++) if (L[i] >= s) { const seg = L[i] - L[i - 1]; return [i, seg > 0 ? (s - L[i - 1]) / seg : 0]; }
  return [n, 1];
}
function pointAt(q, u) {
  const [i, w] = pathLocate(q, u);
  return [q[i - 1][0] + (q[i][0] - q[i - 1][0]) * w, q[i - 1][1] + (q[i][1] - q[i - 1][1]) * w];
}
function splitPath(q, u) { const [i] = pathLocate(q, u), pt = pointAt(q, u); return [[...q.slice(0, i), pt], [pt, ...q.slice(i)]]; }
// The path's two keyframes, if they exist and are neighbors.
function pathKeys(p) {
  const [a, b] = spanOf(p), ka = keyAt(a), kb = keyAt(b);
  return ka && kb && nextKey(a) === kb ? [ka, kb] : null;
}
const inKey = (k, id) => k.strokes.some((s) => s.id === id);
// Strokes that follow the path: in both keys and not pinned anywhere in the span (a pinned
// stroke keeps its pin, and leaving it out on every frame keeps the anchor steady).
function pathMembers(p) {
  const ks = pathKeys(p); if (!ks) return [];
  const [a, b] = spanOf(p);
  const pinned = new Set(S.doc.edits.filter((e) => e.frame > a && e.frame < b).flatMap((e) => e.strokes.map((s) => s.id)));
  return p.ids.filter((id) => inKey(ks[0], id) && inKey(ks[1], id) && !pinned.has(id));
}
function fittedPath(p) {
  const ks = pathKeys(p), ids = pathMembers(p);
  if (!ks || !ids.length) return null;
  const pick = (k) => k.strokes.filter((s) => ids.includes(s.id));
  return fitPath(p.points, groupAnchor(pick(ks[0])), groupAnchor(pick(ks[1])));
}
const offPaper = (q) => q.some(([x, y]) => x < 0 || x > 1 || y < 0 || y > 1);
const pathsAt = (f) => S.doc.paths.filter((p) => { const [a, b] = spanOf(p); return a <= f && f <= b; });
// The span a path drawn on frame f belongs to: from this keyframe to the next (or from the
// previous one, on the last keyframe), or the span this in-between is in.
function spanHere(f) {
  if (keyAt(f)) { const n = nextKey(f), p = prevKey(f); return n ? [f, n.frame] : p ? [p.frame, f] : null; }
  return isBetween(f) ? [prevKey(f).frame, nextKey(f).frame] : null;
}
function commitPath(pts) {
  const sp = spanHere(S.ui.frame);
  if (!sp) return toast("A motion path goes from one keyframe to the next. Add a second keyframe first (F6 / F7).");
  const [a, b] = sp, ka = keyAt(a), kb = keyAt(b), sel = [...S.ui.sel];
  // Strokes in both keys travel along the path; a stroke in only one of them follows the
  // group while it draws on / off (Properties > Appears / Disappears).
  const ids = sel.filter((id) => inKey(ka, id) || inKey(kb, id));
  if (!ids.some((id) => inKey(ka, id) && inKey(kb, id)))
    return toast(`To travel along a path, select strokes that are in both keyframe ${a + 1} and keyframe ${b + 1}.`);
  pushUndo();
  const span = `${a}-${b}`;
  for (const p of S.doc.paths) if (p.span === span) p.ids = p.ids.filter((id) => !ids.includes(id)); // one path per stroke and span
  const path = { span, ids, points: pts.map((p) => [+p[0].toFixed(5), +p[1].toFixed(5)]) };
  S.doc.paths = S.doc.paths.filter((p) => p.ids.length).concat([path]);
  const q = fittedPath(path), left = sel.length - ids.length;
  if (q && offPaper(q)) toast("This path leaves the paper. The robot can't paint there, so the strokes will stop at the edge.");
  else toast(`Motion path: ${ids.length} stroke${ids.length > 1 ? "s" : ""} travel along it from keyframe ${a + 1} to ${b + 1}. Create In-betweens (Ctrl+Enter) to apply it.` +
             (left ? ` ${left} selected stroke${left > 1 ? "s" : ""} not in both keyframes ${left > 1 ? "were" : "was"} left out.` : ""));
  changed(); renderAll();
}
// Timing changes (F5 / Shift+F5) shift span names, as shiftTiming does.
function shiftPaths(f, d) { // motion paths and trims
  for (const p of [...S.doc.paths, ...S.doc.trims]) { let [a, b] = spanOf(p); if (a > f) a += d; if (b > f) b += d; p.span = `${a}-${b}`; }
}
// A new keyframe inside a pathed span splits the path where the anchor had reached on that frame.
function splitPathsAt(f) {
  const out = [];
  for (const p of S.doc.paths) {
    const [a, b] = spanOf(p);
    if (!(a < f && f < b)) { out.push(p); continue; }
    const [q1, q2] = splitPath(fittedPath(p) || p.points, easedT(a, b, f));
    out.push({ span: `${a}-${f}`, ids: [...p.ids], points: q1 }, { span: `${f}-${b}`, ids: [...p.ids], points: q2 });
  }
  S.doc.paths = out;
  // Trims go to both halves; each is used only where the stroke still enters / leaves.
  S.doc.trims = S.doc.trims.flatMap((t) => { const [a, b] = spanOf(t);
    return a < f && f < b ? [{ ...t, span: `${a}-${f}`, ids: [...t.ids] }, { ...t, span: `${f}-${b}`, ids: [...t.ids] }] : [t]; });
}
// A new keyframe at f splits the span it lands in: the path splits where the anchor had
// reached (using the old span's ease), then both new spans keep that ease, as in Animate.
function splitSpanAt(f) {
  splitPathsAt(f);
  const p = prevKey(f), n = nextKey(f);
  if (!p || !n || keyAt(f)) return;
  const old = `${p.frame}-${n.frame}`, ease = S.doc.timing[old];
  if (ease === undefined) return;
  delete S.doc.timing[old];
  S.doc.timing[`${p.frame}-${f}`] = ease; S.doc.timing[`${f}-${n.frame}`] = ease;
}
function dropPathsAt(f) { // a keyframe at either end was cleared (trims there go too)
  const n = S.doc.paths.length;
  S.doc.paths = S.doc.paths.filter((p) => !spanOf(p).includes(f));
  S.doc.trims = S.doc.trims.filter((t) => !spanOf(t).includes(f));
  return n - S.doc.paths.length;
}
function removeFromPaths(f, ids) { // strokes deleted from keyframe f
  for (const p of [...S.doc.paths, ...S.doc.trims]) if (spanOf(p).includes(f)) p.ids = p.ids.filter((id) => !ids.has(id));
  S.doc.paths = S.doc.paths.filter((p) => p.ids.length);
  S.doc.trims = S.doc.trims.filter((t) => t.ids.length);
}
// ------------------------------------------------------------ trims ---
// Draw on / draw off (After Effects' Trim Paths): a stroke that is only in the later key
// of a span draws on over it, one only in the earlier key draws off, instead of popping at
// the middle. Stored in S.doc.trims as {span: "a-b", ids, mode}; see inbetween/trim.py.
const trimMode = (span, id) => (S.doc.trims.find((t) => t.span === span && t.ids.includes(id)) || {}).mode || "";
function setTrim(span, ids, mode) {
  for (const t of S.doc.trims) if (t.span === span) t.ids = t.ids.filter((i) => !ids.includes(i));
  S.doc.trims = S.doc.trims.filter((t) => t.ids.length);
  if (mode) S.doc.trims.push({ span, ids: [...ids], mode });
}
const trimValid = (ka, kb, id, mode) => (mode === "draw_on" ? !inKey(ka, id) && inKey(kb, id) : inKey(ka, id) && !inKey(kb, id));
// Properties: how the selected strokes appear (new since the previous key) or disappear
// (gone in the next key).
function trimProps(ids) {
  const f = S.ui.frame, k = keyAt(f); if (!k) return "";
  const pk = prevKey(f), nk = nextKey(f), out = [];
  const row = (label, span, sel, mode, other) => {
    const cur = sel.every((id) => trimMode(span, id) === mode) ? mode : "";
    return `<div class="prow two"><label>${label}</label><select data-trim="${span}" data-mode="${mode}" data-ids="${sel.join(",")}">
      <option value="" ${cur ? "" : "selected"}>${other}</option><option value="${mode}" ${cur ? "selected" : ""}>${mode === "draw_on" ? "Draw on" : "Draw off"}</option></select></div>`;
  };
  const enter = pk ? ids.filter((id) => !inKey(pk, id)) : [];
  if (enter.length) out.push(row("Appears", `${pk.frame}-${f}`, enter, "draw_on", "Pop at midpoint"));
  const leave = nk ? ids.filter((id) => !inKey(nk, id)) : [];
  if (leave.length) out.push(row("Disappears", `${f}-${nk.frame}`, leave, "draw_off", "Pop at midpoint"));
  if (!out.length) return "";
  return `<div class="psec"><h4>Appear / Disappear</h4>${out.join("")}<div class="pinfo">Draw on: the stroke is drawn
    progressively over the in-betweens, as the robot would draw it, and moves with its motion-path group or its
    nearest stroke. Each partial stroke is a real robot stroke.</div></div>`;
}
function drawMotionPaths(f) {
  for (const p of pathsAt(f)) {
    const q = fittedPath(p); if (!q) continue;
    const [a, b] = spanOf(p), on = p.ids.some((id) => S.ui.sel.has(id)) || S.ui.tool === "motion";
    ctx.save(); ctx.strokeStyle = on ? "rgba(196,64,214,.95)" : "rgba(196,64,214,.45)"; ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]);
    ctx.beginPath(); q.forEach((pt, i) => { const [x, y] = toScreen(pt); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    ctx.setLineDash([]); ctx.strokeStyle = "#c440d6";
    for (let g = a; g <= b; g++) { // one dot per frame, as in After Effects: spacing shows the ease
      const [x, y] = toScreen(pointAt(q, easedT(a, b, g)));
      ctx.fillStyle = g === f ? "#c440d6" : "#fff";
      ctx.beginPath(); ctx.arc(x, y, g === f ? 3.6 : (g === a || g === b ? 3 : 2.2), 0, 7); ctx.fill(); ctx.stroke();
    }
    ctx.restore();
  }
}
function motionProps(paths) {
  if (!paths.length) return "";
  return `<div class="psec"><h4>Motion Path</h4>${paths.map((p) => {
    const [a, b] = spanOf(p), off = p.ids.length - pathMembers(p).length;
    return `<div class="pinfo">Keyframe ${a + 1} → ${b + 1}: <b>${p.ids.join(", ")}</b> travel${p.ids.length > 1 ? "" : "s"} along the drawn path.` +
      (off ? ` <span class="warn">${off} pinned or not in both keyframes: ${off > 1 ? "they don't" : "it doesn't"} follow it.</span>` : "") +
      `</div><button class="pbtn" data-act="rmpath" data-idx="${S.doc.paths.indexOf(p)}">Remove path</button>`;
  }).join("")}</div>`;
}

function diamond(x, y, r) { ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.fill(); }

// ------------------------------------------------- corner handles ---
// Each interior corner of a FRIDA stroke is a circular arc with one roundness value, so
// its two tangent handles always mirror each other (as Figma's "mirror angle and length").
// Dragging either along its edge sets how far from the corner the curve starts.
const SHARP_HANDLE_PX = 10; // where the handles sit on a sharp corner, so they can be grabbed
function cornerHandleScreen(s) {
  const toS = (pt) => toScreen(Geo.toCanvas([pt], s, S.cfg)[0]);
  return Geo.cornerHandles(s).filter((h) => !h.flat).map((h) => {
    const P = toS(h.P);
    const end = (u) => {
      if (h.t > 1e-6) return toS([h.P[0] + u[0] * h.t, h.P[1] + u[1] * h.t]);
      const far = toS([h.P[0] + u[0] * h.tMax, h.P[1] + u[1] * h.tMax]); // sharp: a short stub
      const dx = far[0] - P[0], dy = far[1] - P[1], L = Math.hypot(dx, dy) || 1;
      return [P[0] + dx / L * SHARP_HANDLE_PX, P[1] + dy / L * SHARP_HANDLE_PX];
    };
    return { k: h.i - 1, P, a: end(h.ua), b: end(h.ub), sharp: h.t <= 1e-6, h };
  });
}
// Smallest roundness whose curve starts tDesired from the corner (t only grows with
// roundness, then stops where an edge is used up: the smallest value avoids a dead zone).
function kappaForTangent(s, k, tDesired) {
  const tAt = (kap) => {
    const x = { ...s, kappa: s.kappa.map((v, j) => (j === k ? kap : v)) };
    return Geo.cornerHandles(x)[k].t;
  };
  if (tDesired <= 0) return 0;
  const tTop = tAt(1);
  const goal = Math.min(tDesired, tTop);
  let lo = 0, hi = 1;
  for (let it = 0; it < 30; it++) { const mid = (lo + hi) / 2; if (tAt(mid) < goal - 1e-9) lo = mid; else hi = mid; }
  return hi;
}

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
    // curve handles: where each corner's curve meets its two edges
    for (const c of cornerHandleScreen(s)) {
      const active = drag && drag.kind === "corner" && drag.id === s.id && drag.k === c.k;
      ctx.strokeStyle = active ? "#2d8ceb" : "rgba(45,140,235,.8)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(c.a[0], c.a[1]); ctx.lineTo(c.P[0], c.P[1]); ctx.lineTo(c.b[0], c.b[1]); ctx.stroke();
      for (const q of [c.a, c.b]) {
        ctx.beginPath(); ctx.arc(q[0], q[1], 4.5, 0, Math.PI * 2);
        ctx.fillStyle = c.sharp ? "#fff" : "#2d8ceb"; ctx.fill(); ctx.strokeStyle = "#2d8ceb"; ctx.stroke();
      }
    }
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
// Lightest touch = lightest clearly visible line (matches fit.py's pressure_to_z).
const pressureToZ = (p) => (S.cfg.line_z_min ?? S.cfg.visible_z_min) + (0.95 - (S.cfg.line_z_min ?? S.cfg.visible_z_min)) * clamp(p, 0, 1);

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
  if (t === "pen" || t === "bezsel") { bezPointerDown(x, y, e, t === "pen" ? "pen" : "select"); return; }

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

  if (t === "link") { linkClick(x, y); return; }
  if (t === "motion") { // select strokes (click, Shift adds), then drag the path they travel
    const hit = hitStroke(x, y, c.strokes);
    if (!S.ui.sel.size || e.shiftKey) {
      if (hit) { if (e.shiftKey && S.ui.sel.has(hit.id)) S.ui.sel.delete(hit.id); else S.ui.sel.add(hit.id); }
      else if (!e.shiftKey) toast("Motion Path: click the strokes to move (Shift adds), then drag the path they travel to the next keyframe.");
      renderAll(); return;
    }
    drag = { kind: "mpath", pts: [toProp(x, y)], x0: x, y0: y }; render(); return;
  }

  // selection / subselection
  if (t === "subselection" && S.ui.sel.size) {
    for (const s of c.strokes) if (S.ui.sel.has(s.id)) {
      for (const h of cornerHandleScreen(s)) {
        for (const [q, u] of [[h.a, "ua"], [h.b, "ub"]]) {
          if (Math.hypot(q[0] - x, q[1] - y) < 7) {
            if (!canEdit()) return;
            pushUndo(); drag = { kind: "corner", id: s.id, k: h.k, side: u }; render(); return;
          }
        }
      }
    }
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
    // empty spot: drag a dashed rectangle to select every stroke it touches
    // (Shift+drag adds to the selection), as with Animate's Selection tool
    if (!e.shiftKey) S.ui.sel.clear();
    drag = { kind: "marquee", x0: x, y0: y, x1: x, y1: y, base: new Set(S.ui.sel) };
  }
  renderAll();
});

cv.addEventListener("pointermove", (e) => {
  const x = e.offsetX, y = e.offsetY;
  if (!drag) {
    cv.style.cursor = { hand: "grab", zoom: "zoom-in", brush: "crosshair", motion: "crosshair", link: "pointer", pen: "crosshair", bezsel: "crosshair" }[tool()] || "default";
    if (S.ui.bez) { S.ui.bez.hover = toProp(x, y); render(); }
    return;
  }
  if (drag.kind === "bezanchor") { bezDrag(x, y, e); return; }
  if (drag.kind === "pan") { S.view.ox = drag.ox + x - drag.x; S.view.oy = drag.oy + y - drag.y; S.view.fit = false; render(); return; }
  if (drag.kind === "brush") {
    addBrushPoints(e);
    const cand = traceCandidates(S.ui.frame);
    S.ui.suggest = S.ui.linkTarget ? null : (S.ui.onion && cand.length ? nearestGhost(drag.pts, cand) : null);
    render(); return;
  }
  if (drag.kind === "mpath") {
    const last = toScreen(drag.pts[drag.pts.length - 1]);
    if (Math.hypot(x - last[0], y - last[1]) > 2) { drag.pts.push(toProp(x, y)); render(); }
    return;
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
  if (drag.kind === "corner") {
    const s0 = content(S.ui.frame).strokes.find((s) => s.id === drag.id);
    const h = Geo.cornerHandles(s0)[drag.k];
    const m = Geo.toLocal(toProp(x, y), s0, S.cfg), u = h[drag.side];
    const along = (m[0] - h.P[0]) * u[0] + (m[1] - h.P[1]) * u[1];
    const kap = kappaForTangent(s0, drag.k, Math.max(0, along));
    mutateStroke(drag.id, (s) => { s.kappa[drag.k] = Math.round(kap * 1000) / 1000; });
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

cv.addEventListener("pointerup", async (e) => {
  const d = drag; drag = null;
  if (!d) return;
  if (d.kind === "brush") await finishBrush(d.pts);
  if (d.kind === "mpath") {
    if (d.pts.length >= 4) commitPath(d.pts); // points are >2 px apart, so this is a drag, not a click
    else { const hit = hitStroke(d.x0, d.y0, content(S.ui.frame).strokes); S.ui.sel = new Set(hit ? [hit.id] : []); } // a click selects
  }
  renderAll();
});

cv.addEventListener("dblclick", (e) => {
  if (S.ui.bez && (tool() === "pen" || tool() === "bezsel")) { e.preventDefault(); bezFinish(false); }
});

cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.ctrlKey || e.metaKey) zoomAt(Math.exp(-e.deltaY * 0.002), e.offsetX, e.offsetY);
  else { S.view.ox -= e.deltaX; S.view.oy -= e.deltaY; S.view.fit = false; render(); }
}, { passive: false });

// Any part of a stroke inside the rectangle selects it (as in Animate's Lasso/Selection
// with "contact" selection): a point inside, or the stroke crossing the rectangle's edge.
function strokesInRect(strokes, r) {
  const x0 = Math.min(r.x0, r.x1), x1 = Math.max(r.x0, r.x1), y0 = Math.min(r.y0, r.y1), y1 = Math.max(r.y0, r.y1);
  if (x1 - x0 < 3 && y1 - y0 < 3) return [];
  return strokesInPolygon(strokes, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
}
// Screen-space polygon test shared by the rectangle and the Bezier selection.
function pointInPolygon(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function segmentsCross(a, b, c, d) {
  const o = (p, q, r) => Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
  return o(a, b, c) !== o(a, b, d) && o(c, d, a) !== o(c, d, b);
}
function strokesInPolygon(strokes, poly) {
  if (poly.length < 3) return [];
  return strokes.filter((s) => {
    const pts = strokeScreenPath(s);
    if (pts.some((p) => pointInPolygon(p, poly))) return true;
    for (let i = 0; i + 1 < pts.length; i++)
      for (let j = 0, k = poly.length - 1; j < poly.length; k = j++)
        if (segmentsCross(pts[i], pts[i + 1], poly[k], poly[j])) return true;
    return false;
  }).map((s) => s.id);
}
// ---------------------------------------------- Bezier pen and selection ---
// Pen (N) draws a precise line that is fitted to robot strokes like a brush mark;
// Bezier Selection (Q) outlines an area and selects every stroke that touches it.
// Click = sharp anchor; click-drag = smooth anchor with mirrored handles (Alt while
// dragging breaks the mirror); click the first anchor to close; Enter or double-click
// finishes; Backspace removes the last anchor; Esc cancels. Anchors are kept in canvas
// proportions so zooming or panning mid-path is safe.
const BEZ_CLOSE_PX = 8;
function bezSegments(b, closed) {
  const A = b.anchors, segs = [];
  const n = closed ? A.length : A.length - 1;
  for (let i = 0; i < n; i++) {
    const a = A[i], c = A[(i + 1) % A.length];
    segs.push([a.p, a.hout || a.p, c.hin || c.p, c.p]);
  }
  return segs;
}
function cubicAt(s, u) {
  const v = 1 - u, w = [v * v * v, 3 * v * v * u, 3 * v * u * u, u * u * u];
  return [0, 1].map((d) => w[0] * s[0][d] + w[1] * s[1][d] + w[2] * s[2][d] + w[3] * s[3][d]);
}
// Sample the path densely (about every 1.5 px on screen) in canvas proportions.
function bezSample(b, closed) {
  const out = [];
  for (const s of bezSegments(b, closed)) {
    const L = [0, 1, 2].reduce((acc, i) => acc + Math.hypot((s[i + 1][0] - s[i][0]) * paperW(), (s[i + 1][1] - s[i][1]) * paperH()), 0);
    const n = Math.max(2, Math.ceil(L / 1.5));
    for (let i = out.length ? 1 : 0; i <= n; i++) out.push(cubicAt(s, i / n));
  }
  return out;
}
function bezStart(mode, additive) {
  S.ui.bez = { mode, frame: S.ui.frame, anchors: [], hover: null,
               base: additive ? new Set(S.ui.sel) : new Set() };
}
function bezCancel() { if (S.ui.bez) { S.ui.bez = null; render(); } }
async function bezFinish(closed) {
  const b = S.ui.bez; S.ui.bez = null;
  if (!b || b.frame !== S.ui.frame) { render(); return; }
  // a double-click adds anchors on top of the last one: drop them
  while (b.anchors.length > 1) {
    const a = b.anchors[b.anchors.length - 1], c = b.anchors[b.anchors.length - 2];
    if (Math.hypot((a.p[0] - c.p[0]) * paperW(), (a.p[1] - c.p[1]) * paperH()) < 3 && !a.hout) b.anchors.pop(); else break;
  }
  if (b.mode === "select") {
    if (b.anchors.length < 3) { toast("Bezier Selection: place at least 3 points to outline an area."); renderAll(); return; }
    const poly = bezSample(b, true).map(toScreen);
    S.ui.sel = new Set([...b.base, ...strokesInPolygon(content(S.ui.frame).strokes, poly)]);
    renderAll(); return;
  }
  if (b.anchors.length < 2) { toast("Pen: place at least 2 points."); renderAll(); return; }
  const pts = bezSample(b, closed);
  if (closed) pts.push(pts[0]);
  await finishBrush(pts.map((p) => [p[0], p[1], 0.5]));
  renderAll();
}
function bezPointerDown(x, y, e, mode) {
  const c = content(S.ui.frame);
  if (mode === "pen" && c.mode !== "key") {
    toast(c.mode === "between" ? "Draw on keyframes. On in-betweens, adjust strokes with the Selection tools."
                               : "Insert a keyframe first (F6 copies the drawing, F7 starts blank).");
    return;
  }
  if (S.ui.bez && (S.ui.bez.frame !== S.ui.frame || S.ui.bez.mode !== mode)) S.ui.bez = null;
  if (!S.ui.bez) bezStart(mode, e.shiftKey);
  const b = S.ui.bez, p = toProp(x, y);
  if (b.anchors.length >= 2) { // click on the first anchor closes the path
    const f0 = toScreen(b.anchors[0].p);
    if (Math.hypot(f0[0] - x, f0[1] - y) < BEZ_CLOSE_PX) { bezFinish(true); return; }
  }
  b.anchors.push({ p, hin: null, hout: null });
  drag = { kind: "bezanchor", x0: x, y0: y };
  render();
}
function bezDrag(x, y, e) {
  const b = S.ui.bez; if (!b) return;
  const a = b.anchors[b.anchors.length - 1];
  if (Math.hypot(x - drag.x0, y - drag.y0) < 3) { a.hin = a.hout = null; render(); return; }
  const q = toProp(x, y);
  a.hout = q;
  if (!e.altKey || !a.hin) a.hin = [2 * a.p[0] - q[0], 2 * a.p[1] - q[1]]; // mirrored unless Alt
  render();
}
function drawBez() {
  const b = S.ui.bez; if (!b || b.frame !== S.ui.frame || !b.anchors.length) return;
  const sel = b.mode === "select", col = sel ? "#222" : "#2d8ceb";
  const path = (closedPreview) => {
    const segs = bezSegments(b, false);
    ctx.beginPath();
    const p0 = toScreen(b.anchors[0].p); ctx.moveTo(p0[0], p0[1]);
    for (const s of segs) { const [a, c1, c2, d] = s.map(toScreen); ctx.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], d[0], d[1]); }
    const last = b.anchors[b.anchors.length - 1];
    if (b.hover && !drag) { // rubber band to the pointer
      const c1 = toScreen(last.hout || last.p), h = toScreen(b.hover);
      ctx.bezierCurveTo(c1[0], c1[1], h[0], h[1], h[0], h[1]);
    }
    if (closedPreview && sel) ctx.lineTo(p0[0], p0[1]);
  };
  ctx.save();
  if (sel) {
    ctx.lineWidth = 1; path(true);
    ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.stroke();
    ctx.setLineDash([4, 3]); ctx.strokeStyle = "#222"; ctx.stroke(); ctx.setLineDash([]);
  } else {
    ctx.lineWidth = 1.5; ctx.strokeStyle = col; path(false); ctx.stroke();
  }
  // anchors and handles
  ctx.lineWidth = 1;
  b.anchors.forEach((a, i) => {
    const p = toScreen(a.p);
    for (const h of [a.hin, a.hout]) if (h) {
      const q = toScreen(h);
      ctx.strokeStyle = "rgba(45,140,235,.8)"; ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
      ctx.fillStyle = "#2d8ceb"; ctx.beginPath(); ctx.arc(q[0], q[1], 3.5, 0, Math.PI * 2); ctx.fill();
    }
    const near = i === 0 && b.hover && b.anchors.length >= 2 &&
      Math.hypot(toScreen(b.hover)[0] - p[0], toScreen(b.hover)[1] - p[1]) < BEZ_CLOSE_PX;
    const r = near ? 6 : 4;
    ctx.fillStyle = i === b.anchors.length - 1 ? "#2d8ceb" : "#fff"; ctx.strokeStyle = "#2d8ceb";
    ctx.fillRect(p[0] - r, p[1] - r, 2 * r, 2 * r); ctx.strokeRect(p[0] - r, p[1] - r, 2 * r, 2 * r);
  });
  ctx.restore();
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
    const out = framesForExport();
    if (!out) return;
    download(`frames_${S.gen.method}.json`, JSON.stringify(out, null, 1));
    toast("Exported. Paint with: python -m inbetween.paint_frames --frames <file> …");
  },
  async exportPlans() {
    const out = framesForExport();
    if (!out || S.ui.busy) return;
    S.ui.busy = true; status(`Building FRIDA plans for ${out.frames.length} frames…`);
    try {
      const res = await fetch("/api/export_plans", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ frames: out }),
      });
      if (!res.ok) throw new Error((await res.json()).error || res.statusText);
      downloadBlob(`frida_plans_${S.gen.method}.zip`, await res.blob());
      toast(`Exported ${out.frames.length} FRIDA plans (frame_NNN.pkl + preview PNGs). To paint the animation exactly, use paint_frames with frames.json from the zip.`);
    } catch (e) {
      toast("Could not export FRIDA plans: " + e.message);
    } finally { S.ui.busy = false; status(""); renderAll(); }
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
  leaveLink() { S.ui.linkFirst = null; S.ui.sel.clear(); setTool("selection"); },
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
    pushUndo(); const k = keyAt(f); k.strokes = k.strokes.filter((s) => !S.ui.sel.has(s.id));
    removeFromPaths(f, S.ui.sel); S.ui.sel.clear(); changed(); renderAll();
  },
  zoomIn() { zoomAt(1.25); }, zoomOut() { zoomAt(0.8); }, zoomFit() { zoomFit(); render(); },
  toggleOnion() { S.ui.onion = !S.ui.onion; renderAll(); },
  importReference() { $("refinput").click(); },
  toggleReference() {
    if (!S.ref.images.length) return cmds.importReference();
    S.ref.visible = !S.ref.visible; renderAll();
  },
  removeReference() {
    S.ref.images.forEach((b) => b.close && b.close());
    Object.assign(S.ref, { images: [], name: "", offset: 0 }); renderAll(); toast("Reference removed.");
  },
  togglePreview() { S.ui.preview = !S.ui.preview; renderAll(); },
  insertFrame() { // F5: add a frame here, pushing later frames right (adjusts timing)
    pushUndo(); const f = S.ui.frame;
    for (const k of keys()) if (k.frame > f) k.frame++;
    for (const e of S.doc.edits) if (e.frame > f) e.frame++;
    S.doc.timing = shiftTiming(S.doc.timing, f, +1); shiftPaths(f, +1);
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
    S.doc.timing = shiftTiming(S.doc.timing, f, -1); shiftPaths(f, -1);
    S.doc.editor.length--; S.gen.frames = {}; S.gen.stale = true; S.ui.frame = Math.min(f, lastFrame()); renderAll();
  },
  insertKeyframe() { // F6: copy what is on this frame (in-between included), ids preserved
    const f = S.ui.frame; if (keyAt(f)) return toast("Already a keyframe.");
    pushUndo(); const c = content(f);
    splitSpanAt(f);
    S.doc.keys.push({ frame: f, strokes: clone(c.strokes) });
    S.doc.edits = S.doc.edits.filter((e) => e.frame !== f);
    sortKeys(); grow(f); changed(); renderAll();
    toast("Keyframe: reshape strokes with the Selection (V) and Subselection (A) tools.");
  },
  insertBlankKeyframe() { // F7: empty keyframe, trace over the onion skin
    const f = S.ui.frame; if (keyAt(f)) return toast("Already a keyframe.");
    pushUndo(); splitSpanAt(f); S.doc.keys.push({ frame: f, strokes: [] });
    S.doc.edits = S.doc.edits.filter((e) => e.frame !== f);
    sortKeys(); grow(f); S.ui.onion = true; S.ui.tool = "brush"; changed(); renderAll();
    toast("Blank keyframe: redraw strokes over their ghosts. Each traced stroke replaces its ghost.");
  },
  clearKeyframe() {
    const f = S.ui.frame, k = keyAt(f);
    if (!k) return; if (keys().length <= 1) return toast("Keep at least one keyframe.");
    pushUndo(); S.doc.keys = keys().filter((x) => x !== k);
    const n = dropPathsAt(f);
    if (n) toast(`Removed ${n} motion path${n > 1 ? "s" : ""} that started or ended at this keyframe.`);
    changed(); renderAll();
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
function afterHistory() { S.ui.sel.clear(); S.ui.linkFirst = null; markStale(); S.ui.frame = Math.min(S.ui.frame, lastFrame()); renderAll(); }
function resetDoc(doc) {
  S.doc = doc; S.gen = { frames: {}, stale: true, method: null, metrics: null };
  S.hist = { undo: [], redo: [] }; S.ui.sel.clear(); S.ui.frame = keys()[0]?.frame || 0;
  $("tl-fps").value = S.doc.editor.fps; renderAll();
}
function download(name, text) { downloadBlob(name, new Blob([text], { type: "application/json" })); }
function downloadBlob(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
// Every frame from the first to the last keyframe, as the robot will paint it.
function framesForExport() {
  const ks = keys();
  if (ks.length < 2 || !Object.keys(S.gen.frames).length) { toast("Create in-betweens first."); return null; }
  if (S.gen.stale) { toast("In-betweens are out of date. Create In-betweens again before exporting."); return null; }
  const frames = [];
  for (let f = ks[0].frame; f <= ks[ks.length - 1].frame; f++) {
    const c = content(f), e = editAt(f);
    frames.push({ frame: f, kind: c.mode === "key" ? "key" : "inbetween",
                  locked: c.mode === "key" ? c.strokes.map((s) => s.id) : (e ? e.strokes.map((s) => s.id) : []),
                  strokes: [...c.strokes].sort((a, b) => a.order - b.order || a.id.localeCompare(b.id)) });
  }
  return { format: "frida-inbetween-frames", version: 1, method: S.gen.method, cache_dir: S.cfg.cache_dir, frames };
}
$("fileinput").addEventListener("change", (e) => {
  const file = e.target.files[0]; e.target.value = "";
  if (file) openDocFile(file);
});
async function openDocFile(file) {
  try {
    const d = JSON.parse(await file.text());
    if (d.format !== "frida-inbetween") throw new Error("not a keyframe file");
    d.edits = d.edits || []; d.timing = d.timing || {}; d.paths = d.paths || []; d.trims = d.trims || [];
    d.version = 3; // older files open unchanged and are saved as version 3
    let maxId = 0;
    for (const k of d.keys) for (const s of k.strokes) { const m = /^s(\d+)$/.exec(s.id); if (m) maxId = Math.max(maxId, +m[1]); }
    d.editor = Object.assign({ fps: 8, length: 1, next_id: maxId + 1, name: file.name.replace(/\.json$/, "") }, d.editor || {});
    d.editor.next_id = Math.max(d.editor.next_id, maxId + 1);
    d.editor.length = Math.max(d.editor.length, ...d.keys.map((k) => k.frame + 1));
    d.keys.sort((a, b) => a.frame - b.frame);
    resetDoc(d); toast(`Opened ${file.name}`);
  } catch (err) { toast("Could not open: " + err.message); }
}

// ----------------------------------------------------- drag and drop ---
// Drop the animator's frames (PNGs, a folder of them, or a video) anywhere on the
// window: they become the reference layer. A keyframe file (.json) opens instead.
// Without this, the browser would open a dropped file in the tab and lose the work.
const isRefFile = (f) => f.type.startsWith("image/") || f.type.startsWith("video/");
async function filesFromDrop(dt) {
  const entries = [...(dt.items || [])].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (!entries.length) return [...dt.files];
  const out = [];
  const readDir = (dir) => new Promise((ok) => {
    const r = dir.createReader(), all = [];
    const more = () => r.readEntries((batch) => { if (!batch.length) return ok(all); all.push(...batch); more(); }, () => ok(all));
    more();
  });
  const walk = async (e) => {
    if (e.isFile) out.push(await new Promise((ok, bad) => e.file(ok, bad)));
    else if (e.isDirectory) for (const c of await readDir(e)) await walk(c);
  };
  for (const e of entries) await walk(e);
  return out;
}
async function handleDroppedFiles(files) {
  const json = files.find((f) => f.name.toLowerCase().endsWith(".json"));
  const refs = files.filter(isRefFile);
  if (refs.length) return importReference(refs);
  if (json) return openDocFile(json);
  toast("Drop PNG/JPG frames, a folder of them, or a video for the reference layer, or a .json keyframe file to open.");
}
let dragDepth = 0;
const dragHasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (e) => {
  if (!dragHasFiles(e)) return;
  e.preventDefault(); dragDepth++; $("dropzone").hidden = false;
});
window.addEventListener("dragover", (e) => { if (dragHasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } });
window.addEventListener("dragleave", (e) => {
  if (!dragHasFiles(e)) return;
  if (--dragDepth <= 0) { dragDepth = 0; $("dropzone").hidden = true; }
});
window.addEventListener("drop", async (e) => {
  if (!dragHasFiles(e)) return;
  e.preventDefault(); dragDepth = 0; $("dropzone").hidden = true;
  if (S.ui.busy) return toast("Busy, try again in a moment.");
  handleDroppedFiles(await filesFromDrop(e.dataTransfer));
});

async function generate() {
  const ks = keys();
  if (ks.length < 2) return toast("Add a second keyframe first (F6 or F7).");
  if (S.ui.busy) return;
  const method = $("tl-method").value;
  const doc = clone(S.doc);
  doc.edits = doc.edits.filter((e) => e.frame > ks[0].frame && e.frame < ks[ks.length - 1].frame && !keyAt(e.frame) && e.strokes.length);
  // Send only paths that still join two neighboring keys, with strokes that are in both
  // (strokes not traced yet after F7 rejoin their path once traced).
  doc.trims = doc.trims.map((t) => { const k = pathKeys(t); return k ? { ...t, ids: t.ids.filter((id) => trimValid(k[0], k[1], id, t.mode)) } : null; })
                       .filter((t) => t && t.ids.length);
  const trimmedIn = (span, id) => doc.trims.some((t) => t.span === span && t.ids.includes(id));
  doc.paths = doc.paths.map((p) => { const k = pathKeys(p); return k ? { ...p, ids: p.ids.filter((id) => (inKey(k[0], id) && inKey(k[1], id)) || trimmedIn(p.span, id)) } : null; })
                       .filter((p) => p && p.ids.some((id) => inKey(pathKeys(p)[0], id) && inKey(pathKeys(p)[1], id)));
  S.ui.busy = true; $("tl-generate").disabled = true;
  status(method === "refined" ? "Creating in-betweens and refining for the painted look…" : "Creating in-betweens…");
  try {
    const res = await api("/api/inbetween", { doc, method });
    const frames = {};
    for (const fr of res.frames) if (fr.kind !== "key") frames[fr.frame] = fr;
    S.gen = { frames, stale: false, method, metrics: res.metrics };
    status(`${res.frames.length} frames · ${method} · width deviation max ${res.metrics.width_dev_max_mm.toFixed(2)} mm, mean ${res.metrics.width_dev_mean_mm.toFixed(2)} mm`);
    if (res.warnings && res.warnings.length) toast([...new Set(res.warnings)].join(" "));
  } catch (e) { status(""); toast("Could not create in-betweens: " + e.message); }
  finally { S.ui.busy = false; $("tl-generate").disabled = false; renderAll(); }
}

// ------------------------------------------------------------ playback ---
let playTimer = null;
function goto(f) { S.ui.frame = clamp(f, 0, lastFrame()); S.ui.sel.clear(); S.ui.picking = null; S.ui.bez = null; renderAll(); scrollTimeline(); }
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
  // frames covered by the reference layer: a thin teal line along the top of the row
  if (S.ref.images.length) {
    tctx.fillStyle = S.ref.visible ? "#3fa7a0" : "#2f5e5b";
    tctx.fillRect(S.ref.offset * CW, HEAD, S.ref.images.length * CW, 2);
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
const TOOLKEYS = { v: "selection", a: "subselection", b: "brush", n: "pen", q: "bezsel", m: "motion", l: "link", h: "hand", z: "zoom" };
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
  if (S.ui.bez) { // a Bezier path in progress takes Enter, Esc and Backspace
    if (k === "Enter") { e.preventDefault(); return bezFinish(false); }
    if (k === "Escape") { e.preventDefault(); return bezCancel(); }
    if (k === "Backspace" || k === "Delete") {
      e.preventDefault(); S.ui.bez.anchors.pop(); if (!S.ui.bez.anchors.length) S.ui.bez = null; render(); return;
    }
  }
  if (k === "F5") return run(e.shiftKey ? "removeFrame" : "insertFrame");
  if (k === "F6") return run(e.shiftKey ? "clearKeyframe" : "insertKeyframe");
  if (k === "F7") return run("insertBlankKeyframe");
  if (k === "Enter") return run("play");
  if (k === "," || k === "<") return run(e.shiftKey ? "prevKey" : "prevFrame");
  if (k === "." || k === ">") return run(e.shiftKey ? "nextKey" : "nextFrame");
  if (k === "Delete" || k === "Backspace") return run("delete");
  if (k === "Escape") return run(S.ui.tool === "link" ? "leaveLink" : "deselect");
  if (k === "o" || k === "O") return run("toggleOnion");
  if (k === "p" || k === "P") return run("togglePreview");
  if (k === "r" || k === "R") return run("toggleReference");
  if (k === "?") return run("shortcuts");
  if (k === " ") { e.preventDefault(); if (!S.ui.spaceHand) { S.ui.spaceHand = true; cv.style.cursor = "grab"; } return; }
  const t = TOOLKEYS[k.toLowerCase()];
  if (t) { setTool(t); e.preventDefault(); }
});
document.addEventListener("keyup", (e) => { if (e.key === " ") S.ui.spaceHand = false; });
function setTool(t) {
  // leaving the Pen with a path in progress keeps the line; an unfinished selection outline is dropped
  if (S.ui.bez && S.ui.tool !== t) {
    if (S.ui.bez.mode === "pen" && S.ui.bez.anchors.length >= 2) bezFinish(false); else S.ui.bez = null;
  }
  S.ui.tool = t; S.ui.picking = null; S.ui.linkFirst = null;
  if (t === "link") {
    S.ui.onion = true; S.ui.preview = false; S.ui.sel.clear();
    if (!keyAt(S.ui.frame) || !prevKey(S.ui.frame)) toast("Link mode: go to a keyframe that has a keyframe before it.");
  }
  renderAll();
}
document.querySelectorAll(".tool").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));

// ------------------------------------------------------------ properties ---
function renderProps() {
  const body = $("propbody"), f = S.ui.frame, c = content(f);
  const sel = c.strokes.filter((s) => S.ui.sel.has(s.id));
  let h = "";
  if (sel.length === 1) h = strokeProps(sel[0], c);
  else if (sel.length > 1) h = `<div class="psec"><h4>Selection</h4><div class="pinfo"><b>${sel.length}</b> strokes selected. Drag to move together, or use the round handle to rotate.
    With the Motion Path tool (M), drag the path they travel to the next keyframe.</div>
    <button class="pbtn" data-act="delete">Delete</button></div>` + trimProps(sel.map((s) => s.id)) + motionProps(pathsAt(f).filter((p) => p.ids.some((id) => S.ui.sel.has(id))));
  else h = frameProps(c);
  if (S.ui.tool === "link" && sel.length) h = linkSummary(f) + h; // in Link mode the count comes first
  body.innerHTML = h;
  bindProps(sel[0], c);
  const op = body.querySelector('[data-ref="opacity"]');
  if (op) op.addEventListener("input", () => { S.ref.opacity = +op.value / 100; S.ref.visible = true; render(); $("tl-refop").value = op.value; $("tl-ref").classList.add("on"); });
  const off = body.querySelector('[data-ref="offset"]');
  if (off) off.addEventListener("change", () => { S.ref.offset = Math.max(0, Math.round(+off.value || 1) - 1); grow(S.ref.offset + S.ref.images.length - 1); renderAll(); });
  const box = body.querySelector('[data-act="refimport"]');
  if (box) {
    box.addEventListener("click", (e) => { e.stopImmediatePropagation(); cmds.importReference(); });
    box.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); cmds.importReference(); } });
  }
  const rm = body.querySelector('[data-act="refremove"]');
  if (rm) rm.addEventListener("click", (e) => { e.stopImmediatePropagation(); cmds.removeReference(); });
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
      : trimMode(`${pk.frame}-${S.ui.frame}`, s.id) === "draw_on"
        ? `New stroke: it draws on over the in-betweens from keyframe ${pk.frame + 1} to this one.`
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
  ${motionProps(pathsAt(S.ui.frame).filter((p) => p.ids.includes(s.id)))}
  ${link}${trimProps([s.id])}${c.mode === "key" && S.ui.tool !== "link" ? linkSummary(S.ui.frame) : ""}`;
}
function f1() { return S.ui.frame + 1; }
function frameProps(c) {
  const f = S.ui.frame, pk = prevKey(f), nk = nextKey(f), k = keyAt(f);
  let h = "";
  if (k) {
    const missing = traceCandidates(f);
    h += `<div class="psec"><h4>Keyframe ${f + 1}</h4><div class="pinfo"><b>${k.strokes.length}</b> strokes.</div>
      ${missing.length ? `<div class="pinfo warn">${missing.length} stroke${missing.length > 1 ? "s" : ""} from keyframe ${pk.frame + 1} ${missing.length > 1 ? "are" : "is"} not in this keyframe yet, so ${missing.length > 1 ? "they disappear" : "it disappears"} halfway there. Trace ${missing.length > 1 ? "them" : "it"} with the Brush, or:</div>
      <button class="pbtn" data-act="carry">Bring back unchanged</button>` : ""}</div>` + linkSummary(f) + motionProps(pathsAt(f));
  } else if (pk && nk) {
    const span = `${pk.frame}-${nk.frame}`, ease = S.doc.timing[span] || "linear";
    h += `<div class="psec"><h4>In-betweens ${pk.frame + 1}–${nk.frame + 1}</h4>
      <div class="prow two"><label>Ease</label><select data-p="ease">${EASES.map(([v, l]) => `<option value="${v}" ${v === ease ? "selected" : ""}>${l}</option>`).join("")}</select></div>
      <div class="pinfo">${nk.frame - pk.frame - 1} in-between frames. Add or remove frames (F5 / Shift+F5) to change timing.</div>
      <div class="pinfo">${S.gen.frames[f] ? (S.gen.stale ? `<span class="warn">Out of date.</span> Create In-betweens again.` : `Created (${S.gen.method}). Adjust a stroke here to pin it.`) : "Not created yet: Create In-betweens (Ctrl+Enter)."}</div></div>` + motionProps(pathsAt(f));
  } else {
    h += `<div class="psec"><h4>Frame ${f + 1}</h4><div class="pinfo">${pk ? `Holds keyframe ${pk.frame + 1}.` : "Empty."} Press F6 to copy the drawing into a new keyframe, or F7 for a blank keyframe to trace over.</div></div>`;
  }
  if (S.ref.images.length) {
    const i = f - S.ref.offset, n = S.ref.images.length;
    h += `<div class="psec"><h4>Reference layer</h4>
      <div class="pinfo">${S.ref.name}: ${i >= 0 && i < n ? `showing reference frame <b>${i + 1}</b> of ${n}` : `no reference frame here (it covers frames ${S.ref.offset + 1}–${S.ref.offset + n})`}. Never painted or exported.</div>
      <div class="prow two"><label>Opacity</label><input type="range" data-ref="opacity" min="0" max="100" step="1" value="${Math.round(S.ref.opacity * 100)}"></div>
      <div class="prow two"><label>Starts at frame</label><input type="number" data-ref="offset" min="1" step="1" value="${S.ref.offset + 1}"></div>
      <button class="pbtn" data-act="refremove">Remove reference</button></div>`;
  }
  if (!S.ref.images.length) {
    h += `<div class="psec"><h4>Reference layer</h4>
      <div class="refbox" data-act="refimport" role="button" tabindex="0">Drop the animator's frames here<br><span>PNG/JPG frames, a folder, or a video<br>or click to choose files</span></div></div>`;
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
  body.querySelectorAll("input[type=range][data-p]").forEach((inp) => {
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
  body.querySelectorAll("select[data-trim]").forEach((sel) => sel.addEventListener("change", () => {
    pushUndo(); setTrim(sel.dataset.trim, sel.dataset.ids.split(","), sel.value);
    if (sel.value) toast(`${sel.value === "draw_on" ? "Draws on" : "Draws off"} over the in-betweens. Create In-betweens (Ctrl+Enter) to see it.`);
    changed(); renderAll();
  }));
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
    if (act === "rmpath") { pushUndo(); S.doc.paths.splice(+b.dataset.idx, 1); toast("Motion path removed."); changed(); renderAll(); }
    if (act === "carry") { pushUndo(); keyAt(f).strokes.push(...clone(traceCandidates(f))); changed(); renderAll(); }
  }));
}

// ------------------------------------------------------------ misc UI ---
function updateModeHint() {
  const f = S.ui.frame, c = content(f);
  let t = "";
  if (S.ui.picking) t = "Click the ghost stroke to link to";
  else if (S.ui.tool === "link") t = S.ui.linkFirst ? "Link · click the ghost it continues (Esc leaves)" : "Link · click a stroke, then its ghost (Esc leaves)";
  else if (S.ui.tool === "pen") t = S.ui.bez ? "Pen · click = sharp point, drag = curve (Alt breaks the handles) · click first point to close · Enter finishes · Esc cancels" : "Pen · click to start a line (drag for a curve)";
  else if (S.ui.tool === "bezsel") t = S.ui.bez ? "Bezier Selection · click / drag points · click first point or Enter to select · Esc cancels" : "Bezier Selection · outline an area; every stroke it touches is selected (Shift adds)";
  else if (S.ui.tool === "subselection" && S.ui.sel.size) t = "Subselection · squares move corners · round handles set how wide each curve is";
  else if (S.ui.tool === "motion") t = S.ui.sel.size ? "Motion Path · drag the path the selected strokes travel" : "Motion Path · click the strokes to move";
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
  const rows = [["V", "Selection tool: click, or drag a rectangle (it selects every stroke it touches; Shift adds), drag to move; round handle rotates"], ["A", "Subselection tool: drag corner points (squares); drag a round handle along its edge for a wider or tighter curve (both handles mirror)"], ["B", "Brush tool (stylus pressure sets width)"],
    ["N", "Pen tool: click for sharp points, drag for curves (Alt breaks the handles); click the first point to close, Enter or double-click finishes, Backspace removes the last point, Esc cancels. The line becomes robot strokes."],
    ["Q", "Bezier Selection: outline an area the same way; every stroke it touches is selected (Shift adds)"],
    ["M", "Motion Path tool: click strokes (Shift adds), then drag the path they travel to the next keyframe"],
    ["L", "Link mode: click a stroke, then the ghost it continues; repeat. Esc leaves"],
    ["Esc", "Deselect (in Link mode: leave it)"], ["Ctrl+A", "Select all"],
    ["Alt+click ghost", "Brush: choose which ghost the next stroke traces"], ["H / Space", "Hand tool / temporary pan"], ["Z", "Zoom tool (Alt+click zooms out)"],
    ["F5 / Shift+F5", "Insert / remove frame (timing)"], ["F6", "Insert keyframe (copy the drawing)"], ["F7", "Insert blank keyframe (trace over onion skin)"],
    ["Shift+F6", "Clear keyframe"], ["Ctrl+Enter", "Create in-betweens"], ["Enter", "Play / stop"], [", / .", "Previous / next frame"],
    ["Shift+, / Shift+.", "Previous / next keyframe"], ["O", "Onion skin"], ["R", "Reference layer on / off (File > Import Reference Frames)"], ["P", "Paint preview (stroke model)"], ["Del", "Delete stroke, or unpin on an in-between"],
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
  const hasRef = S.ref.images.length > 0;
  $("tl-ref").classList.toggle("on", hasRef && S.ref.visible);
  $("tl-ref").textContent = hasRef ? "Reference" : "Reference…";
  $("tl-refop").hidden = !hasRef;
  $("tl-refop").value = Math.round(S.ref.opacity * 100);
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
$("tl-ref").onclick = () => cmds.toggleReference();
$("tl-refop").oninput = (e) => { S.ref.opacity = clamp(+e.target.value / 100, 0, 1); S.ref.visible = true; $("tl-ref").classList.add("on"); render(); };
$("refinput").addEventListener("change", (e) => { const files = [...e.target.files]; e.target.value = ""; importReference(files); });
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
