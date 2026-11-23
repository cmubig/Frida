# CLAUDE.md: Robot Flipbook Artist (in-betweening on top of FRIDA)

Context for coding agents in this repository (`cmubig/Frida`, branch `b-spline-dev`, based on
`b-spline`). Read this before changing code.

## Project

An animator draws only keyframes; the system generates the in-between frames and a robot paints
every frame on paper. Two contributions:
1. **Appearance-space refinement**: in-betweens are adjusted through FRIDA's calibrated,
   differentiable stroke model so the *painted* width changes evenly, not just the geometry.
2. **Animator-centered tool (HCI)**: reduce the animator's workload while keeping their autonomy.
   Artist edits are never overridden; the system re-flows around them.

**Current deadline: ICRA 2027 Arts in Robotics track, December 1, 2026, 23:59 PST.** Needs a
2-4 page proposal and a video (<= 3 min) of the physical system painting a finished piece. Scope
is cut to what that needs: simulation pipeline + robot-painted marker sequence. Flare, user
study, learning from corrections, coloring and real paint are future work.

## Code layout

FRIDA (do not edit unless a small hook is truly needed):
- `src/brush_stroke.py`: `BrushStroke`. 4 corner `vertices` (meters, stroke-local), 2 corner
  roundness logits `radii` (kappa = sigmoid), 4 pressure values `z`, pose `xt`, `yt`
  (canvas proportions), `a` (radians). `get_path()` -> (32, 3).
- `src/param2stroke.py`: differentiable soft polyline renderer. Width is linear in pressure with
  a negative bias, so the Sharpie model leaves no mark below z ~ 0.23.
- Calibrated models: `src/caches/{vae_sharpie_final, vae_brush_final, vae_largebrush_final,
  small_brush}/param2img.pt` (all load with the current code).
- `src/paint.py`: FRIDA's own executor (re-plans while painting; we don't use it for frames).

Ours, `src/inbetween/` (see its README.md; run commands from `src/`):
- `strokes.py`: keyframe JSON format (FRIDA stroke params + stable `id` + `order`), loading,
  `stroke_path()` (pure function, identical to `BrushStroke.get_path`, no cache).
- `render.py`: the single `Renderer` interface. `response="model"` or the ASSUMED
  `"placeholder_sigmoid"`. Swapping stroke models must only touch this file.
- `interpolate.py`: `linear` and `arc` (BetweenIT-style log-spiral motion + intrinsic shape),
  eased timing, strokes popping in/out, artist edits as extra per-stroke keys.
- `motion_path.py`: motion paths (keyframe file version 2, `"paths"`): a group's anchor (mean
  of its strokes' centres) follows a drawn polyline, fitted so its ends land on the anchor at
  the two keys, at eased t by arc length. Only in-between `xt`/`yt` change; members exclude
  strokes pinned in the span (on every frame, so the anchor stays steady); the shift keeps the
  group on the paper (`xt`, `yt` in [0, 1] as `make_valid`). `editor/app.js` mirrors
  `fit_path` / `point_at` (the per-frame dots); the browser test checks they agree.
- `refine.py`: pressure-only refinement toward evenly changing rendered width profiles
  between fixed frames (keys + artist edits); coarse search then gradient descent.
- `metrics.py`: width deviation from even change ("popping"), per stroke.
- `cli.py`: `python -m inbetween.cli <keys.json> --out <dir> --compare`.
- `paint_frames.py`: robot paints each frame exactly as planned, one sheet per frame.
- `export_plans.py`: each frame -> FRIDA plan pickle `(Painting, palette)` + preview PNG (always
  renders on the CPU, GPU or not, so the pickles load anywhere). Also the editor's File > Export
  FRIDA Plans (`/api/export_plans`, a .zip; app.py's `--materials_json` picks the materials file).
- `fit.py`: freehand mark + stylus pressure -> robot stroke (pressure mapped to visible width).
  `fit_mark` splits a mark one stroke can't follow within 1 mm at its sharpest turns (up to 8
  connected pieces); the editor links each traced piece to its nearest ghost.
  Stylus pressure 0 maps to `min_line_z` (lightest pressure painting a ~0.4 mm line, measured
  from the stroke model), not `visible_z_min` (where the mark starts: a near-zero hairline).
- `import_frames.py`: animator's raster frames (PNG) -> robot strokes: skeletonize lines, trace
  polylines, `fit_mark`; lines in every drawing -> background; holds detected; gray fills only
  reported. Strokes are not matched across drawings.
- `app.py` + `editor/`: local server and Animate/Flare-style browser editor
  (`python -m inbetween.app`). F6 = copy keyframe and reshape; F7 = blank keyframe, trace over
  onion skin (traced stroke replaces its ghost; Alt+click picks the ghost). `editor/geometry.js`
  must stay identical in behavior to FRIDA's path math (tested by `test_editor_geometry.py`).
  Reference layer (`S.ref`, File > Import Reference Frames, R): animator's PNGs or a video under
  the drawing for tracing; never in the document, exports or paintings.
- `tests/`: `python -m pytest -q inbetween/tests` (must pass before committing; the browser test
  needs Playwright, the geometry test needs Node). On the lab machine (no sudo), Node is in
  `~/.local/bin` and Chromium's missing `libasound.so.2` in `~/.local/lib/chromium-extra`:
  `PATH=$HOME/.local/bin:$PATH LD_LIBRARY_PATH=$HOME/.local/lib/chromium-extra uv run --with pytest --with playwright python -m pytest -q inbetween/tests`

FRIDA change: `src/filleted_stroke.py` treats corners within 0.02 rad of straight as straight
(`FLAT_ANGLE`). The old 1e-3 threshold sat below a clamp floor, so near-straight corners could
produce paths meters away (about 1.4% of random near-straight strokes). Shape change on ordinary
strokes is at most 0.6 mm.

## Rules

- Keyframes and artist-edited strokes are never changed by interpolation or refinement.
- Store what the robot controls (pressure), never visible width.
- Respect robot limits: clamp like `BrushStroke.make_valid`; only palette colors.
- Keep stroke `order` identical across frames (consistent layering, less flicker).
- Never present `placeholder_sigmoid` results as real-brush results.
- Commit messages and PRs: describe what changed and how it was tested.

## Next steps (priority order for the deadline)

1. Animator draws real keyframes in the editor; test with their actual stylus.
2. Robot session 1: check Sharpie model vs real strokes; time one frame; test paper registration
   (peg bar) and fixed-camera photos via `paint_frames.py --photo_dir`.
3. Measure popping on photographed frames (same width-profile idea on photos).
4. Video and proposal figures (`compare.gif`, `width_plot.png`).

## Closest prior work

FRIDA (arXiv 2210.00664), Spline-FRIDA / Traj2Stroke (arXiv 2412.00597), BetweenIT (Whited et
al., Eurographics 2010), Sederberg et al. SIGGRAPH 1993, Barroso et al. CGF 2025 (painterly
in-betweening, digital only), AniPainter (Schaldenbrand & Oh 2021; robot paints existing video
frames, no in-betweening; its author is on the arts-track committee).
