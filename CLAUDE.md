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
- `refine.py`: pressure-only refinement toward evenly changing rendered width profiles
  between fixed frames (keys + artist edits); coarse search then gradient descent.
- `metrics.py`: width deviation from even change ("popping"), per stroke.
- `cli.py`: `python -m inbetween.cli <keys.json> --out <dir> --compare`.
- `paint_frames.py`: robot paints each frame exactly as planned, one sheet per frame.
- `tests/`: `python -m pytest -q inbetween/tests` (must pass before committing).

## Rules

- Keyframes and artist-edited strokes are never changed by interpolation or refinement.
- Store what the robot controls (pressure), never visible width.
- Respect robot limits: clamp like `BrushStroke.make_valid`; only palette colors.
- Keep stroke `order` identical across frames (consistent layering, less flicker).
- Never present `placeholder_sigmoid` results as real-brush results.
- Commit messages and PRs: describe what changed and how it was tested.

## Next steps (priority order for the deadline)

1. Animator's real keyframes in the JSON format (an authoring tool or script that writes it).
2. Robot session 1: check Sharpie model vs real strokes; time one frame; test paper registration
   (peg bar) and fixed-camera photos via `paint_frames.py --photo_dir`.
3. Measure popping on photographed frames (same width-profile idea on photos).
4. Video and proposal figures (`compare.gif`, `width_plot.png`).

## Closest prior work

FRIDA (arXiv 2210.00664), Spline-FRIDA / Traj2Stroke (arXiv 2412.00597), BetweenIT (Whited et
al., Eurographics 2010), Sederberg et al. SIGGRAPH 1993, Barroso et al. CGF 2025 (painterly
in-betweening, digital only), AniPainter (Schaldenbrand & Oh 2021; robot paints existing video
frames, no in-betweening; its author is on the arts-track committee).
