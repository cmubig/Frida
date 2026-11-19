# inbetween: in-betweening for robot-painted animation

Generates the frames between an animator's keyframes, refines them so the
*painted* result changes smoothly, and has the robot paint every frame.
Built on FRIDA's stroke representation (`brush_stroke.py`) and calibrated
stroke model (`param2stroke.py`); FRIDA files are not modified.

All commands run from `src/`.

## Editor (for animators)

```bash
python -m inbetween.app        # opens the editor at http://localhost:8765
```

The layout follows Adobe Animate / Flare: tools on the left, stage in the
middle, Properties on the right, Timeline at the bottom, with Animate's
shortcuts (press `?` in the editor for the full list).

- **Brush (B)** draws with stylus pressure. Each mark is fitted to a robot
  stroke right away, so what you see is what the robot will paint; a mark too
  complex for one stroke is split into several. Pressure maps to visible
  width: the lightest touch still leaves a mark.
- **F6 Insert Keyframe** copies the current drawing. Reshape it with
  **Selection (V)** (move; the round handle rotates) and **Subselection (A)**
  (drag corner points). Strokes keep their identity, so in-betweening knows
  what moves where.
- **F7 Insert Blank Keyframe** starts empty with onion skin on. Redraw strokes
  over their ghosts. Each traced stroke replaces the ghost it was drawn over:
  the nearest ghost is suggested (orange), or **Alt+click** a ghost first to
  choose it, which is needed when something moved far. "Pick another ghost"
  in Properties fixes a wrong link.
- **Create In-betweens (Ctrl+Enter)** runs the pipeline (Refined, Arc or
  Linear). **Ease** is set per span in Properties; **F5 / Shift+F5** add or
  remove frames to change timing, as in Animate.
- Changing a stroke on an in-between **pins** it (orange diamond); the next
  Create In-betweens keeps it and re-flows the neighbors. Delete unpins.
- **Paint Preview (P)** shows the stroke model's predicted painting.
- **Ctrl+C / X / V** copy, cut and paste strokes (pasted strokes are new); **Ctrl+D** duplicates. **Ctrl+Shift+V** pastes in place and keeps a stroke's identity when the keyframe doesn't have it yet, so it animates from where it was copied.
- **File > Save** writes the keyframe file below; **File > Export Robot
  Frames** writes the input for `paint_frames.py`.

## Command line

```bash
# 1. (optional) regenerate the example keyframes
python -m inbetween.examples.make_examples

# 2. generate in-betweens with all three methods and compare them
python -m inbetween.cli inbetween/examples/bend.json --out ../out/bend --compare

# 3. paint the refined sequence, one sheet per frame (add --simulate for a dry run)
python -m inbetween.paint_frames --frames ../out/bend/frames_refined.json \
    --photo_dir ../out/bend/photos --cache_dir caches/vae_sharpie_final \
    --materials_json ../materials_vae_sharpie.json --robot xarm --ink

# tests
python -m pytest -q inbetween/tests
```

`--out` gets: `frames_<method>.json` (stroke plans for the robot),
`<method>.gif` and per-frame PNGs, `compare.gif` (methods side by side),
`width_plot.png` and `metrics.json`.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Keyframes | `strokes.py` | JSON keyframes using FRIDA's own stroke parameters, with stable stroke IDs |
| Rendering | `render.py` | One `Renderer` around the calibrated stroke model; optional placeholder pressure response |
| Interpolation | `interpolate.py` | `linear` baseline and `arc` (BetweenIT-style: log-spiral motion + intrinsic shape) |
| Refinement | `refine.py` | Adjusts pressure so painted width changes evenly between fixed frames |
| Measurement | `metrics.py` | Width deviation from an even change ("popping"), per stroke |
| Robot | `paint_frames.py` | Paints each frame exactly as planned, no re-planning |
| Drawing input | `fit.py` | Fits a freehand mark + stylus pressure to a robot stroke |
| Editor | `app.py`, `editor/` | Local server + browser editor; `editor/geometry.js` mirrors FRIDA's stroke path math |

### How refinement works

A brush or marker does not turn even pressure changes into even width
changes. The calibrated Sharpie model in `caches/vae_sharpie_final` already
shows this: below a pressure of about 0.23 it leaves no mark, so an evenly
interpolated pressure change makes the stroke appear suddenly.

For each stroke, refinement reads its painted width at 8 places along the
stroke off the rendered image (so it works with any stroke model). For
every in-between, the target width profile changes evenly, in the animator's
eased time, between the surrounding fixed frames. It then solves for the
pressure values that produce that width: a coarse search first (inside the
no-mark zone the renderer has no gradient), then gradient descent through the
differentiable renderer. Only pressure changes, so motion and shape stay as
interpolated.

Fixed frames are keyframes and artist edits. An edit is a stroke saved on an
in-between frame (`"edits"` in the keyframe file). Interpolation and
refinement both treat it as a fixed point and re-flow the neighboring frames
around it. See `examples/bend_with_edit.json`.

A marker cannot paint arbitrarily thin lines. When the even target is thinner
than the thinnest mark the model can make, the fit gets as close as it can,
which can leave one visible jump near the no-mark end.

### Pressure response

- `--response model` (default): the calibrated model as fitted.
- `--response placeholder_sigmoid`: an **assumed** S-shaped pressure curve,
  for developing refinement before real calibration data exists. Do not
  report its numbers as real-brush results.

To use a better stroke model later, change `render.py` only.

## Keyframe file

```json
{
  "format": "frida-inbetween", "version": 1,
  "cache_dir": "caches/vae_sharpie_final",
  "keys": [
    {"frame": 0,  "strokes": [ STROKE, ... ]},
    {"frame": 10, "strokes": [ STROKE, ... ]}
  ],
  "edits":  [ {"frame": 5, "strokes": [ STROKE ]} ],
  "timing": { "0-10": "ease_in_out" }
}
```

`STROKE`:

| Field | Meaning |
|---|---|
| `id` | Stable ID. The same ID in two keyframes is the same stroke (that is the correspondence). |
| `order` | Painting order in the frame; keep it the same across frames. |
| `color` | `[r, g, b]` in 0..1 (ignored with `--ink`). |
| `vertices` | 4 `[x, y]` corner points in meters, stroke-local. |
| `kappa` | 2 corner roundness values, 0 = sharp, 1 = round. |
| `z` | 4 pressure values along the stroke, 0 = lightest touch, 1 = deepest. What the robot commands; visible width is predicted, never stored. |
| `xt`, `yt` | Placement on the canvas as proportions 0..1 (FRIDA's convention). |
| `a` | Rotation in radians. |

Timing options: `linear`, `ease_in`, `ease_out`, `ease_in_out`. A stroke that
is missing from one of two neighboring keys pops in or out at the middle of
that segment. Make keyframe B by duplicating keyframe A and editing it, so
the IDs carry over.

## Known limitations

- Pose (`xt`, `yt`, `a`) is not refined; refinement only changes pressure.
- Strokes popping in or out appear all at once; there is no grow or fade yet.
- `paint_frames.py` was only dry-run in `--simulate` mode, not on hardware.
- A robot stroke has only 4 corners. A drawn mark that one stroke can't follow
  within 1 mm (a loop, a zigzag, a long scribble) is split automatically into
  up to 8 connected strokes, cut at its sharpest turns. When tracing, each
  piece links to its nearest ghost. Pressure detail inside one piece is kept
  only at 4 points.
- The editor was tested with a mouse in a browser, not yet with a real stylus.
- Metrics come from the stroke model. Claims about real paintings need
  photographed frames, measured the same way.
