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
- **Motion Path (M)**, as in After Effects: click the strokes to move (Shift
  adds, or select them first with V), then drag the path they travel to the
  next keyframe ("draw the arc"). The group's centre follows the path instead
  of the arc method's own motion; its rotation and shape still change as
  between the keys. The ends of the path snap to where the group is on the two
  keyframes, and editing a key bends the path with it. One dot per frame on the
  path shows the spacing, so the ease is visible. A stroke pinned on an
  in-between keeps its pin and doesn't follow the path. A path that leaves the
  paper stops the strokes at the edge (with a warning). F5 / Shift+F5 move the
  path with its keyframes, F6 / F7 inside the span split it in two, and
  clearing a keyframe at either end removes it. Properties > Motion Path >
  Remove path.
- **Paint Preview (P)** shows the stroke model's predicted painting.
- **File > Import Reference Frames** loads the animator's own frames (a PNG
  sequence, sorted by number, or a video) under the drawing, one image per
  timeline frame, to trace over. **R** toggles it; the slider next to the
  Reference button sets its opacity, and Properties can move where it starts.
  It is placed exactly as `import_frames.py` places strokes, so converted
  strokes line up with it. It is never saved, exported or painted.
- **Ctrl+C / X / V** copy, cut and paste strokes (pasted strokes are new); **Ctrl+D** duplicates. **Ctrl+Shift+V** pastes in place and keeps a stroke's identity when the keyframe doesn't have it yet, so it animates from where it was copied.
- **File > Save** writes the keyframe file below; **File > Export Robot
  Frames** writes the input for `paint_frames.py`. **File > Export FRIDA
  Plans** downloads a .zip with one FRIDA plan per frame (`frame_NNN.pkl`,
  the format `paint.py --painting_path` uses), a preview PNG of each, and the
  frames JSON. It reads the robot's materials file: start the editor with
  `--materials_json <file>` if the default (`../materials_vae_sharpie.json`
  for ink, `../materials_xarm_paint.json` for paint) is not the right one.

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

# 4. (optional) export each frame as a FRIDA plan, the pickle paint.py uses
python -m inbetween.export_plans --frames ../out/bend/frames_refined.json \
    --out ../out/bend/plans --cache_dir caches/vae_sharpie_final \
    --materials_json ../materials_vae_sharpie.json --ink

# 5. (optional) convert an animator's finished PNG frames into robot strokes
python -m inbetween.import_frames "path/to/Jump/*.png" --out ../out/jump

# tests
python -m pytest -q inbetween/tests
```

`--out` gets: `frames_<method>.json` (stroke plans for the robot),
`<method>.gif` and per-frame PNGs, `compare.gif` (methods side by side),
`width_plot.png` and `metrics.json`.

`frames_<method>.json` is the plan for the whole animation and is what
`paint_frames.py` paints. `export_plans.py` turns it into one FRIDA plan per
frame for FRIDA's own tools. `paint.py --painting_path` can load one, but it
re-optimizes strokes toward its objective between batches, which changes the
frame; use `paint_frames.py` to paint frames exactly as planned.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Keyframes | `strokes.py` | JSON keyframes using FRIDA's own stroke parameters, with stable stroke IDs |
| Rendering | `render.py` | One `Renderer` around the calibrated stroke model; optional placeholder pressure response |
| Interpolation | `interpolate.py` | `linear` baseline and `arc` (BetweenIT-style: log-spiral motion + intrinsic shape) |
| Motion paths | `motion_path.py` | Moves a group of strokes along a path the animator drew, between two keys |
| Refinement | `refine.py` | Adjusts pressure so painted width changes evenly between fixed frames |
| Measurement | `metrics.py` | Width deviation from an even change ("popping"), per stroke |
| Robot | `paint_frames.py` | Paints each frame exactly as planned, no re-planning |
| FRIDA plans | `export_plans.py` | Writes each frame as `frame_NNN.pkl` = `(Painting, palette)`, FRIDA's own plan format, plus a preview PNG |
| Drawing input | `fit.py` | Fits a freehand mark + stylus pressure to a robot stroke |
| Raster import | `import_frames.py` | Traces finished PNG frames (line centers -> polylines -> `fit_mark`); static lines become one background |
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
  "format": "frida-inbetween", "version": 2,
  "cache_dir": "caches/vae_sharpie_final",
  "keys": [
    {"frame": 0,  "strokes": [ STROKE, ... ]},
    {"frame": 10, "strokes": [ STROKE, ... ]}
  ],
  "edits":  [ {"frame": 5, "strokes": [ STROKE ]} ],
  "timing": { "0-10": "ease_in_out" },
  "paths":  [ {"span": "0-10", "ids": ["s3", "s4"], "points": [[0.2, 0.6], [0.4, 0.3], ...]} ]
}
```

`paths` (version 2; version 1 files without it still load) are motion paths:
the strokes in `ids` travel as a group along `points` (the drawn polyline, in
canvas proportions like `xt`, `yt`) between the neighboring keyframes in
`span`, at the span's eased time. Only `xt`, `yt` of the in-betweens change;
keys, `id` and `order` don't. Paths are never painted. The frames file that
`paint_frames.py` reads is unchanged (still version 1).

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
