"""In-betweening for robot-painted animation, built on FRIDA's stroke model.

Pipeline: keyframes (JSON, stable stroke IDs) -> interpolation (linear or arc)
-> optional appearance-space refinement through the differentiable stroke
renderer -> rendered frames / GIFs / per-frame stroke plans for the robot.

Run commands from the repo's ``src/`` directory, e.g.::

    python -m inbetween.cli inbetween/examples/bend.json --out ../out/bend --compare
"""
