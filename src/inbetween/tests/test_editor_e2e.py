"""Browser test of the editor: draw, F7-trace (with Alt+click linking), create
in-betweens, pin an edit, rotate. Needs Playwright with Chromium; skipped
otherwise. Run from src/:  python -m pytest -q inbetween/tests/test_editor_e2e.py"""

import math
import os
import threading
from http.server import ThreadingHTTPServer

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from inbetween.app import State, make_handler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "..", "caches", "vae_sharpie_final")


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(State(CACHE, "model", ink=True)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()


def test_trace_generate_pin_rotate(server):
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # browser not installed
            pytest.skip(f"no Chromium: {e}")
        pg = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(server)
        pg.wait_for_function("S.cfg && S.doc")
        st = pg.locator("#canvas").bounding_box()
        tl = pg.locator("#tlcanvas").bounding_box()

        def scr(px, py):
            v = pg.evaluate("[S.view.ox, S.view.oy, S.view.scale, S.cfg.canvas_width_m/S.cfg.canvas_height_m]")
            return st["x"] + v[0] + px * v[2] * v[3], st["y"] + v[1] + py * v[2]

        def draw(pts):
            pg.mouse.move(*scr(*pts[0]))
            pg.mouse.down()
            for q in pts[1:]:
                pg.mouse.move(*scr(*q), steps=2)
            pg.mouse.up()
            pg.wait_for_function("!pendingFit", timeout=30000)

        def arc(cx, cy, r, a0, a1, n=20):
            return [(cx + r / 1.25 * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n))
                    for i in range(n + 1)]

        def frame(f):
            pg.mouse.click(tl["x"] + f * 14 + 7, tl["y"] + 30)

        draw(arc(0.25, 0.45, 0.08, math.pi, 2 * math.pi))  # ball top
        draw([(0.1 + 0.8 * i / 20, 0.75) for i in range(21)])  # ground
        frame(9)
        pg.keyboard.press("F7")
        draw([(0.1 + 0.8 * i / 20, 0.752) for i in range(21)])  # traced near its ghost: auto link
        pg.keyboard.down("Alt")
        pg.mouse.click(*scr(0.25, 0.37))  # choose the ball's ghost explicitly
        pg.keyboard.up("Alt")
        draw(arc(0.72, 0.4, 0.08, math.pi, 2 * math.pi))  # ball moved far away
        assert pg.evaluate("S.doc.keys[0].strokes.map(s=>s.id).sort()") == \
            pg.evaluate("S.doc.keys[1].strokes.map(s=>s.id).sort()")

        pg.keyboard.press("Control+Enter")
        pg.wait_for_function("!S.ui.busy && Object.keys(S.gen.frames).length === 8", timeout=240000)

        frame(4)  # pin: move the ground on an in-between
        pg.keyboard.press("v")
        x, y = scr(0.5, 0.75)
        pg.mouse.move(x, y)
        pg.mouse.down()
        pg.mouse.move(x, y - 12, steps=4)
        pg.mouse.up()
        assert pg.evaluate("S.doc.edits.length === 1 && S.doc.edits[0].frame === 4 && S.gen.stale")

        pg.keyboard.press("Control+z")
        assert pg.evaluate("S.doc.edits.length") == 0
        assert errors == []
        browser.close()
