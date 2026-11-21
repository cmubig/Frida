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


def test_marquee_selects_strokes_fully_inside(server):
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"no Chromium: {e}")
        pg = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(server)
        pg.wait_for_function("S.cfg && S.doc")
        st = pg.locator("#canvas").bounding_box()

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

        def box(a, b, shift=False):
            if shift:
                pg.keyboard.down("Shift")
            pg.mouse.move(*scr(*a))
            pg.mouse.down()
            pg.mouse.move(*scr(*b), steps=6)
            pg.mouse.up()
            if shift:
                pg.keyboard.up("Shift")
            return sorted(pg.evaluate("[...S.ui.sel]"))

        draw([(0.2 + 0.1 * i / 10, 0.3) for i in range(11)])  # s1, top left
        draw([(0.2 + 0.1 * i / 10, 0.5) for i in range(11)])  # s2, middle left
        draw([(0.6 + 0.1 * i / 10, 0.3) for i in range(11)])  # s3, top right
        pg.keyboard.press("v")
        assert box((0.1, 0.2), (0.4, 0.6)) == ["s1", "s2"]         # both left strokes
        assert box((0.25, 0.2), (0.45, 0.4)) == []                 # s1 only partly inside
        assert box((0.5, 0.2), (0.8, 0.4), shift=True) == ["s3"]   # Shift adds to an empty selection
        assert box((0.1, 0.45), (0.4, 0.55), shift=True) == ["s2", "s3"]
        pg.mouse.click(*scr(0.5, 0.8))                             # click on empty paper clears
        assert pg.evaluate("S.ui.sel.size") == 0
        assert errors == []
        browser.close()


def test_copy_paste_duplicate(server):
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:
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

        key = lambda i: pg.evaluate(f"S.doc.keys[{i}].strokes.map(s => [s.id, +s.xt.toFixed(5), +s.yt.toFixed(5)])")

        draw([(0.2 + 0.1 * i / 10, 0.3) for i in range(11)])  # s1
        draw([(0.2 + 0.1 * i / 10, 0.5) for i in range(11)])  # s2
        pg.keyboard.press("v")
        pg.mouse.click(*scr(0.25, 0.3))  # select s1
        pg.keyboard.press("Control+c")
        pg.keyboard.press("Control+v")  # new stroke, offset
        k0 = key(0)
        assert [s[0] for s in k0] == ["s1", "s2", "s3"]
        assert k0[2][1] > k0[0][1] and k0[2][2] > k0[0][2]

        pg.keyboard.press("Control+d")  # duplicate the pasted s3
        assert [s[0] for s in key(0)] == ["s1", "s2", "s3", "s4"]

        pg.mouse.click(tl["x"] + 9 * 14 + 7, tl["y"] + 30)
        pg.keyboard.press("F7")
        pg.keyboard.press("Control+Shift+v")  # s1 is not in this keyframe yet: keeps its ID, same spot
        assert key(1) == [k0[0]]
        pg.keyboard.press("Control+Shift+v")  # s1 now exists here: a new stroke
        assert [s[0] for s in key(1)] == ["s1", "s5"]

        pg.keyboard.press("Control+x")  # cut s5
        assert [s[0] for s in key(1)] == ["s1"]
        pg.keyboard.press("Control+z")
        assert [s[0] for s in key(1)] == ["s1", "s5"]
        assert errors == []
        browser.close()


def test_complex_mark_splits_into_linked_strokes(server):
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:
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
            pg.wait_for_function("!pendingFit", timeout=60000)

        def loop(cx, cy):
            return [(cx + 0.1 * math.cos(2 * math.pi * 1.3 * i / 60), cy + 0.1 * math.sin(2 * math.pi * 1.3 * i / 60))
                    for i in range(61)]

        draw(loop(0.35, 0.5))
        ids0 = pg.evaluate("S.doc.keys[0].strokes.map(s => s.id)")
        assert len(ids0) > 1  # one gesture, several robot strokes
        assert sorted(pg.evaluate("[...S.ui.sel]")) == sorted(ids0)

        # Trace the same loop on a blank keyframe: every piece links to a ghost.
        pg.mouse.click(tl["x"] + 9 * 14 + 7, tl["y"] + 30)
        pg.keyboard.press("F7")
        draw(loop(0.36, 0.5))
        ids1 = pg.evaluate("S.doc.keys[1].strokes.map(s => s.id)")
        assert len(set(ids1) & set(ids0)) >= min(len(ids0), len(ids1)) - 1
        pg.keyboard.press("Control+z")
        assert pg.evaluate("S.doc.keys[1].strokes.length") == 0  # one undo removes all pieces
        assert errors == []
        browser.close()


def test_export_frida_plans_zip(server, tmp_path):
    import pickle
    import zipfile
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"no Chromium: {e}")
        pg = browser.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(server)
        pg.wait_for_function("S.cfg && S.doc")
        st = pg.locator("#canvas").bounding_box()
        tl = pg.locator("#tlcanvas").bounding_box()

        def scr(px, py):
            v = pg.evaluate("[S.view.ox, S.view.oy, S.view.scale, S.cfg.canvas_width_m/S.cfg.canvas_height_m]")
            return st["x"] + v[0] + px * v[2] * v[3], st["y"] + v[1] + py * v[2]

        pg.mouse.move(*scr(0.2, 0.5))
        pg.mouse.down()
        for i in range(1, 11):
            pg.mouse.move(*scr(0.2 + 0.03 * i, 0.5), steps=2)
        pg.mouse.up()
        pg.wait_for_function("!pendingFit", timeout=30000)
        pg.mouse.click(tl["x"] + 4 * 14 + 7, tl["y"] + 30)
        pg.keyboard.press("F6")  # keyframe at frame 4, then move the stroke
        pg.keyboard.press("v")
        x, y = scr(0.35, 0.5)
        pg.mouse.move(x, y)
        pg.mouse.down()
        pg.mouse.move(x + 40, y - 30, steps=4)
        pg.mouse.up()
        pg.select_option("#tl-method", "arc")
        pg.keyboard.press("Control+Enter")
        pg.wait_for_function("!S.ui.busy && Object.keys(S.gen.frames).length === 3", timeout=120000)

        with pg.expect_download(timeout=180000) as dl:
            pg.evaluate("cmds.exportPlans()")
        path = tmp_path / "plans.zip"
        dl.value.save_as(path)
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            assert {f"frame_{i:03d}.pkl" for i in range(5)} <= names
            assert {"frame_000.png", "frames.json", "README.txt"} <= names
            painting, palette = pickle.loads(z.read("frame_002.pkl"))
        assert len(painting) == 1
        assert errors == []
        browser.close()
