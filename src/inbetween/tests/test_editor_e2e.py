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


def test_reference_layer(server, tmp_path):
    from PIL import Image, ImageDraw
    files = []
    for i in range(5):
        im = Image.new("RGB", (320, 180), "white")
        ImageDraw.Draw(im).ellipse([40 + 40 * i, 60, 80 + 40 * i, 100], outline="black", width=3)
        fn = tmp_path / f"ref_{i + 1}.png"  # numeric order, not text order, decides frame order
        im.save(fn)
        files.append(str(fn))
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
        pg.set_input_files("#refinput", files[::-1])
        pg.wait_for_function("S.ref.images.length === 5 && !S.ui.busy", timeout=30000)
        assert pg.evaluate("lastFrame()") >= 4
        assert pg.evaluate("S.ref.visible")
        pg.keyboard.press("r")
        assert not pg.evaluate("S.ref.visible")
        pg.keyboard.press("r")
        pg.locator("#tl-refop").fill("80")
        assert abs(pg.evaluate("S.ref.opacity") - 0.8) < 1e-6
        # the reference never enters the document or its exports
        assert pg.evaluate("JSON.stringify(S.doc).includes('ref_')") is False
        assert pg.evaluate("S.doc.keys.every(k => k.strokes.length === 0)")
        # the frame shown follows the timeline: frame 3 shows the 3rd image (circle moved right)
        pg.evaluate("goto(2)")
        assert pg.evaluate("refImage(2) === S.ref.images[2]")
        # moving the reference later grows the timeline to cover it
        before = pg.evaluate("lastFrame()")
        off = pg.locator('[data-ref="offset"]')
        off.fill(str(before + 1))
        off.dispatch_event("change")
        assert pg.evaluate("lastFrame()") == before + 4
        assert pg.evaluate(f"refImage({before}) === S.ref.images[0]")
        pg.evaluate("cmds.removeReference()")
        assert pg.evaluate("S.ref.images.length") == 0
        assert errors == []
        browser.close()


def test_motion_path(server, tmp_path):
    """Draw a motion path with the Motion Path tool (M): the group travels along it in the
    created in-betweens (same positions the stage's per-frame dots show), through Paint
    Preview and both exports; F5 / Shift+F5 shift it, F6 inside the span splits it, deleting
    a stroke removes it from the path, clearing an end keyframe drops it."""
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

        def scr(px, py):
            v = pg.evaluate("[S.view.ox, S.view.oy, S.view.scale, S.cfg.canvas_width_m/S.cfg.canvas_height_m]")
            return st["x"] + v[0] + px * v[2] * v[3], st["y"] + v[1] + py * v[2]

        def drag(pts, wait_fit=False):
            pg.mouse.move(*scr(*pts[0]))
            pg.mouse.down()
            for q in pts[1:]:
                pg.mouse.move(*scr(*q), steps=2)
            pg.mouse.up()
            if wait_fit:
                pg.wait_for_function("!pendingFit", timeout=30000)

        drag([(0.2 + 0.01 * i, 0.6) for i in range(11)], True)  # two short strokes on keyframe 1
        drag([(0.2 + 0.01 * i, 0.66) for i in range(11)], True)
        pg.evaluate("goto(8)")
        pg.keyboard.press("F6")  # keyframe 9: the group moved right
        pg.evaluate("pushUndo(); keyAt(8).strokes.forEach(s => { s.xt += 0.45; }); changed(); renderAll();")
        pg.evaluate("goto(0)")
        pg.keyboard.press("m")
        assert pg.evaluate("S.ui.tool") == "motion"
        pg.keyboard.press("Control+a")
        ids = sorted(pg.evaluate("[...S.ui.sel]"))
        assert len(ids) == 2
        # the arc: from the group up and over to where it lands
        drag([(0.25 + 0.45 * i / 20, 0.63 - 0.3 * (1 - (2 * i / 20 - 1) ** 2)) for i in range(21)])
        assert pg.evaluate("S.doc.paths.length") == 1
        assert pg.evaluate("S.doc.paths[0].span") == "0-8"
        assert sorted(pg.evaluate("S.doc.paths[0].ids")) == ids
        assert pg.evaluate("S.doc.version") == 2

        pg.evaluate("S.doc.timing['0-8'] = 'ease_in_out'")
        pg.select_option("#tl-method", "arc")
        pg.keyboard.press("Control+Enter")
        pg.wait_for_function("!S.ui.busy && Object.keys(S.gen.frames).length === 7", timeout=120000)
        # the server's in-betweens put the anchor exactly where the editor's dots are
        dev = pg.evaluate("""(() => { const p = S.doc.paths[0], q = fittedPath(p); let worst = 0;
            for (let f = 1; f < 8; f++) { const g = S.gen.frames[f].strokes.filter(s => p.ids.includes(s.id));
              const a = groupAnchor(g), w = pointAt(q, easedT(0, 8, f)); worst = Math.max(worst, Math.hypot(a[0] - w[0], a[1] - w[1])); }
            return worst; })()""")
        assert dev < 1e-4
        lift = pg.evaluate("0.63 - groupAnchor(S.gen.frames[4].strokes)[1]")
        assert lift > 0.15  # it went up the arc, not along the straight line
        assert pg.evaluate("S.doc.keys.map(k => k.strokes.length)") == [2, 2]

        pg.evaluate("goto(4)")  # Paint Preview renders the moved group
        pg.keyboard.press("p")
        pg.wait_for_function("[...S.previewCache.values()].some(v => v instanceof Image && v.complete)", timeout=60000)
        pg.keyboard.press("p")
        frames = pg.evaluate("framesForExport()")  # Export Robot Frames
        f4 = {s["id"]: s for s in frames["frames"][4]["strokes"]}
        gen4 = {s["id"]: s for s in pg.evaluate("S.gen.frames[4].strokes")}
        assert all(abs(f4[i]["yt"] - gen4[i]["yt"]) < 1e-9 for i in ids)
        with pg.expect_download(timeout=180000) as dl:  # Export FRIDA Plans
            pg.evaluate("cmds.exportPlans()")
        zpath = tmp_path / "plans.zip"
        dl.value.save_as(zpath)
        with zipfile.ZipFile(zpath) as z:
            painting, _ = pickle.loads(z.read("frame_004.pkl"))
        yts = sorted(float(b.yt.detach()) for b in painting.brush_strokes)
        assert yts == pytest.approx(sorted(gen4[i]["yt"] for i in ids), abs=1e-5)

        pg.evaluate("goto(2)")  # F5 / Shift+F5 move the span's end, the path goes with it
        pg.keyboard.press("F5")
        assert pg.evaluate("S.doc.paths[0].span") == "0-9"
        pg.keyboard.press("Shift+F5")
        assert pg.evaluate("S.doc.paths[0].span") == "0-8"

        reached = pg.evaluate("pointAt(fittedPath(S.doc.paths[0]), easedT(0, 8, 4))")
        pg.evaluate("goto(4)")  # F6 inside the span splits the path where the anchor had reached
        pg.keyboard.press("F6")
        spans = pg.evaluate("S.doc.paths.map(p => p.span)")
        assert spans == ["0-4", "4-8"]
        ends = pg.evaluate("[S.doc.paths[0].points.at(-1), S.doc.paths[1].points[0]]")
        assert ends[0] == ends[1]
        assert math.hypot(ends[0][0] - reached[0], ends[0][1] - reached[1]) < 1e-9
        pg.keyboard.press("Control+z")
        assert pg.evaluate("S.doc.paths.map(p => p.span)") == ["0-8"]

        pg.evaluate("goto(0)")  # deleting a stroke from a key removes it from the path
        pg.evaluate(f"S.ui.sel = new Set(['{ids[0]}'])")
        pg.keyboard.press("Delete")
        assert pg.evaluate("S.doc.paths[0].ids") == [ids[1]]
        pg.keyboard.press("Control+z")

        pg.evaluate("goto(8)")  # clearing an end keyframe drops the path, with a toast
        pg.keyboard.press("Shift+F6")
        assert pg.evaluate("S.doc.paths.length") == 0
        assert "Removed 1 motion path" in pg.locator("#toast").inner_text()
        assert errors == []
        browser.close()
