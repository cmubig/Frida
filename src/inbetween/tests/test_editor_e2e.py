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


def test_marquee_selects_strokes_it_touches(server):
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
        assert box((0.25, 0.2), (0.45, 0.4)) == ["s1"]             # touching is enough: s1 is partly inside
        assert box((0.26, 0.27), (0.28, 0.33)) == ["s1"]           # a box crossing s1 with no point inside
        assert box((0.5, 0.2), (0.8, 0.4), shift=True) == ["s1", "s3"]   # Shift adds
        assert box((0.1, 0.45), (0.4, 0.55), shift=True) == ["s1", "s2", "s3"]
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
        assert pg.evaluate("S.doc.version") == 3

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


def test_new_keyframe_keeps_span_ease(server):
    """F6 / F7 inside a span split it; both halves keep its ease, for timing and motion paths."""
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

        def drag(pts):
            pg.mouse.move(*scr(*pts[0]))
            pg.mouse.down()
            for q in pts[1:]:
                pg.mouse.move(*scr(*q), steps=2)
            pg.mouse.up()
            pg.wait_for_function("!pendingFit", timeout=30000)

        drag([(0.2 + 0.01 * i, 0.6) for i in range(11)])
        pg.evaluate("goto(8)")
        pg.keyboard.press("F6")
        pg.evaluate("pushUndo(); keyAt(8).strokes[0].xt += 0.4; changed(); renderAll();")
        pg.evaluate("goto(0)")
        pg.keyboard.press("m")
        pg.keyboard.press("Control+a")
        drag([(0.25 + 0.4 * i / 20, 0.6 - 0.3 * (1 - (2 * i / 20 - 1) ** 2)) for i in range(21)])
        pg.evaluate("goto(3)")  # set the ease in Properties, as the animator would
        pg.select_option('[data-p="ease"]', "ease_in")
        assert pg.evaluate("S.doc.timing") == {"0-8": "ease_in"}
        reached = pg.evaluate("pointAt(fittedPath(S.doc.paths[0]), easedT(0, 8, 4))")
        for key in ("F6", "F7"):
            pg.evaluate("goto(4)")
            pg.keyboard.press(key)
            assert pg.evaluate("S.doc.timing") == {"0-4": "ease_in", "4-8": "ease_in"}, key
            assert pg.evaluate("S.doc.paths.map(p => p.span)") == ["0-4", "4-8"]
            end = pg.evaluate("S.doc.paths[0].points.at(-1)")  # split with the old span's ease
            assert math.hypot(end[0] - reached[0], end[1] - reached[1]) < 1e-9
            pg.keyboard.press("Control+z")
            assert pg.evaluate("S.doc.timing") == {"0-8": "ease_in"}
        pg.evaluate("delete S.doc.timing['0-8']")  # a span with the default ease stays default
        pg.evaluate("goto(4)")
        pg.keyboard.press("F6")
        assert pg.evaluate("S.doc.timing") == {}
        assert errors == []
        browser.close()


def test_draw_on(server, tmp_path):
    """A new stroke set to Draw on (Properties > Appears) is drawn progressively on the
    in-betweens, moving with its motion-path group, through Paint Preview and both exports;
    its trim follows F5, F6 splits and clearing / deleting like motion paths."""
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

        def drag(pts, fit=True):
            pg.mouse.move(*scr(*pts[0]))
            pg.mouse.down()
            for q in pts[1:]:
                pg.mouse.move(*scr(*q), steps=2)
            pg.mouse.up()
            if fit:
                pg.wait_for_function("!pendingFit", timeout=30000)

        drag([(0.2 + 0.01 * i, 0.6 - 0.02 * math.sin(i / 3)) for i in range(11)])  # the body
        pg.evaluate("goto(8)")
        pg.keyboard.press("F6")
        pg.evaluate("pushUndo(); keyAt(8).strokes[0].xt += 0.4; changed(); renderAll();")
        drag([(0.6 + 0.012 * i, 0.55 - 0.004 * i) for i in range(11)])  # a new line beside it
        line = pg.evaluate("[...S.ui.sel][0]")
        body = pg.evaluate(f"keyAt(8).strokes.find(s => s.id !== '{line}').id")
        assert "appears halfway" in pg.locator("#propbody").inner_text()
        pg.select_option('select[data-trim]', "draw_on")
        assert pg.evaluate("S.doc.trims") == [{"span": "0-8", "ids": [line], "mode": "draw_on"}]
        assert "draws on over the in-betweens" in pg.locator("#propbody").inner_text()
        # motion path for the body; the drawing-on line follows the group
        pg.keyboard.press("m")
        pg.keyboard.press("Control+a")
        drag([(0.25 + 0.4 * i / 20, 0.6 - 0.25 * (1 - (2 * i / 20 - 1) ** 2)) for i in range(21)], fit=False)
        assert sorted(pg.evaluate("S.doc.paths[0].ids")) == sorted([line, body])

        pg.select_option("#tl-method", "arc")
        pg.keyboard.press("Control+Enter")
        pg.wait_for_function("!S.ui.busy && Object.keys(S.gen.frames).length === 7", timeout=180000)
        lengths = pg.evaluate(f"""[1,2,3,4,5,6,7].map(f => {{ const s = S.gen.frames[f].strokes.find(x => x.id === '{line}');
            if (!s) return null; const p = Geo.canvasPath(s, S.cfg); let L = 0;
            for (let i = 1; i < p.length; i++) L += Math.hypot((p[i][0]-p[i-1][0]) * aspect(), p[i][1]-p[i-1][1]); return L; }})""")
        drawn = [x for x in lengths if x is not None]
        assert len(drawn) >= 5 and drawn == sorted(drawn), lengths
        assert lengths[-1] is not None
        # it moved with the group: its start keeps the same offset from the group's anchor
        off = pg.evaluate(f"""(() => {{ const at = (st) => {{ const b = st.find(s => s.id === '{body}'), l = st.find(s => s.id === '{line}');
            const c = center(b), p0 = Geo.canvasPath(l, S.cfg)[0]; return [p0[0] - c[0], p0[1] - c[1]]; }};
            return [at(keyAt(8).strokes), at(S.gen.frames[6].strokes)]; }})()""")
        assert math.hypot(off[0][0] - off[1][0], off[0][1] - off[1][1]) < 0.01

        pg.evaluate("goto(5)")  # Paint Preview renders the partial stroke
        pg.keyboard.press("p")
        pg.wait_for_function("[...S.previewCache.values()].some(v => v instanceof Image && v.complete)", timeout=60000)
        pg.keyboard.press("p")
        frames = pg.evaluate("framesForExport()")  # Export Robot Frames
        assert line in [s["id"] for s in frames["frames"][5]["strokes"]]
        with pg.expect_download(timeout=180000) as dl:  # Export FRIDA Plans
            pg.evaluate("cmds.exportPlans()")
        zpath = tmp_path / "plans.zip"
        dl.value.save_as(zpath)
        with zipfile.ZipFile(zpath) as z:
            painting, _ = pickle.loads(z.read("frame_005.pkl"))
        assert len(painting.brush_strokes) == 2

        pg.evaluate("goto(2)")  # F5 / Shift+F5 move the trim's span like timing
        pg.keyboard.press("F5")
        assert pg.evaluate("S.doc.trims[0].span") == "0-9"
        pg.keyboard.press("Shift+F5")
        pg.evaluate("goto(4)")  # F7 inside the span: the trim moves to where the stroke enters
        pg.keyboard.press("F7")
        assert pg.evaluate("S.doc.trims.map(t => t.span)") == ["0-4", "4-8"]
        pg.keyboard.press("Control+z")
        pg.evaluate(f"goto(8); S.ui.sel = new Set(['{line}']); renderAll()")  # delete: leaves the trim
        pg.keyboard.press("Delete")
        assert pg.evaluate("S.doc.trims") == []
        pg.keyboard.press("Control+z")
        pg.keyboard.press("Shift+F6")  # clear the keyframe: the trim goes too
        assert pg.evaluate("S.doc.trims") == []
        assert errors == []
        browser.close()


def test_link_mode(server):
    """Link mode (L): click a stroke, then its ghost, pair after pair; unlinked strokes and
    ghosts without a partner show in the warning color; Properties counts 'n of m linked';
    Ctrl+Z undoes one link; Esc leaves the mode."""
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

        def drag(pts):
            pg.mouse.move(*scr(*pts[0]))
            pg.mouse.down()
            for q in pts[1:]:
                pg.mouse.move(*scr(*q), steps=2)
            pg.mouse.up()
            pg.wait_for_function("!pendingFit", timeout=30000)

        def arc(cx, cy, r, a0, a1, n=16):
            return [(cx + r / 1.25 * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n))
                    for i in range(n + 1)]

        def click_on(key_js, sid):  # middle of a stroke of keyAt(..) / prevKey(..)
            x, y = pg.evaluate(f"toScreen(Geo.canvasPath({key_js}.strokes.find(s => s.id === '{sid}'), S.cfg)[16])")
            pg.mouse.click(st["x"] + x, st["y"] + y)

        def red_at(key_js, sid):  # is the stroke drawn in the warning color on the stage?
            return pg.evaluate(f"""(() => {{ render(); const d = devicePixelRatio || 1;
                const p = toScreen(Geo.canvasPath({key_js}.strokes.find(s => s.id === '{sid}'), S.cfg)[16]);
                const px = ctx.getImageData(Math.round(p[0] * d) - 2, Math.round(p[1] * d) - 2, 5, 5).data; let best = 0;
                for (let i = 0; i < px.length; i += 4) best = Math.max(best, px[i] - Math.max(px[i+1], px[i+2]));
                return best > 60; }})()""")

        def summary():
            return pg.locator("[data-links]").inner_text()

        drag([(0.2 + 0.4 * i / 20, 0.8) for i in range(21)])          # ground
        drag(arc(0.25, 0.4, 0.07, math.pi, 2 * math.pi))               # ball top
        drag(arc(0.25, 0.4, 0.07, 0, math.pi))                         # ball bottom
        old = pg.evaluate("keyAt(0).strokes.map(s => s.id)")
        pg.evaluate("goto(8)")
        pg.keyboard.press("F7")
        pg.keyboard.press("o")  # onion off: nothing links by itself while drawing
        drag([(0.2 + 0.4 * i / 20, 0.8) for i in range(21)])
        drag(arc(0.7, 0.3, 0.07, math.pi, 2 * math.pi))
        drag(arc(0.7, 0.3, 0.07, 0, math.pi))
        new = pg.evaluate("keyAt(8).strokes.map(s => s.id)")
        assert not set(new) & set(old)
        pg.keyboard.press("o")
        pg.keyboard.press("Escape")
        assert "0 of 3" in summary() and "3 ghosts without a partner" in summary()
        assert red_at("keyAt(8)", new[1]) and red_at("prevKey(8)", old[1])  # warning color

        pg.keyboard.press("l")
        assert pg.evaluate("S.ui.tool") == "link"
        click_on("keyAt(8)", new[0])      # the new ground...
        assert pg.evaluate("S.ui.linkFirst") == new[0]
        click_on("prevKey(8)", old[0])    # ...then its ghost, right underneath it
        assert pg.evaluate("keyAt(8).strokes[0].id") == old[0]
        assert "1 of 3" in summary()
        click_on("keyAt(8)", new[1])
        click_on("prevKey(8)", old[1])    # ball top
        assert "2 of 3" in summary()
        assert not red_at("keyAt(8)", old[1])
        # a wrong link takes the ghost from the stroke that had it; Ctrl+Z puts it back
        click_on("keyAt(8)", new[2])
        click_on("prevKey(8)", old[1])
        assert pg.evaluate("keyAt(8).strokes[2].id") == old[1]
        assert "now unlinked" in pg.locator("#toast").inner_text()
        assert "2 of 3" in summary() and "1 ghost without" in summary()  # the ball top lost its link
        assert pg.evaluate("keyAt(8).strokes[1].id") not in old
        pg.keyboard.press("Control+z")
        assert pg.evaluate("keyAt(8).strokes.map(s => s.id)") == [old[0], old[1], new[2]]
        assert pg.evaluate("S.ui.tool") == "link"
        click_on("keyAt(8)", new[2])
        click_on("prevKey(8)", old[2])    # ball bottom
        assert "3 of 3" in summary() and "0 ghosts without a partner" in summary()
        assert not red_at("keyAt(8)", old[2]) and not red_at("prevKey(8)", old[2])
        pg.keyboard.press("Escape")
        assert pg.evaluate("S.ui.tool") == "selection"

        pg.select_option("#tl-method", "linear")  # linked strokes now move instead of popping
        pg.keyboard.press("Control+Enter")
        pg.wait_for_function("!S.ui.busy && Object.keys(S.gen.frames).length === 7", timeout=120000)
        assert pg.evaluate("S.gen.frames[4].strokes.length") == 3
        assert errors == []
        browser.close()


def test_drag_and_drop_reference_and_keyframe_file(server, tmp_path):
    import base64
    import json as _json
    from PIL import Image, ImageDraw
    pngs = []
    for i in range(4):
        im = Image.new("RGB", (320, 180), "white")
        ImageDraw.Draw(im).ellipse([40 + 40 * i, 60, 80 + 40 * i, 100], outline="black", width=3)
        fn = tmp_path / f"drop_{i + 1}.png"
        im.save(fn)
        pngs.append({"name": fn.name, "type": "image/png", "b64": base64.b64encode(fn.read_bytes()).decode()})
    doc = {"format": "frida-inbetween", "version": 1, "keys": [{"frame": 0, "strokes": []}, {"frame": 6, "strokes": []}],
           "edits": [], "timing": {}}
    jsonfile = {"name": "keys.json", "type": "application/json",
                "b64": base64.b64encode(_json.dumps(doc).encode()).decode()}
    drop_js = """async (files) => {
        const dt = new DataTransfer();
        for (const f of files) {
            const bytes = Uint8Array.from(atob(f.b64), c => c.charCodeAt(0));
            dt.items.add(new File([bytes], f.name, {type: f.type}));
        }
        const stage = document.getElementById('stage');
        stage.dispatchEvent(new DragEvent('dragenter', {dataTransfer: dt, bubbles: true}));
        const shown = !document.getElementById('dropzone').hidden;
        stage.dispatchEvent(new DragEvent('dragover', {dataTransfer: dt, bubbles: true}));
        stage.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}));
        return shown && document.getElementById('dropzone').hidden;
    }"""
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
        assert pg.locator(".refbox").count() == 1          # the import box shows while there's no reference
        assert pg.evaluate(drop_js, pngs[::-1])             # overlay appears on drag, hides on drop
        pg.wait_for_function("S.ref.images.length === 4 && !S.ui.busy", timeout=30000)
        assert pg.locator(".refbox").count() == 0           # replaced by the reference controls
        assert pg.evaluate(drop_js, [jsonfile])             # a keyframe file opens instead
        pg.wait_for_function("S.doc.keys.length === 2", timeout=10000)
        assert pg.evaluate("S.doc.keys[1].frame") == 6
        assert errors == []
        browser.close()


def test_corner_curve_handles(server):
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
        # one stroke with two clear corners, set directly so the test doesn't depend on the fitter
        pg.evaluate("""() => {
            const s = {id: 's1', order: 0, color: [0.05, 0.05, 0.05], xt: 0.3, yt: 0.4, a: 0,
                       vertices: [[0, 0], [0.06, 0], [0.06, 0.06], [0.12, 0.06]], kappa: [0.1, 0.1], z: [0.6, 0.6, 0.6, 0.6]};
            S.doc.keys[0].strokes.push(s); S.ui.sel = new Set(['s1']); S.ui.tool = 'subselection'; renderAll();
        }""")
        st = pg.locator("#canvas").bounding_box()
        hs = pg.evaluate("cornerHandleScreen(S.doc.keys[0].strokes[0]).map(c => ({P: c.P, a: c.a, b: c.b}))")
        assert len(hs) == 2
        h = hs[0]
        # drag the handle on the first edge outward along that edge: the first corner gets rounder
        dx, dy = h["a"][0] - h["P"][0], h["a"][1] - h["P"][1]
        L = (dx * dx + dy * dy) ** 0.5
        pg.mouse.move(st["x"] + h["a"][0], st["y"] + h["a"][1])
        pg.mouse.down()
        pg.mouse.move(st["x"] + h["P"][0] + dx / L * (L + 25), st["y"] + h["P"][1] + dy / L * (L + 25), steps=5)
        pg.mouse.up()
        k = pg.evaluate("S.doc.keys[0].strokes[0].kappa")
        assert k[0] > 0.15 and abs(k[1] - 0.1) < 1e-9   # only that corner changed
        # the two handles mirror: same distance from the corner along each edge
        a2 = pg.evaluate("cornerHandleScreen(S.doc.keys[0].strokes[0])[0]")
        # in stroke space (FRIDA's calibration scales x and y slightly differently on paper)
        da, db = pg.evaluate("""() => { const s = S.doc.keys[0].strokes[0], c = cornerHandleScreen(s)[0];
            const A = Geo.toLocal(toProp(c.a[0], c.a[1]), s, S.cfg), B = Geo.toLocal(toProp(c.b[0], c.b[1]), s, S.cfg);
            const P = c.h.P; return [Math.hypot(A[0] - P[0], A[1] - P[1]), Math.hypot(B[0] - P[0], B[1] - P[1])]; }""")
        assert abs(da - db) < 1e-4  # meters
        # drag the other handle all the way into the corner: sharp
        pg.mouse.move(st["x"] + a2["b"][0], st["y"] + a2["b"][1])
        pg.mouse.down()
        pg.mouse.move(st["x"] + a2["P"][0], st["y"] + a2["P"][1], steps=5)
        pg.mouse.up()
        assert pg.evaluate("S.doc.keys[0].strokes[0].kappa[0]") < 0.02
        pg.keyboard.press("Control+z")
        assert abs(pg.evaluate("S.doc.keys[0].strokes[0].kappa[0]") - k[0]) < 1e-9
        assert errors == []
        browser.close()



def _editor(p, server):
    browser = p.chromium.launch()
    pg = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server)
    pg.wait_for_function("S.cfg && S.doc")
    st = pg.locator("#canvas").bounding_box()

    def scr(px, py):
        v = pg.evaluate("[S.view.ox, S.view.oy, S.view.scale, S.cfg.canvas_width_m/S.cfg.canvas_height_m]")
        return st["x"] + v[0] + px * v[2] * v[3], st["y"] + v[1] + py * v[2]
    return browser, pg, errors, scr


def test_pen_tool_draws_fitted_strokes(server):
    with playwright.sync_playwright() as p:
        try:
            browser, pg, errors, scr = _editor(p, server)
        except Exception as e:
            pytest.skip(f"no Chromium: {e}")
        pg.keyboard.press("n")
        assert pg.evaluate("S.ui.tool") == "pen"
        pg.mouse.click(*scr(0.2, 0.6))                  # sharp start
        x, y = scr(0.4, 0.3)                            # smooth middle: drag out a handle
        pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x + 60, y, steps=4); pg.mouse.up()
        pg.mouse.click(*scr(0.6, 0.6))                  # sharp end
        assert pg.evaluate("S.ui.bez.anchors.length") == 3
        assert pg.evaluate("S.ui.bez.anchors[1].hout !== null && S.ui.bez.anchors[1].hin !== null")
        pg.keyboard.press("Backspace")                  # removes the last point
        assert pg.evaluate("S.ui.bez.anchors.length") == 2
        pg.mouse.click(*scr(0.6, 0.6))
        pg.keyboard.press("Enter")
        pg.wait_for_function("!pendingFit && S.doc.keys[0].strokes.length > 0", timeout=60000)
        assert pg.evaluate("S.ui.bez") is None
        # the fitted strokes pass through the start, the curve's top and the end
        d = pg.evaluate("""(pts) => {
            const path = S.doc.keys[0].strokes.flatMap(s => Geo.canvasPath(s, S.cfg));
            const a = S.cfg.canvas_width_m / S.cfg.canvas_height_m;
            return pts.map(q => Math.min(...path.map(p => Math.hypot((p[0] - q[0]) * a, p[1] - q[1]))));
        }""", [[0.2, 0.6], [0.4, 0.3], [0.6, 0.6]])
        assert max(d) < 0.01, d                         # within 1% of the paper height
        pg.keyboard.press("Control+z")                  # one undo removes the pen line
        assert pg.evaluate("S.doc.keys[0].strokes.length") == 0
        # Esc cancels a path in progress
        pg.mouse.click(*scr(0.2, 0.2)); pg.mouse.click(*scr(0.3, 0.2))
        pg.keyboard.press("Escape")
        assert pg.evaluate("S.ui.bez") is None and pg.evaluate("S.doc.keys[0].strokes.length") == 0
        assert errors == []
        browser.close()


def test_bezier_selection_selects_touched_strokes(server):
    with playwright.sync_playwright() as p:
        try:
            browser, pg, errors, scr = _editor(p, server)
        except Exception as e:
            pytest.skip(f"no Chromium: {e}")
        pg.evaluate("""() => {
            const mk = (id, xt, yt) => ({id, order: +id.slice(1), color: [0.05, 0.05, 0.05], xt, yt, a: 0,
                vertices: [[0, 0], [0.02, 0], [0.04, 0], [0.06, 0]], kappa: [0.5, 0.5], z: [0.6, 0.6, 0.6, 0.6]});
            S.doc.keys[0].strokes.push(mk('s1', 0.2, 0.3), mk('s2', 0.2, 0.6), mk('s3', 0.6, 0.3)); renderAll();
        }""")
        pg.keyboard.press("q")
        # an outline around the left half of s1 only (touching is enough)
        for q in [(0.15, 0.25), (0.25, 0.25), (0.25, 0.35)]:
            pg.mouse.click(*scr(*q))
        x, y = scr(0.15, 0.35)
        pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x - 20, y + 10, steps=3); pg.mouse.up()
        pg.mouse.click(*scr(0.15, 0.25))                # click the first point: closes and selects
        assert sorted(pg.evaluate("[...S.ui.sel]")) == ["s1"]
        # Shift on the first click adds; Enter closes
        pg.keyboard.down("Shift"); pg.mouse.click(*scr(0.55, 0.2)); pg.keyboard.up("Shift")
        for q in [(0.9, 0.2), (0.9, 0.4), (0.55, 0.4)]:
            pg.mouse.click(*scr(*q))
        pg.keyboard.press("Enter")
        assert sorted(pg.evaluate("[...S.ui.sel]")) == ["s1", "s3"]
        assert pg.evaluate("S.doc.keys[0].strokes.length") == 3   # selecting never draws
        assert errors == []
        browser.close()
