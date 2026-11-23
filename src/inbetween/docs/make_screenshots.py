"""Screenshots for docs/guide.md, taken from the real editor with Playwright.

Run from src/ (needs Playwright + Chromium; the drawings are made here, no
animator's artwork):

    python inbetween/docs/make_screenshots.py inbetween/docs/img /tmp/guide [section ...]

sections: keys trace ref path limits trim (default: all). The draw-on
filmstrip (img/22-draw-on-filmstrip.png) is rendered from the frames this
writes to <tmp>/trim_frames.json with Renderer.render_frame.
"""
import json, math, os, sys, threading, zipfile, io
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
from inbetween.app import State, make_handler

OUT = sys.argv[1]
TMP = sys.argv[2]
only = set(sys.argv[3:])
srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(State("caches/vae_sharpie_final", "model", ink=True)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_address[1]}/"
W, H = 1440, 900


class Ed:
    def __init__(self, browser):
        self.pg = browser.new_page(viewport={"width": W, "height": H}, accept_downloads=True)
        self.errors = []
        self.pg.on("pageerror", lambda e: self.errors.append(str(e)))
        self.pg.goto(URL)
        self.pg.wait_for_function("S.cfg && S.doc")
        self.st = self.pg.locator("#canvas").bounding_box()

    def ev(self, js):
        return self.pg.evaluate(js)

    def scr(self, px, py):
        v = self.ev("[S.view.ox, S.view.oy, S.view.scale, S.cfg.canvas_width_m/S.cfg.canvas_height_m]")
        return self.st["x"] + v[0] + px * v[2] * v[3], self.st["y"] + v[1] + py * v[2]

    def down(self, pts):
        self.pg.mouse.move(*self.scr(*pts[0]))
        self.pg.mouse.down()
        for q in pts[1:]:
            self.pg.mouse.move(*self.scr(*q), steps=2)

    def up(self, fit=True):
        self.pg.mouse.up()
        if fit:
            self.pg.wait_for_function("!pendingFit", timeout=30000)

    def drag(self, pts, fit=True):
        self.down(pts)
        self.up(fit)

    def key(self, k):
        self.pg.keyboard.press(k)

    def notoast(self):
        self.ev("document.getElementById('toast').classList.remove('show')")
        self.pg.wait_for_timeout(450)  # let the fade-out finish

    def shot(self, name, toast=False, clip=None):
        if not toast:
            self.notoast()
        else:
            self.pg.wait_for_timeout(450)  # let the fade-in finish
        self.ev("render(); renderTimeline()")
        self.pg.screenshot(path=os.path.join(OUT, name), clip=clip)
        print("wrote", name)

    def generate(self, method="refined", n=None):
        self.pg.select_option("#tl-method", method)
        self.key("Control+Enter")
        self.pg.wait_for_function("!S.ui.busy && !S.gen.stale && Object.keys(S.gen.frames).length > 0", timeout=600000)

    def click_stroke_mid(self, sid, shift=False):
        x, y = self.ev(f"toScreen(Geo.canvasPath(content(S.ui.frame).strokes.find(s => s.id === '{sid}'), S.cfg)[16])")
        if shift:
            self.pg.keyboard.down("Shift")
        self.pg.mouse.click(self.st["x"] + x, self.st["y"] + y)
        if shift:
            self.pg.keyboard.up("Shift")

    def move_stroke(self, sid, dx, dy, steps=10):
        x, y = self.ev(f"toScreen(Geo.canvasPath(content(S.ui.frame).strokes.find(s => s.id === '{sid}'), S.cfg)[16])")
        x, y = self.st["x"] + x, self.st["y"] + y
        self.pg.mouse.move(x, y)
        self.pg.mouse.down()
        for i in range(1, steps + 1):
            self.pg.mouse.move(x + dx * i / steps, y + dy * i / steps)
        self.pg.mouse.up()


def arc(cx, cy, r, a0, a1, n=16):  # a circle on the paper (x scaled for the paper's aspect)
    return [(cx + r / 1.25 * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def ball(ed, cx, cy, r=0.07):
    ed.drag(arc(cx, cy, r, math.pi, 2 * math.pi))   # top half
    ed.drag(arc(cx, cy, r, 0, math.pi))             # bottom half


GROUND = [(0.12 + 0.52 * i / 24, 0.82) for i in range(25)]  # 132 mm: one stroke reaches 15 cm at most

with sync_playwright() as p:
    browser = p.chromium.launch()
    want = lambda s: not only or s in only

    # ---------------------------------------------------------------- 1. draw keys
    if want("keys"):
        ed = Ed(browser)
        ed.drag(GROUND)
        ball(ed, 0.25, 0.42)
        ed.down(arc(0.25, 0.42, 0.025, math.pi * 1.1, math.pi * 1.9, 8))  # a highlight, caught mid-stroke
        ed.shot("01-brush.png")
        ed.up()
        ed.drag([(0.62 + 0.03 * i, 0.82 - (0.07 if i % 2 else 0)) for i in range(9)])  # grass zigzag: too complex
        ed.shot("02-split.png", toast=True)
        ed.key("Control+z")
        ed.key("v")
        x0, y0 = ed.scr(0.12, 0.3); x1, y1 = ed.scr(0.4, 0.55)
        ed.pg.mouse.move(x0, y0); ed.pg.mouse.down(); ed.pg.mouse.move(x1, y1, steps=8)
        ed.shot("03-marquee.png")
        ed.pg.mouse.up()
        ed.shot("04-selected.png")
        ed.ev("goto(8)")
        ed.key("F6")
        ed.key("v"); ed.key("Control+a")
        ids = ed.ev("keyAt(8).strokes.filter(s => s.id !== 's1').map(s => s.id)")
        ed.ev(f"S.ui.sel = new Set({json.dumps(ids)}); renderAll()")
        ed.move_stroke(ids[0], 330, -40)
        ed.ev(f"S.ui.sel = new Set({json.dumps(ids)}); renderAll()")
        ed.shot("05-f6-keyframe.png")
        ed.key("Escape")
        ed.key("a")
        ed.click_stroke_mid("s2")
        x, y = ed.ev("toScreen(center(keyAt(8).strokes.find(s => s.id === 's2')))")
        ed.ev(f"zoomAt(3, {x}, {y})")
        ed.shot("06-subselection.png")
        ed.ev("zoomFit(); render()")
        assert not ed.errors, ed.errors

    # ---------------------------------------------------------------- 2. trace + 4/5 generate, pins
    if want("trace"):
        ed = Ed(browser)
        ed.drag(GROUND)
        ball(ed, 0.22, 0.45)
        ed.ev("goto(8)")
        ed.key("F7")
        ed.shot("07-f7-blank.png", toast=True)
        g2 = [(x, 0.822) for x, _ in GROUND]
        ed.down(g2[:14])
        ed.shot("08-trace-suggest.png")
        ed.pg.mouse.move(*ed.scr(*g2[-1]), steps=6)
        ed.up()
        ed.pg.keyboard.down("Alt")
        ed.pg.mouse.click(*ed.scr(0.22, 0.38))  # choose the ball top's ghost
        ed.pg.keyboard.up("Alt")
        ed.shot("09-alt-click.png", toast=True)
        ed.drag(arc(0.7, 0.3, 0.07, math.pi, 2 * math.pi))
        ed.pg.keyboard.down("Alt"); ed.pg.mouse.click(*ed.scr(0.22, 0.52)); ed.pg.keyboard.up("Alt")
        ed.drag(arc(0.7, 0.3, 0.07, 0, math.pi))
        ed.ev("S.ui.tool = 'selection'; S.ui.sel = new Set(['s2']); renderAll()")
        ed.ev("[...document.querySelectorAll('#propbody h4')].find(h => h.textContent === 'Correspondence').scrollIntoView()")
        ed.shot("10-correspondence.png")
        ed.ev("goto(4)")
        ed.shot("11-not-created.png")
        ed.generate("refined")
        ed.ev("goto(4)")
        ed.shot("12-inbetweens.png")
        ed.key("p")
        ed.pg.wait_for_function("[...S.previewCache.values()].some(v => v instanceof Image && v.complete)", timeout=60000)
        ed.pg.wait_for_timeout(300)
        ed.shot("13-paint-preview.png")
        ed.key("p")
        ed.key("v")
        ed.move_stroke("s2", 0, -60)
        ed.click_stroke_mid("s2")
        ed.shot("14-pin.png")
        assert not ed.errors, ed.errors

    # ---------------------------------------------------------------- 3. reference layer
    if want("ref"):
        from PIL import Image, ImageDraw
        files = []
        for i in range(9):  # a ball falling and squashing, as an animator's rough frames
            im = Image.new("RGB", (1000, 800), "white")
            d = ImageDraw.Draw(im)
            y = 120 + 440 * (i / 8) ** 2
            sq = 1 + (0.35 if i == 8 else 0)
            d.ellipse([500 - 70 * sq, y - 70 / sq, 500 + 70 * sq, y + 70 / sq + (0 if i < 8 else 20)], outline=(40, 40, 40), width=7)
            d.line([230, 660, 770, 660], fill=(40, 40, 40), width=7)
            fn = os.path.join(TMP, f"rough_{i + 1:02d}.png"); im.save(fn); files.append(fn)
        ed = Ed(browser)
        ed.pg.set_input_files("#refinput", files)
        ed.pg.wait_for_function("S.ref.images.length === 9 && !S.ui.busy", timeout=30000)
        ed.ev("S.ref.opacity = 0.45; goto(0)")
        ed.drag([(0.23 + 0.54 * i / 24, 0.825) for i in range(25)])
        ed.drag(arc(0.5, 0.15, 0.0875, math.pi, 2 * math.pi))
        ed.ev("S.ui.sel.clear(); renderAll()")
        ed.shot("15-reference.png")
        assert not ed.errors, ed.errors

    # ---------------------------------------------------------------- 6. motion paths + 8. export + limits
    if want("path"):
        ed = Ed(browser)
        ed.drag(GROUND)
        ball(ed, 0.2, 0.72, 0.07)
        ed.ev("goto(10)")
        ed.key("F6")
        ed.key("v")
        ed.ev("S.ui.sel = new Set(['s2', 's3']); renderAll()")
        ed.move_stroke("s2", 400, 0)
        ed.ev("goto(0)")
        ed.key("m")
        ed.click_stroke_mid("s2")
        ed.click_stroke_mid("s3", shift=True)
        ed.shot("16-motion-path-select.png")
        a0 = ed.ev("center(keyAt(0).strokes.find(s => s.id === 's2'))")
        a1 = ed.ev("center(keyAt(10).strokes.find(s => s.id === 's2'))")
        pts = [(a0[0] + (a1[0] - a0[0]) * i / 24, a0[1] + (a1[1] - a0[1]) * i / 24 - 0.42 * (1 - (2 * i / 24 - 1) ** 2)) for i in range(25)]
        ed.down(pts)
        ed.shot("17-motion-path-draw.png")
        ed.up(fit=False)
        ed.ev("S.doc.timing['0-10'] = 'ease_in_out'; renderAll()")
        ed.shot("18-motion-path-dots.png")
        ed.generate("refined")
        ed.ev("goto(3); S.ui.sel = new Set(['s2','s3']); renderAll()")
        ed.shot("19-motion-path-inbetween.png")
        # export
        ed.ev("S.ui.sel.clear(); renderAll()")
        ed.pg.locator('.menu[data-menu="file"]').click()
        ed.shot("24-file-menu.png")
        ed.ev("closeMenus()")
        with ed.pg.expect_download(timeout=300000) as dl:
            ed.ev("cmds.exportPlans()")
        z = zipfile.ZipFile(io.BytesIO(open(dl.value.path(), "rb").read()))
        open(os.path.join(OUT, "25-frida-plan-frame4.png"), "wb").write(z.read("frame_003.png"))
        print("zip:", sorted(z.namelist())[:4], "...", len(z.namelist()), "files")
        ed.shot("26-export-toast.png", toast=True)
        # limits: a path off the paper
        ed.ev("goto(0)")
        ed.key("m")
        ed.click_stroke_mid("s2"); ed.click_stroke_mid("s3", shift=True)
        ed.drag([(0.2 + 0.52 * i / 24, 0.72 - 1.0 * (1 - (2 * i / 24 - 1) ** 2)) for i in range(25)], fit=False)
        ed.shot("28-off-paper.png", toast=True)
        assert not ed.errors, ed.errors

    # ---------------------------------------------------------------- limits
    if want("limits"):
        ed = Ed(browser)
        ed.drag([(0.3 + 0.3 * i / 12, 0.4) for i in range(13)])
        ed.drag([(0.3 + 0.3 * i / 12, 0.5) for i in range(13)])
        ed.key("v")
        ed.click_stroke_mid("s2")
        for i in range(4):
            sl = ed.pg.locator(f'input[data-p="z{i}"]')
            sl.fill("0.15")
            sl.dispatch_event("input")
            sl.dispatch_event("change")
        ed.key("Escape")
        ed.ev("zoomAt(1.8)")
        ed.shot("27-too-light.png")
        ed.ev("zoomFit(); render()")
        ed.key("Control+a"); ed.key("Delete")
        ed.key("b")
        long = [(0.08 + 0.84 * i / 30, 0.6) for i in range(31)]  # 213 mm
        ed.drag(long)
        ed.key("v"); ed.key("Control+a")
        print("long mark ->", ed.ev("keyAt(0).strokes.length"), "strokes")
        ed.shot("30-long-mark.png")
        assert not ed.errors, ed.errors

    # ---------------------------------------------------------------- 7. draw on
    if want("trim"):
        ed = Ed(browser)
        A = lambda a0, a1, n=14: [(0.27 + 0.11 * math.cos(a0 + (a1 - a0) * i / n), 0.66 - 0.17 * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
        ed.drag(A(math.pi, math.pi / 2)); ed.drag(A(math.pi / 2, 0))
        ed.drag([(0.15 + 0.24 * i / 12, 0.665) for i in range(13)])
        ed.ev("goto(10)"); ed.key("F6")
        ed.key("v"); ed.key("Control+a")
        h = ed.ev("rotateHandlePos(selected())"); b0 = ed.ev("selBox(selected())")
        st = ed.st
        cx, cy = st["x"] + (b0[0] + b0[2]) / 2, st["y"] + (b0[1] + b0[3]) / 2
        r0 = math.hypot(h[0] + st["x"] - cx, h[1] + st["y"] - cy)
        ed.pg.mouse.move(st["x"] + h[0], st["y"] + h[1]); ed.pg.mouse.down()
        for i in range(1, 11):
            ang = -math.pi / 2 + math.radians(28) * i / 10
            ed.pg.mouse.move(cx + r0 * math.cos(ang), cy + r0 * math.sin(ang))
        ed.pg.mouse.up()
        ed.move_stroke("s3", 300, -80)
        pts = ed.ev("""(() => { const P = [...Geo.canvasPath(keyAt(10).strokes[0], S.cfg), ...Geo.canvasPath(keyAt(10).strokes[1], S.cfg).slice(1)];
            const c = center(keyAt(10).strokes[2]); return P.map(p => { const dx = (p[0]-c[0]) * aspect(), dy = p[1]-c[1], n = Math.hypot(dx, dy) || 1;
            return [p[0] + 0.035 * dx / n / aspect(), p[1] + 0.035 * dy / n]; }).filter((_, i) => i % 2 === 0); })()""")
        ed.key("b"); ed.drag([tuple(q) for q in pts])
        line = ed.ev("[...S.ui.sel][0]")
        ed.key("v"); ed.ev(f"S.ui.sel = new Set(['{line}']); S.ui.onion = false; renderAll()")
        ed.pg.select_option("select[data-trim]", "draw_on")
        ed.pg.locator("select[data-trim]").scroll_into_view_if_needed()
        ed.shot("20-draw-on-setting.png", toast=True)
        ed.generate("refined")
        ed.ev("goto(5); S.ui.sel.clear(); renderAll()")
        ed.shot("21-draw-on-inbetween.png")
        json.dump(ed.ev("framesForExport()"), open(os.path.join(TMP, "trim_frames.json"), "w"))
        assert not ed.errors, ed.errors

    browser.close()
