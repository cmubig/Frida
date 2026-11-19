"""Local server for the in-betweening editor.

Run from src/:

    python -m inbetween.app                      # opens http://localhost:8765
    python -m inbetween.app --cache_dir caches/vae_brush_final --port 9000

The editor (inbetween/editor/) runs in the browser; this server gives it the
stroke model's calibration, fits freehand marks to robot strokes, generates
in-betweens with the pipeline and renders exact stroke-model previews. It
only listens on localhost and uses the Python standard library.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import torch

from .fit import fit_mark, fit_stroke, visible_z_min
from .interpolate import interpolate
from .metrics import popping
from .refine import RefineConfig, refine
from .render import RESPONSES, Renderer
from .strokes import Stroke, frames_to_dict, sequence_from_dict

EDITOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor")
MAX_BODY = 20 * 1024 * 1024


class State:
    def __init__(self, cache_dir: str, response: str, ink: bool):
        self.cache_dir = cache_dir
        self.ink = ink
        self.lock = threading.Lock()
        self.renderers = {}
        self.response = response

    def renderer(self, response=None, height=256) -> Renderer:
        key = (response or self.response, height)
        if key not in self.renderers:
            self.renderers[key] = Renderer(self.cache_dir, render_height=height, response=key[0])
        return self.renderers[key]

    def config(self) -> dict:
        r = self.renderer()
        m = r.model
        return {
            "cache_dir": self.cache_dir,
            "calibrated": r.calibrated,
            "response": self.response,
            "responses": list(RESPONSES),
            "ink": self.ink,
            "canvas_width_m": r.settings.CANVAS_WIDTH_M,
            "canvas_height_m": r.settings.CANVAS_HEIGHT_M,
            "min_stroke_z": r.settings.MIN_STROKE_Z,
            "max_stroke_length_m": r.settings.MAX_STROKE_LENGTH,
            "visible_z_min": visible_z_min(r),
            "x_m": float(m.x_m), "y_m": float(m.y_m), "x_b": float(m.x_b), "y_b": float(m.y_b),
            "thick_mult": float(m.renderer.thick_mult), "thick_bias": float(m.renderer.thick_bias),
            "dark_exp": float(m.renderer.dark_exp),
        }


def make_handler(state: State):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=EDITOR_DIR, **kw)

        def log_message(self, fmt, *args):  # quieter console
            if self.path.startswith("/api/"):
                return
            super().log_message(fmt, *args)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/config":
                with state.lock:
                    return self._json(200, state.config())
            return super().do_GET()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self._json(413, {"error": "request too large"})
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                with state.lock:
                    out = self.route(req)
                if out is None:
                    return self._json(404, {"error": "unknown endpoint"})
                return self._json(200, out)
            except Exception as e:  # report errors to the editor instead of dropping the connection
                traceback.print_exc()
                return self._json(400, {"error": str(e)})

        def route(self, req):
            if self.path == "/api/fit":
                color = req.get("color", [0.05, 0.05, 0.05])
                if req.get("split", True):
                    pieces = fit_mark(req["points"], req.get("pressure", []), state.renderer(),
                                      max_err_mm=float(req.get("max_err_mm", 1.0)), color=color)
                else:
                    pieces = [fit_stroke(req["points"], req.get("pressure", []), state.renderer(), color=color)]
                return {"strokes": [{"stroke": s.to_dict(), "error_mm": e} for s, e in pieces],
                        "stroke": pieces[0][0].to_dict(), "error_mm": max(e for _, e in pieces)}
            if self.path == "/api/inbetween":
                seq = sequence_from_dict(req["doc"])
                r = state.renderer(req.get("response"))
                method = req.get("method", "refined")
                frames = interpolate(seq, "linear" if method == "linear" else "arc", r)
                if method == "refined":
                    frames, _ = refine(seq, frames, r, RefineConfig(verbose=False), ink=state.ink)
                summary = popping(seq, frames, r)["summary_mean_over_strokes"]
                return {**frames_to_dict(frames, {"method": method}), "metrics": summary}
            if self.path == "/api/render":
                r = state.renderer(req.get("response"), int(req.get("height", 384)))
                strokes = [Stroke.from_dict(d) for d in req.get("strokes", [])]
                img = r.render_frame(strokes)
                return {"png": _png_b64(img)}
            return None

    return Handler


def _png_b64(img: torch.Tensor) -> str:
    from PIL import Image
    im = Image.fromarray((img.cpu().numpy() * 255).astype("uint8"))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", default="caches/vae_sharpie_final")
    ap.add_argument("--response", choices=RESPONSES, default="model")
    ap.add_argument("--paint", action="store_true", help="paint setup instead of ink/marker")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no_browser", action="store_true")
    args = ap.parse_args(argv)

    state = State(args.cache_dir, args.response, ink=not args.paint)
    state.config()  # load the model once up front so errors show immediately
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    url = f"http://localhost:{args.port}/"
    print(f"[inbetween] editor running at {url}  (stroke model: {args.cache_dir}; Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
