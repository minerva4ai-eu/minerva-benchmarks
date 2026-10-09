"""Always-on dashboard server (standard library only).

    python serve.py                                  # http://0.0.0.0:8080
    python serve.py --host 127.0.0.1 --port 8080 --poll 30

All charts are computed in each visitor's browser, so the server only hands out one
cached, compressed file and any number of simultaneous users is cheap. The page is rebuilt
automatically when a CSV in data/, dashboard/defaults.json or dashboard/template.html changes;
if a rebuild fails the previous version keeps being served.

Endpoints: /  (dashboard), /healthz  (status for monitoring), /vendor/plotly.min.js
"""
import argparse
import gzip
import hashlib
import logging
import re
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import build_dashboard

HERE = Path(__file__).resolve().parent
DASH = HERE / "dashboard"
VENDOR_FILE = DASH / "vendor" / "plotly.min.js"
WATCHED = [HERE / "data", HERE / "data" / "training", DASH / "template.html", DASH / "defaults.json", DASH / "assets" / "minerva-logo.png"]
CDN = re.compile(r"https://cdn\.jsdelivr\.net/npm/plotly\.js-dist-min@[\d.]+/plotly\.min\.js")
log = logging.getLogger("minerva")
page, vendor = {}, {}


def signature():
    files = [f for p in WATCHED for f in (sorted(p.glob("*.csv")) if p.is_dir() else [p])]
    return tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files if f.exists())


def rebuild():
    global page
    html, _ = build_dashboard.build_html()
    if vendor:  # serve Plotly ourselves so browsers do not need internet access
        html = CDN.sub("/vendor/plotly.min.js", html)
    raw = html.encode("utf-8")
    page = {"raw": raw, "gz": gzip.compress(raw, 6), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "etag": '"%s"' % hashlib.sha1(raw).hexdigest()[:16]}
    log.info("Dashboard ready (%d KB, %d KB gzipped)", len(raw) // 1024, len(page["gz"]) // 1024)


def watch(poll, stop):
    last = signature()
    while not stop.wait(poll):
        now = signature()
        if now == last:
            continue
        last = now  # a half-copied file changes again when finished, which triggers another rebuild
        log.info("Change detected, rebuilding")
        try:
            rebuild()
        except Exception:
            log.exception("Rebuild failed; still serving the previous version")


class Handler(BaseHTTPRequestHandler):
    server_version = "MinervaDashboard"

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    def do_GET(self):
        self.respond(True)

    def do_HEAD(self):
        self.respond(False)

    def respond(self, body):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self.send_data(page["raw"], "text/html; charset=utf-8", body, page["gz"], page["etag"])
        elif path == "/healthz":
            self.send_data(f"ok built={page['built']}\n".encode(), "text/plain", body)
        elif path == "/vendor/plotly.min.js" and vendor:
            self.send_data(vendor["raw"], "application/javascript", body, vendor["gz"],
                           vendor["etag"], "public, max-age=86400")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def send_data(self, raw, ctype, body, gz=None, etag=None, cache="no-cache"):
        use_gz = gz is not None and "gzip" in self.headers.get("Accept-Encoding", "")
        tag = etag[:-1] + ("-gz" if use_gz else "") + '"' if etag else None
        if tag and self.headers.get("If-None-Match") == tag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", tag)
            self.end_headers()
            return
        data = gz if use_gz else raw
        self.send_response(HTTPStatus.OK)
        for k, v in [("Content-Type", ctype), ("Content-Length", len(data)), ("Cache-Control", cache),
                     ("Vary", "Accept-Encoding"), ("X-Content-Type-Options", "nosniff")]:
            self.send_header(k, str(v))
        if use_gz:
            self.send_header("Content-Encoding", "gzip")
        if tag:
            self.send_header("ETag", tag)
        self.end_headers()
        if body:
            self.wfile.write(data)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def handle_error(self, request, client_address):  # clients closing tabs mid-download are normal
        log.debug("Connection error from %s", client_address, exc_info=True)


def main():
    ap = argparse.ArgumentParser(description="Serve the Minerva benchmarks dashboard.")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable by other machines; 127.0.0.1 = this machine only")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--poll", type=float, default=30, help="seconds between checks for changed data")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if VENDOR_FILE.exists():
        raw = VENDOR_FILE.read_bytes()
        vendor.update(raw=raw, gz=gzip.compress(raw, 6), etag='"%s"' % hashlib.sha1(raw).hexdigest()[:16])
    else:
        log.info("dashboard/vendor/plotly.min.js not found: browsers will load Plotly from the CDN "
                 "(run `python fetch_plotly.py` for offline use)")
    rebuild()

    stop = threading.Event()
    threading.Thread(target=watch, args=(args.poll, stop), daemon=True).start()
    httpd = Server((args.host, args.port), Handler)
    quit_ = lambda *_: threading.Thread(target=httpd.shutdown).start()
    signal.signal(signal.SIGTERM, quit_)
    signal.signal(signal.SIGINT, quit_)
    log.info("Serving on http://%s:%d (Ctrl+C to stop)", args.host, args.port)
    httpd.serve_forever()
    stop.set()
    httpd.server_close()
    log.info("Stopped")


if __name__ == "__main__":
    main()
