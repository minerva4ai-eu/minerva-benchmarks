"""Download Plotly.js next to the dashboard so it works without internet access.

    python fetch_plotly.py

Writes dashboard/vendor/plotly.min.js (the version pinned in dashboard/template.html).
serve.py uses it automatically when present.
"""
import io
import re
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    template = (HERE / "dashboard" / "template.html").read_text(encoding="utf-8")
    version = re.search(r"plotly\.js-dist-min@([\d.]+)/", template).group(1)
    url = f"https://registry.npmjs.org/plotly.js-dist-min/-/plotly.js-dist-min-{version}.tgz"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        js = tar.extractfile("package/plotly.min.js").read()
    out = HERE / "dashboard" / "vendor" / "plotly.min.js"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(js)
    print(f"Saved Plotly {version} to {out.relative_to(HERE)} ({len(js) // 1024} KB)")


if __name__ == "__main__":
    main()
