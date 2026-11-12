"""Render README.md to README-preview.html (GitHub-like), to read it with its pictures.

    uv run --with markdown tools/preview_readme.py
"""
import base64
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
body = markdown.markdown((ROOT / "README.md").read_text(encoding="utf-8"), extensions=["fenced_code", "tables"])
# The pictures go inside the page, so it can be opened or sent on its own.
for img in sorted((ROOT / "docs" / "img").glob("*.jpg")):
    data = base64.b64encode(img.read_bytes()).decode()
    for prefix in ("docs/img/", "https://raw.githubusercontent.com/aynader/cacography/main/docs/img/"):
        body = body.replace(f'src="{prefix}{img.name}"', f'src="data:image/jpeg;base64,{data}"')
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>cacography README</title>
<style>
body {{ max-width: 880px; margin: 40px auto; padding: 0 16px; font: 16px/1.6 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2328; }}
h1, h2 {{ border-bottom: 1px solid #d1d9e0; padding-bottom: .3em; }}
img {{ max-width: 100%; margin: 6px 0 18px; }}
code {{ background: #eff1f3; padding: .15em .35em; border-radius: 6px; font-size: 85%; }}
pre {{ background: #f6f8fa; padding: 14px; border-radius: 6px; overflow: auto; }}
pre code {{ background: none; padding: 0; }}
a {{ color: #0969da; }}
</style></head><body>{body}</body></html>"""
(ROOT / "README-preview.html").write_text(html, encoding="utf-8")
print("wrote README-preview.html")
