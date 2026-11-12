# cacography

Typeset a LaTeX document, get back a phone photo of it handwritten. LaTeX does the layout,
equations and structure; a handwriting model trained on real pen recordings writes every
page on procedurally made paper, which is then photographed on a desk. The PDF keeps the
typeset text invisibly underneath, so it stays searchable.

This bundle is the LaTeX side (`cacography.sty`, `cacography.lua`; needs LuaLaTeX). The
pages are written by a separate Python program, `cacography` on PyPI
(`pip install cacography`, then `cacography setup`). Compile with `--shell-escape` and
LuaLaTeX starts it by itself. See `doc/cacography-manual.pdf`.

English and German. The LaTeX package is under the LPPL 1.3c. The handwriting models are a
separate download for non-commercial research only.

Version 0.14 (beta), November 2026.
