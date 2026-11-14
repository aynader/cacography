# cacography

Typeset a LaTeX document, get back a phone photo of it handwritten. LaTeX does the layout,
equations and structure; a handwriting model trained on real pen recordings writes every
page on procedurally made paper, which is then photographed on a desk. The PDF keeps the
typeset text invisibly underneath, so it stays searchable.

This bundle is the LaTeX side (`cacography.sty`, `cacography.lua`; needs LuaLaTeX). The
pages are drawn by a separate Python program, `cacography` on PyPI
(`pip install cacography`, then `cacography setup`). Compile with `--shell-escape` and
LuaLaTeX starts it by itself. Without that program, documents compile normally and show
the typeset text. See `doc/cacography-manual.pdf`.

English and German. Version 0.14 (beta), November 2026.

## Author and contact

A. Nader. Bug reports and questions: https://github.com/aynader/cacography/issues
Source repository: https://github.com/aynader/cacography

## Licence

Every file in this bundle is under the LaTeX Project Public License, version 1.3c or
later (https://www.latex-project.org/lppl.txt):

- `tex/cacography.sty`, `tex/cacography.lua`, `tex/LICENSE`
- `README.md`
- `doc/cacography-manual.tex`, `doc/cacography-manual.pdf`
- `doc/examples/*.tex`

No file in this bundle is under any other licence. The following are not part of this
bundle and are distributed separately under their own terms:

- the Python program `cacography` (PyPI, GitHub), under the MIT licence;
- its trained handwriting models (a download from the GitHub release), which are for
  non-commercial research use only, because the datasets they were trained on allow
  only that.
