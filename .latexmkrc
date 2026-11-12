# latexmk settings for cacography documents (used by VS Code's LaTeX Workshop, Emacs, ...).
# Copy this file next to your .tex file, or into your home folder as .latexmkrc.
# LuaLaTeX draws the pages itself when shell-escape is on; latexmk then runs it again to
# put them in, because the page status file it reads has changed.
$pdf_mode = 4;
$lualatex = 'lualatex -shell-escape -synctex=1 -interaction=nonstopmode %O %S';
$max_repeat = 5;
