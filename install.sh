#!/bin/sh
# cacography installer for macOS and Linux.
#
#   From a download of this repository:   sh install.sh
#   With the models zip somewhere else:   sh install.sh --models /path/to/cacography-models.zip
#   Into a Python that already has PyTorch (say, after `conda activate base`), nothing big
#   is downloaded; it is used by itself when active, or name it:  sh install.sh --python /path/to/python
#
# Otherwise it installs the renderer with uv, which brings its own Python and PyTorch. Then
# it puts the LaTeX package where TeX finds it, fetches the handwriting models and checks
# everything.
set -e
models=""
python=""
while [ $# -gt 0 ]; do
  case "$1" in
    --models) models="$2"; shift 2 ;;
    --python) python="$2"; shift 2 ;;
    *) models="$1"; shift ;;
  esac
done
echo "cacography: installing"

if ! command -v lualatex >/dev/null 2>&1; then
  echo "LuaLaTeX was not found. Install TeX Live (https://tug.org/texlive, or MacTeX on a Mac),"
  echo "open a new terminal, and run this again."
  exit 1
fi

# The renderer: from this folder when run from a download of the repository, else PyPI.
here="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$here/pyproject.toml" ]; then source="$here"; else source="cacography"; fi

# An active conda or virtual environment that has PyTorch already is used as it is.
if [ -z "$python" ]; then
  for p in "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"; do
    if [ -n "$p" ] && [ -x "$p" ] && "$p" -c "import torch" 2>/dev/null; then python="$p"; break; fi
  done
fi

if [ -n "$python" ]; then
  echo "Installing the renderer into $python (its PyTorch is kept) ..."
  "$python" -m pip install --upgrade "$source"
else
  if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv, which installs Python programs (https://docs.astral.sh/uv) ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
  fi
  echo "Installing the renderer from $source (PyTorch for your GPU, if there is one) ..."
  uv tool install --force --python 3.12 --torch-backend auto "$source"
  uv tool update-shell >/dev/null 2>&1 || true
fi

cli() {
  if [ -n "$python" ]; then "$python" -m cacography_render.cli "$@"; else "$(uv tool dir --bin)/cacography" "$@"; fi
}

if [ -n "$models" ]; then
  cli setup --models "$models"
elif [ -f "$here/cacography-models.zip" ]; then
  cli setup --models "$here/cacography-models.zip"
elif [ -d "$here/models/hand" ]; then
  cli setup --models "$here/models"
else
  cli setup
fi
