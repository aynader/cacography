# cacography installer for Windows.
#
#   From a download of this repository:   .\install.ps1
#   With the models zip somewhere else:   .\install.ps1 -Models C:\path\cacography-models.zip
#   Into a Python that already has PyTorch (say, after `conda activate base`), nothing big
#   is downloaded; it is used by itself when active, or name it:  .\install.ps1 -Python C:\path\python.exe
#
# Otherwise it installs the renderer with uv, which brings its own Python and PyTorch. Then
# it puts the LaTeX package where TeX finds it, fetches the handwriting models and checks
# everything.
param([string]$Models = "", [string]$Python = "")
# Not "Stop": Windows PowerShell treats any warning a program prints (pip's, uv's) as fatal.
# Each step's exit code is checked instead.
$ErrorActionPreference = "Continue"

Write-Host "cacography: installing" -ForegroundColor Cyan

if (-not (Get-Command lualatex -ErrorAction SilentlyContinue)) {
    Write-Host "LuaLaTeX was not found. Install MiKTeX (https://miktex.org/download) or TeX Live"
    Write-Host "(https://tug.org/texlive), open a new terminal, and run this again."
    exit 1
}

# The renderer: from this folder when run from a download of the repository, else PyPI.
$here = $PSScriptRoot
$source = if ($here -and (Test-Path (Join-Path $here "pyproject.toml"))) { $here } else { "cacography" }

# An active conda or virtual environment that has PyTorch already is used as it is.
if (-not $Python) {
    $active = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" }
              elseif ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "" }
    if ($active -and (Test-Path $active)) {
        & $active -c "import torch" 2>$null
        if ($LASTEXITCODE -eq 0) { $Python = $active }
    }
}

if ($Python) {
    Write-Host "Installing the renderer into $Python (its PyTorch is kept) ..."
    & $Python -m pip install --upgrade $source
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $run = @($Python, "-m", "cacography_render.cli")
} else {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Installing uv, which installs Python programs (https://docs.astral.sh/uv) ..."
        powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    }
    Write-Host "Installing the renderer from $source (PyTorch for your GPU, if there is one) ..."
    uv tool install --force --python 3.12 --torch-backend auto $source
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    uv tool update-shell | Out-Null
    $run = @(Join-Path (uv tool dir --bin) "cacography.exe")
}

$setup = @("setup")
if ($Models) {
    $setup += @("--models", $Models)
} elseif ($here -and (Test-Path (Join-Path $here "cacography-models.zip"))) {
    $setup += @("--models", (Join-Path $here "cacography-models.zip"))
} elseif ($here -and (Test-Path (Join-Path $here "models\hand"))) {
    $setup += @("--models", (Join-Path $here "models"))
}
$exe, $rest = $run
& $exe @rest @setup
exit $LASTEXITCODE
