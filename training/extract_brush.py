"""Extract the BRUSH archive after checking every path stays inside the target."""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brushio import extract  # noqa: E402

zip_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as zf:
    infos = zf.infolist()
    print(len(infos), "entries,", round(sum(i.file_size for i in infos) / 1e6), "MB uncompressed")
    print("first entries:", [i.filename for i in infos[:5]])
    print("resampled entries:", sum("resample" in i.filename for i in infos))
out_dir.mkdir(parents=True, exist_ok=True)
extract(zip_path, out_dir)
print("extracted to", out_dir)
