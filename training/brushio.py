"""Safe reading of the BRUSH dataset.

BRUSH samples are Python pickles from a third party. A normal pickle.load can
run arbitrary code, so this unpickler only resolves the NumPy names needed to
rebuild plain arrays and refuses everything else.
"""
import io
import pickle
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

_ALLOWED = {
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"blocked {module}.{name}") from None


def load_sample(raw: bytes):
    """Return (sentence, drawing, label), validated, from one pickle's bytes."""
    obj = SafeUnpickler(io.BytesIO(raw), encoding="latin1").load()
    if not isinstance(obj, (list, tuple)) or len(obj) != 3:
        raise ValueError("unexpected sample structure")
    sentence, drawing, label = obj
    if not isinstance(sentence, str):
        raise ValueError("sentence is not a string")
    drawing = np.asarray(drawing, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)
    if drawing.ndim != 2 or drawing.shape[1] != 3:
        raise ValueError(f"bad drawing shape {drawing.shape}")
    if label.ndim != 2 or label.shape != (len(drawing), len(sentence)):
        raise ValueError(f"bad label shape {label.shape}")
    return sentence, drawing, label


def extract(zip_path: Path, out_dir: Path):
    """Extract only regular files whose paths stay inside out_dir."""
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            if ".." in parts or (parts and parts[0].endswith(":")) or info.filename.startswith(("/", "\\")):
                raise ValueError(f"unsafe path in archive: {info.filename}")
            target = (out_dir / Path(*parts)).resolve()
            if out_dir not in target.parents:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())


def iter_original_files(root: Path):
    """Yield (writer_id, path) for the original 10 ms drawings only."""
    for writer in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(writer.iterdir()):
            if path.is_file() and "resample" not in path.name:
                yield writer.name, path
