from __future__ import annotations
import struct
from pathlib import Path
import numpy as np

MAGIC = b"CHDS"
VERSION = 1

# dataset.bin layout:
# 4 bytes magic
# uint32 version
# uint32 input_dim
# uint64 count
# then repeated:
# float32[input_dim] x
# float32 y

def write_dataset_bin(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    assert X.dtype == np.float32
    assert y.dtype == np.float32
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    count = X.shape[0]
    input_dim = X.shape[1]

    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIQ", VERSION, input_dim, 0, count))  # reserved=0
        # stream rows
        f.write(X.tobytes(order="C"))
        f.write(y.tobytes(order="C"))

def read_dataset_bin_memmap(path: Path):
    import mmap
    with path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    # header
    if mm[:4] != MAGIC:
        raise ValueError("Bad dataset magic")
    version, input_dim, _reserved, count = struct.unpack("<IIIQ", mm[4:4+4+4+4+8])
    if version != VERSION:
        raise ValueError(f"Unsupported dataset version {version}")
    header_size = 4 + 4 + 4 + 4 + 8
    x_bytes = count * input_dim * 4
    X = np.frombuffer(mm, dtype=np.float32, count=count*input_dim, offset=header_size).reshape(count, input_dim)
    y = np.frombuffer(mm, dtype=np.float32, count=count, offset=header_size + x_bytes)
    return mm, X, y
