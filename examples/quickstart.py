"""Build, save, reopen, and search a tiny collection."""
from pathlib import Path
import numpy as np
from vortex import Index

path = Path("data/example.vtx")
index = Index(3, metric="cosine", threads=2)
index.add(100, np.array([1, 0, 0], dtype=np.float32))
index.add(200, np.array([0, 1, 0], dtype=np.float32))
index.add(300, np.array([0.8, 0.2, 0], dtype=np.float32))
index.build_ivf(2, iterations=4)
index.save(str(path))
reopened = Index.open(str(path), cache_bytes=4096)
print(reopened.search(np.array([1, 0, 0], dtype=np.float32), k=2, nprobe=2))
print(reopened.cache_stats())
