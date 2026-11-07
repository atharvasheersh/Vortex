"""Persistent vector search; higher scores are better for ip/cosine, lower for l2."""
from _vortex import Index, __version__, avx2_available

__all__ = ["Index", "__version__", "avx2_available"]
