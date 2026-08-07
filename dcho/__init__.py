"""dcho - Persian text-to-speech built to run in real time on a CPU.

Importing this package pulls in nothing heavy. The inference runtime lives
in `dcho.infer` and needs only onnxruntime and numpy; everything under
`dcho.model`, `dcho.data` and `dcho.train` requires torch and is imported
lazily so that a deployment install never loads it.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "phonemize", "normalize", "Frontend"]


def __getattr__(name: str):
    if name in ("phonemize", "Frontend"):
        from .text import frontend

        return getattr(frontend, name)
    if name == "normalize":
        from .text.normalizer import normalize

        return normalize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
