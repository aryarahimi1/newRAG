"""Side-effect module: configure HuggingFace `transformers` before first import.

If `tensorflow` is installed but broken/incompatible, `transformers` may still
try to wire TF integrations and crash (e.g. `module 'tensorflow' has no
attribute 'data'`). This stack is **PyTorch-only** — we disable TF/Flax before
`sentence_transformers` / `transformers` load.
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
