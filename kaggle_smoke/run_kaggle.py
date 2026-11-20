"""Smoke kernel: run only the 3B fp16 candidate on 15 dev questions at k_ctx=5.

Reuses the archived main kernel script from the source dataset with
``RAG_SMOKE=1`` so the staged pipeline, per-request constrained decoding,
subprocess isolation, and resource diagnostics can be validated before the
full benchmark run. Smoke outputs are diagnostics only and are never
committed as results.
"""

import os
import runpy
import sys
from pathlib import Path

os.environ["RAG_SMOKE"] = "1"

candidates = [
    p
    for p in Path("/kaggle/input").rglob("run_kaggle.py")
    if p.parent.name == "kaggle"
]
if not candidates:
    raise SystemExit("main kernel script not found in the source dataset")
target = sorted(candidates)[0]
print(f"smoke: delegating to {target}", flush=True)
sys.argv = [str(target)]
runpy.run_path(str(target), run_name="__main__")
