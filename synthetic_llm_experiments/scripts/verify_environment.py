#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path

from packaging.version import Version

EXPECTED = {
    "torch": "2.7.0",
    "transformers": "4.53.3",
    "vllm": "0.9.1",
    "ray": "2.52.1",
    "huggingface-hub": "0.36.2",
}


def main() -> int:
    bundle = Path(__file__).resolve().parents[1]
    runtime = Path(os.environ.get("RUNTIME_DIR", bundle / ".runtime")).resolve()
    sys.path.insert(0, str(runtime))
    failures: list[str] = []

    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python {sys.version.split()[0]} found; expected 3.12")

    for package, expected in EXPECTED.items():
        actual = Version(importlib.metadata.version(package)).base_version
        if actual != expected:
            failures.append(f"{package} {actual} found; expected {expected}")

    for module in ("verl", "flash_attn", "omegaconf", "hydra"):
        importlib.import_module(module)

    try:
        importlib.import_module("uvloop")
    except ModuleNotFoundError:
        pass
    else:
        failures.append("uvloop must be uninstalled")

    for failure in failures:
        print(f"[FAIL] {failure}", file=sys.stderr)
    if failures:
        return 1
    print("[ok] environment versions and imports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
