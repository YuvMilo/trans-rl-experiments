#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

EXPERIMENTS = {
    "hard-op20": (500_000, {20}),
    "uniform-op10-15": (300_000, set(range(10, 16))),
    "uniform-op10-17": (400_000, set(range(10, 18))),
    "uniform-op10-20": (550_000, set(range(10, 21))),
    "uniform-op10-15-20": (412_500, {10, 15, 20}),
    "uniform-op10-13-17-20": (550_000, {10, 13, 17, 20}),
    "uniform-op17-20": (550_000, {17, 18, 19, 20}),
    "uniform-op18-20": (412_500, {18, 19, 20}),
    "uniform-op5-10-15-20": (550_000, {5, 10, 15, 20}),
}
REQUIRED_FIELDS = {"problem", "question", "solution", "op"}


def expected_rows(path: Path) -> int:
    match = re.search(r"-(50k|500k|137500|200)\.jsonl$", path.name)
    if not match:
        raise ValueError(f"unknown row count in filename: {path}")
    return {"50k": 50_000, "500k": 500_000, "137500": 137_500, "200": 200}[
        match.group(1)
    ]


def expected_op(path: Path) -> int:
    match = re.match(r"op(\d+)-", path.name)
    if not match:
        raise ValueError(f"unknown op in filename: {path}")
    return int(match.group(1))


def validate_jsonl(path: Path) -> str:
    wanted_rows = expected_rows(path)
    wanted_op = expected_op(path)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for count, raw in enumerate(handle, 1):
            digest.update(raw)
            item = json.loads(raw)
            missing = REQUIRED_FIELDS - item.keys()
            if missing:
                raise ValueError(f"{path}:{count}: missing {sorted(missing)}")
            if int(item["op"]) != wanted_op:
                raise ValueError(f"{path}:{count}: expected op {wanted_op}")
    if count != wanted_rows:
        raise ValueError(f"{path}: {count:,} rows; expected {wanted_rows:,}")
    print(f"[ok] {path.name}: {count:,} rows, op {wanted_op}")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()

    bundle = Path(__file__).resolve().parents[1]
    runtime = Path(os.environ.get("RUNTIME_DIR", bundle / ".runtime")).resolve()
    if not (runtime / "verl/trainer/config").is_dir():
        raise FileNotFoundError("runtime missing; run ./bootstrap.sh first")
    os.chdir(runtime)

    configs: dict[str, dict] = {}
    with initialize_config_dir(config_dir=str(bundle / "configs"), version_base=None):
        for name, (wanted_total, wanted_ops) in EXPERIMENTS.items():
            config = OmegaConf.to_container(compose(config_name=name), resolve=True)
            train_files = [Path(item) for item in config["data"]["train_files"]]
            total = sum(expected_rows(path) for path in train_files)
            ops = {expected_op(path) for path in train_files}
            if total != wanted_total or ops != wanted_ops:
                raise ValueError(f"{name}: total={total}, ops={sorted(ops)}")
            configs[name] = config

    print(f"[ok] composed {len(configs)} experiment configs")
    if args.skip_assets:
        return 0

    paths: set[Path] = set()
    for config in configs.values():
        paths.update(Path(item) for item in config["data"]["train_files"])
        paths.update(Path(item) for item in config["data"]["val_files"])
        model = Path(config["actor_rollout_ref"]["model"]["path"])
        for filename in ("config.json", "model.safetensors", "tokenizer.json"):
            if not (model / filename).is_file():
                raise FileNotFoundError(model / filename)

    generated = json.loads((bundle / "manifests/generated-datasets.json").read_text())
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = validate_jsonl(path)
        if path.name in generated and digest != generated[path.name]:
            raise ValueError(f"{path}: SHA-256 mismatch")

    print("[ok] all assets validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
