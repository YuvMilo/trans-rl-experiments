#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Interplay-LM-Reasoning/composition"
REVISION = "a09d5c14c02bfa339143fb00a93274d1a84aa31d"


def rows(path: Path) -> Iterable[tuple[str, dict]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = line.rstrip("\n")
            if not raw:
                continue
            try:
                yield raw, json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def row_key(row: dict) -> bytes:
    digest = hashlib.sha256()
    for key in ("problem", "question", "solution"):
        digest.update(str(row.get(key, "")).encode())
        digest.update(b"\0")
    return digest.digest()


def validate_op(row: dict, expected: int, source: Path) -> None:
    try:
        actual = int(row["op"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid op={row.get('op')!r}") from exc
    if actual != expected:
        raise ValueError(f"{source}: found op={actual}, expected op={expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, type=int)
    parser.add_argument("--target", required=True, type=int)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--max-shards", default=2, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.heldout.is_file():
        raise FileNotFoundError(args.heldout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    seen: set[bytes] = set()
    written = 0

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".op{args.op}-{args.target}-",
        suffix=".jsonl",
        dir=args.output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    def append_unique(source: Path, output) -> bool:
        nonlocal written
        for raw, row in rows(source):
            validate_op(row, args.op, source)
            key = row_key(row)
            if key in seen:
                continue
            seen.add(key)
            output.write(raw + "\n")
            written += 1
            if written >= args.target:
                return True
        return False

    try:
        with temporary.open("w", encoding="utf-8") as output:
            complete = append_unique(args.heldout, output)
            for index in range(args.max_shards):
                if complete:
                    break
                filename = f"train/{args.op}/op{args.op}_shard{index}_1B.jsonl"
                shard = Path(
                    hf_hub_download(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        revision=REVISION,
                        filename=filename,
                        local_dir=args.cache_dir,
                    )
                )
                complete = append_unique(shard, output)
        if written != args.target:
            raise RuntimeError(
                f"found {written:,} unique rows; expected {args.target:,}"
            )
        shutil.move(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"[ok] {args.output}: {written:,} unique rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
