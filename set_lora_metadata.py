#!/usr/bin/env python3
"""
Replace / edit safetensors LoRA metadata without touching a single tensor.

Metadata lives in the safetensors header; tensors are re-serialized unchanged
(bit-for-bit identical values). Combine operations freely:

  --set      add or overwrite individual keys (repeatable)
  --del      remove individual keys (repeatable)
  --json     load a full metadata dict from a JSON file (keys become strings)
  --clear    start from an empty header instead of inheriting the existing one

Usage:
  python set_lora_metadata.py lora.safetensors --set name="My Custom" --set ss_output_name=custom_v1
  python set_lora_metadata.py lora.safetensors --del ss_learning_rate --del ss_epoch
  python set_lora_metadata.py lora.safetensors --json meta.json --out out.safetensors
  python set_lora_metadata.py lora.safetensors --clear --set name=Clean --dry-run

Metadata values must be strings in safetensors; non-string JSON values are
converted (dicts/lists become JSON strings). Without --out the result is
written to <stem>_meta.safetensors next to the input.
"""

import argparse
import json
import os
import sys

from safetensors.torch import load_file, save_file
from safetensors import safe_open


def read_metadata(path: str) -> dict:
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return dict(f.metadata() or {})
    except Exception as e:
        print(f"Warning: could not read metadata header: {e}")
        return {}


def to_metadata_value(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def parse_set(items) -> dict:
    pairs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set has empty key: {item!r}")
        pairs[key] = value
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Replace or edit safetensors LoRA metadata (tensors untouched).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input .safetensors LoRA")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Set a metadata key (repeatable)")
    parser.add_argument("--del", dest="delete", action="append", default=[], metavar="KEY",
                        help="Delete a metadata key (repeatable)")
    parser.add_argument("--json", dest="json_file", default=None, metavar="FILE",
                        help="Load metadata dict from a JSON file (values stringified)")
    parser.add_argument("--clear", action="store_true",
                        help="Start from an empty header (drop ALL existing metadata)")
    parser.add_argument("--out", default=None, help="Output path (default: <stem>_meta.safetensors)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resulting metadata without writing a file")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found — {args.input}")
        sys.exit(1)

    # ── Build the new metadata dict ──
    if args.clear:
        meta: dict = {}
    else:
        meta = read_metadata(args.input)

    before = dict(meta)

    if args.json_file:
        if not os.path.isfile(args.json_file):
            print(f"Error: JSON file not found — {args.json_file}")
            sys.exit(1)
        with open(args.json_file) as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            print("Error: JSON file must contain an object (key/value map)")
            sys.exit(1)
        for k, v in loaded.items():
            meta[str(k)] = to_metadata_value(v)

    try:
        sets = parse_set(args.set)
    except ValueError as e:
        parser.error(str(e))

    for k, v in sets.items():
        meta[k] = v

    for k in args.delete:
        meta.pop(k, None)

    # ── Show the diff ──
    tensors = load_file(args.input, device="cpu")
    print(f"File  : {args.input}")
    print(f"Tensors: untouched ({len(tensors)} keys)")
    print(f"Metadata: {len(before)} -> {len(meta)} keys")

    added = {k: v for k, v in meta.items() if k not in before}
    removed = [k for k in before if k not in meta]
    changed = {k: (before[k], meta[k]) for k in meta if k in before and before[k] != meta[k]}

    if removed:
        print(f"  removed: {', '.join(removed)}")
    for k in added:
        print(f"  added  : {k} = {meta[k]}")
    for k, (old, new) in changed.items():
        print(f"  changed: {k}: {old!r} -> {new!r}")
    if not (added or removed or changed):
        print("  (no metadata changes)")

    if args.dry_run:
        print("\n[dry-run] no file written")
        return

    out = args.out or (
        os.path.splitext(args.input)[0] + "_meta" + os.path.splitext(args.input)[1]
    )
    if os.path.abspath(out) == os.path.abspath(args.input):
        print("Error: --out must differ from the input path")
        sys.exit(1)

    save_file(tensors, out, metadata=meta)
    print(f"\n[done] wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
