#!/usr/bin/env python3
"""
Display safetensors metadata and tensor information for LoRA files.

Shows:
  - File-level JSON header metadata (training info, base model, hash, etc.)
  - Per-tensor keys grouped by block with shape, dtype, and element count
  - Total size summary

Usage:
  python lora_metadata.py                     # defaults to L.safetensors
  python lora_metadata.py path/to/lora.safetensors
  python lora_metadata.py path/to/lora.safetensors --json   # raw metadata JSON
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from safetensors import safe_open
from safetensors.torch import load_file


def format_size(num_bytes: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def show_raw_metadata(state_dict: dict, filepath: str):
    """Dump the raw JSON metadata from the safetensors header."""
    try:
        # The easiest way: re-read the first bytes of the file for the header
        with open(filepath, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header_json = f.read(header_len).decode("utf-8")
        metadata = json.loads(header_json).get("__metadata__", {})
    except Exception:
        metadata = {}

    print("=" * 70)
    print("RAW METADATA (safetensors header)")
    print("=" * 70)
    if metadata:
        print(json.dumps(metadata, indent=2))
    else:
        print("(no metadata in header)")

    # Also check if the state_dict itself has metadata attached
    if hasattr(state_dict, "metadata") and state_dict.metadata:
        print("\n--- PyTorch state_dict metadata ---")
        print(json.dumps(state_dict.metadata, indent=2))


# ──────────────────────────────────────────────
# Parsed / pretty-printed metadata display
# ──────────────────────────────────────────────

def show_pretty_metadata(header_metadata: dict):
    """Display common LoRA training metadata fields in a friendly format."""
    if not header_metadata:
        print("(no metadata found in safetensors header)")
        print()
        return

    print("─" * 70)
    print("METADATA FIELDS")
    print("─" * 70)

    # Standard fields often present
    fields = [
        ("Name", "name"),
        ("Output name", "ss_output_name"),
        ("Base model version", "ss_base_model_version"),
        ("Safetensors version", "version"),
        ("Format", "format"),
        ("Model hash (SHA256)", "sshs_model_hash"),
        ("Legacy hash", "sshs_legacy_hash"),
    ]

    for label, key in fields:
        val = header_metadata.get(key)
        if val:
            print(f"  {label:<30} = {val}")

    # Software info (stored as JSON string)
    sw = header_metadata.get("software")
    if sw:
        try:
            sw_obj = json.loads(sw)
            print(f"  {'Software':<30} = {sw_obj.get('name', '')} {sw_obj.get('version', '')}  ({sw_obj.get('repo', '')})")
        except (json.JSONDecodeError, TypeError):
            print(f"  {'Software':<30} = {sw}")

    # Training info (stored as JSON string)
    ti = header_metadata.get("training_info")
    if ti:
        try:
            ti_obj = json.loads(ti)
            step = ti_obj.get("step", "?")
            epoch = ti_obj.get("epoch", "?")
            print(f"  {'Training step':<30} = {step}")
            print(f"  {'Training epoch':<30} = {epoch}")
        except (json.JSONDecodeError, TypeError):
            print(f"  {'Training info':<30} = {ti}")

    # Any remaining unknown metadata fields
    known_keys = {"version", "ss_base_model_version", "ss_output_name",
                  "sshs_model_hash", "format", "software", "name",
                  "sshs_legacy_hash", "training_info"}
    extra = {k: v for k, v in header_metadata.items() if k not in known_keys}
    if extra:
        print(f"\n  Additional fields:")
        for k, v in extra.items():
            print(f"    {k} = {v}")
    print()


# ──────────────────────────────────────────────
# Tensor inspection
# ──────────────────────────────────────────────

def show_tensors(state_dict: dict):
    """Print tensor keys grouped by block, with shape / dtype / element count."""
    print("─" * 70)
    print("TENSOR INVENTORY")
    print("─" * 70)

    total_elements = 0
    total_bytes = 0

    # Group by block type / number
    double_blocks = defaultdict(list)
    single_blocks = defaultdict(list)
    other_keys = []

    for key, tensor in state_dict.items():
        el = tensor.numel()
        tb = tensor.element_size() * el
        total_elements += el
        total_bytes += tb

        entry = (key, list(tensor.shape), str(tensor.dtype), el, tb)

        if "double_blocks." in key:
            parts = key.split("double_blocks.")[1]
            block_num = parts.split(".")[0]
            double_blocks[block_num].append(entry)
        elif "single_blocks." in key:
            parts = key.split("single_blocks.")[1]
            block_num = parts.split(".")[0]
            single_blocks[block_num].append(entry)
        else:
            other_keys.append(entry)

    # ── double_blocks ──
    if double_blocks:
        for bnum in sorted(double_blocks.keys(), key=int):
            print(f"\n  double_blocks.{bnum}:")
            for _, shape, dtype, el, tb in sorted(double_blocks[bnum], key=lambda x: x[0]):  # sorted by full key
                pass
            # Print with full key names
            for key, shape, dtype, el, tb in sorted(double_blocks[bnum], key=lambda x: x[0]):
                shape_str = f"{'×'.join(str(s) for s in shape)}" if shape else "scalar"
                print(f"    {key:.<72} [{shape_str}]  {dtype}  {el:>10,} elems  {format_size(tb)}")
    else:
        print("  (no double_blocks)")

    # ── single_blocks ──
    if single_blocks:
        print()
        for bnum in sorted(single_blocks.keys(), key=int):
            print(f"\n  single_blocks.{bnum}:")
            for key, shape, dtype, el, tb in sorted(single_blocks[bnum], key=lambda x: x[0]):
                shape_str = f"{'×'.join(str(s) for s in shape)}" if shape else "scalar"
                print(f"    {key:.<72} [{shape_str}]  {dtype}  {el:>10,} elems  {format_size(tb)}")

    # ── Other keys ──
    if other_keys:
        print(f"\n  Other keys ({len(other_keys)}):")
        for key, shape, dtype, el, tb in sorted(other_keys, key=lambda x: x[0]):
            shape_str = f"{'×'.join(str(s) for s in shape)}" if shape else "scalar"
            print(f"    {key:.<72} [{shape_str}]  {dtype}  {el:>10,} elems  {format_size(tb)}")

    # ── Summary ──
    print()
    print("─" * 70)
    print("SUMMARY")
    print("─" * 70)
    print(f"  Total tensors        : {len(state_dict)}")
    print(f"  Total elements       : {total_elements:>15,}")
    print(f"  Total size (tensors) : {format_size(total_bytes)}")
    print(f"  Double blocks        : {len(double_blocks)}")
    print(f"  Single blocks        : {len(single_blocks)}")
    print(f"  Others               : {len(other_keys)}")
    print()


# ──────────────────────────────────────────────
# Main (safe_open parsing to get metadata)
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Show safetensors LoRA metadata and tensor inventory."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="L.safetensors",
        help="Input .safetensors file (default: L.safetensors)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump raw JSON metadata only",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found — {args.input}")
        sys.exit(1)

    # ── Read metadata from safetensors header ──
    header_metadata = {}
    try:
        with safe_open(args.input, framework="pt", device="cpu") as f:
            header_metadata = f.metadata() or {}
    except Exception as e:
        print(f"Warning: could not read safetensors metadata: {e}")

    # ── Load full state dict for tensor inspection ──
    print(f"\n  File : {args.input}")
    print(f"  Size : {format_size(os.path.getsize(args.input))}\n")

    if args.json:
        show_raw_metadata({}, args.input)
        return

    # Pretty-printed metadata
    show_pretty_metadata(header_metadata)

    # Load the full state dict for tensor inspection
    try:
        state_dict = load_file(args.input, device="cpu")
    except Exception as e:
        print(f"Error loading tensors: {e}")
        sys.exit(1)

    show_tensors(state_dict)


if __name__ == "__main__":
    main()
