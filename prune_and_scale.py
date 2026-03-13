#!/usr/bin/env python3

import argparse
import re
import torch
from safetensors.torch import load_file, save_file


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def extract_layer_index(key: str):
    m = LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def should_prune(key: str) -> bool:
    return ".attention." in key


def match_any_range(layer_idx: int, ranges):
    for start, end, mult in ranges:
        if start <= layer_idx <= end:
            return mult
    return 1.0


def tensor_stat(t: torch.Tensor) -> float:
    return t.abs().mean().item()


def prune_and_scale(
    input_path: str,
    output_path: str,
    scale_ranges,
    debug: bool = False,
):
    tensors = load_file(input_path)
    new_tensors = {}

    pruned = 0
    kept = 0

    for key, value in tensors.items():
        # ---- PRUNE ----
        if should_prune(key):
            pruned += 1
            continue

        before = None
        after = None
        mult = 1.0

        if debug:
            before = tensor_stat(value)

        # ---- SCALE ----
        layer_idx = extract_layer_index(key)
        if layer_idx is not None:
            mult = match_any_range(layer_idx, scale_ranges)
            if mult != 1.0:
                value = value * mult

        if debug:
            after = tensor_stat(value)
            kept += 1

            lname = f"layers.{layer_idx}" if layer_idx is not None else "no-layer"
            print(f"[KEEP] {lname} {key} value: {before:.10f} → {after:.10f}  (×{mult})")

        new_tensors[key] = value

    save_file(new_tensors, output_path)

    print(f"Saved: {output_path}")
    print(f"Pruned tensors: {pruned}")
    print(f"Kept tensors: {len(new_tensors)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prune attention layers and scale LoRA layers by index ranges"
    )

    parser.add_argument("input", help="Input .safetensors file")

    parser.add_argument(
        "--scale-range",
        nargs=3,
        action="append",
        metavar=("START", "END", "MULT"),
        help="Scale layers START–END by MULT",
        required=True,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-layer stats before/after",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output file",
    )

    args = parser.parse_args()

    scale_ranges = [
        (int(start), int(end), float(mult))
        for start, end, mult in args.scale_range
    ]

    output = args.output
    if output is None:
        output = args.input.replace(".safetensors", "_pruned_scaled.safetensors")

    prune_and_scale(
        input_path=args.input,
        output_path=output,
        scale_ranges=scale_ranges,
        debug=args.debug,
    )
1
