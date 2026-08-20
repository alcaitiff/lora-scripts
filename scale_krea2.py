#!/usr/bin/env python3
"""
Scale all tensor values in a KREA2 LoRA by a given factor and write new safetensors file(s).

Usage:
  ./scale_krea2.py 3.0 lora.safetensors
  ./scale_krea2.py 0.5 lora.safetensors --out halved.safetensors
  ./scale_krea2.py 10.0 --dry-run lora.safetensors
  ./scale_krea2.py 3.0 *.safetensors
"""

from safetensors.torch import load_file, save_file
import argparse
import glob
import os
import sys

parser = argparse.ArgumentParser(
    description="Scale all tensor values in KREA2 LoRA files by a factor"
)
parser.add_argument(
    "factor",
    type=float,
    help="Scaling factor (e.g. 3.0 for 3x strength, 0.5 for half)",
)
parser.add_argument(
    "inputs",
    nargs="+",
    help="Input .safetensors files or glob patterns",
)
parser.add_argument(
    "--out",
    default=None,
    help="Output file (single input only)",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without writing output files",
)
args = parser.parse_args()

if args.factor <= 0:
    print("Factor must be positive.")
    sys.exit(1)

# Resolve input files
input_patterns = args.inputs
input_files = []

for pattern in input_patterns:
    matches = glob.glob(pattern)
    if not matches:
        print(f"Warning: no files matched '{pattern}'")
    input_files.extend(matches)

if not input_files:
    print("No input files to process.")
    sys.exit(1)

# Deduplicate, preserve order
seen = set()
input_files = [f for f in input_files if not (f in seen or seen.add(f))]

output_override = None
if args.out:
    if len(input_files) != 1:
        print("--out can only be used when exactly one input file is provided.")
        sys.exit(1)
    output_override = args.out

# Process
for input_path in input_files:
    base, ext = os.path.splitext(input_path)
    # Build output filename
    if output_override:
        output_path = output_override
    else:
        # insert _x<factor> before extension
        factor_str = str(args.factor).rstrip("0").rstrip(".")
        output_path = f"{base}_x{factor_str}{ext}"

    print(f"\nLoading: {input_path}")
    state = load_file(input_path)

    # Stats before
    n_tensors = len(state)
    n_elements = sum(t.numel() for t in state.values())
    mean_before = sum(t.abs().mean().item() for t in state.values()) / n_tensors
    min_before = min(t.min().item() for t in state.values())
    max_before = max(t.max().item() for t in state.values())

    if args.dry_run:
        print(f"Would scale {n_tensors} tensors ({n_elements:,} elements) by {args.factor}x")
        print(f"  abs mean: {mean_before:.6f} -> {mean_before * args.factor:.6f}")
        print(f"  range:    [{min_before:.6f}, {max_before:.6f}] -> [{min_before * args.factor:.6f}, {max_before * args.factor:.6f}]")
        print(f"  output:   {output_path}")
        continue

    # Scale
    state = {k: v * args.factor for k, v in state.items()}

    # Stats after
    mean_after = sum(v.abs().mean().item() for v in state.values()) / n_tensors

    print(f"Saving: {output_path}")
    save_file(state, output_path)

    print(f"  tensors:      {n_tensors}")
    print(f"  elements:     {n_elements:,}")
    print(f"  abs mean:     {mean_before:.6f} -> {mean_after:.6f}")
    print(f"  factor:       {args.factor}x")
    print(f"  dtype:        {next(iter(state.values())).dtype}")
