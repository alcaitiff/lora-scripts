#!/usr/bin/env python3
"""
Remove LoRA tensors whose keys contain any of the provided substrings
by deleting them from the safetensors file and saving a new one.

Usage:
  ./prune.py --match ".attention." ".layers.20." file1.safetensors file2.safetensors
  ./prune.py --match ".attention." "to_k" name*.safetensors
"""

from safetensors.torch import load_file, save_file
import argparse
import glob
import os
import sys

def should_remove(key: str, substrings) -> bool:
    return any(s in key for s in substrings)

# =========================
# INPUT FILES
# =========================
parser = argparse.ArgumentParser(
    description="Remove tensors whose keys contain any of the provided substrings"
)
parser.add_argument(
    "--match",
    nargs="+",
    required=True,
    help="One or more substrings to match against tensor keys",
)
parser.add_argument(
    "inputs",
    nargs="+",
    help="Input .safetensors files or glob patterns",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show which keys would be removed without writing output files",
)
args = parser.parse_args()

input_patterns = args.inputs
input_files = []

for pattern in input_patterns:
    matches = glob.glob(pattern)
    if not matches:
        print(f"Warning: no files matched '{pattern}'")
    input_files.extend(matches)

# remove duplicates, preserve order
seen = set()
input_files = [f for f in input_files if not (f in seen or seen.add(f))]

if not input_files:
    print("No input files to process.")
    sys.exit(1)

# =========================
# PROCESS
# =========================
for input_lora in input_files:
    base, ext = os.path.splitext(input_lora)
    output_lora = f"{base}_pruned{ext}"

    print(f"\nLoading: {input_lora}")
    state = load_file(input_lora)

    new_state = {}
    removed_keys = []
    keep_keys = []

    for key, tensor in state.items():
        if should_remove(key, args.match):
            removed_keys.append(key)
            continue
        new_state[key] = tensor
        keep_keys.append(key)

    if args.dry_run:
        print("Dry run: no output file written.")
    else:
        print(f"Saving: {output_lora}")
        save_file(new_state, output_lora)

    print("===================================")
    print(f"File               : {input_lora}")
    print(f"Total tensors      : {len(state)}")
    print(f"Removed tensors    : {len(removed_keys)}")
    print(f"Kept tensors       : {len(keep_keys)}")
    if args.dry_run:
        print("Mode               : dry-run (no file written)")
    print("===================================")

    if removed_keys:
        print("Removed keys:")
        for k in removed_keys:
            print(" -", k)
    else:
        print("No keys were removed.")

    print("Kept keys:")
    for k in keep_keys:
        print(" -", k)
