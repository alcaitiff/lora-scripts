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
    required=False,
    help="One or more substrings to match against tensor keys",
)
parser.add_argument(
    "--blocks",
    nargs="+",
    help="Block indexes or ranges (e.g. 4 7 10-13) to match as 'block.<n>.'",
)
parser.add_argument(
    "inputs",
    nargs="*",
    help="Input .safetensors files or glob patterns",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show which keys would be removed without writing output files",
)
args = parser.parse_args()

def is_block_token(tok: str) -> bool:
    if tok.isdigit():
        return True
    if "-" in tok:
        parts = tok.split("-", 1)
        return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()
    return False

def parse_block_tokens(tokens):
    if not tokens:
        return []
    out = []
    for tok in tokens:
        if "-" in tok:
            parts = tok.split("-", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid block range: {tok}")
            start = int(parts[0])
            end = int(parts[1])
            if end < start:
                raise ValueError(f"Invalid block range: {tok}")
            out.extend(range(start, end + 1))
        else:
            out.append(int(tok))
    return out

block_matches = []

# If inputs were omitted, try to infer them from --blocks tokens too.
if args.blocks and not args.inputs:
    inferred_inputs = []
    remaining_blocks = []
    for token in args.blocks:
        if is_block_token(token):
            remaining_blocks.append(token)
            continue
        if any(ch in token for ch in "*?["):
            matches = glob.glob(token)
            if matches:
                inferred_inputs.extend(matches)
                continue
        if os.path.exists(token):
            inferred_inputs.append(token)
        else:
            print(f"Invalid block token: {token}")
            sys.exit(1)
    if inferred_inputs:
        args.inputs = inferred_inputs
    args.blocks = remaining_blocks

if args.blocks:
    try:
        block_ids = parse_block_tokens(args.blocks)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    block_matches = [f"block.{i}." for i in sorted(set(block_ids))]

# If inputs were omitted, try to infer them from --match tokens.
# This handles cases where a long --match list swallows the inputs.
if not args.inputs:
    inferred_inputs = []
    remaining_matches = []
    for token in args.match:
        if any(ch in token for ch in "*?["):
            matches = glob.glob(token)
            if matches:
                inferred_inputs.extend(matches)
                continue
        if os.path.exists(token):
            inferred_inputs.append(token)
        else:
            remaining_matches.append(token)

    if inferred_inputs:
        args.inputs = inferred_inputs
        args.match = remaining_matches

if not args.inputs:
    print("No input files provided.")
    print("Tip: add `--` before your input files if you pass a long --match list.")
    sys.exit(1)

match_list = args.match or []
combined_matches = list(match_list) + block_matches
if not combined_matches:
    print("No match substrings provided.")
    sys.exit(1)

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
        if should_remove(key, combined_matches):
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
