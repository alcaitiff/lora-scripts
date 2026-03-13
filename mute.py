#!/usr/bin/env python3
"""
Mute unsupported diffusion_model.layers.*.attention.* LoRA tensors
by zeroing their weights and saving new safetensors files.

Usage:
  ./mute.py file1.safetensors file2.safetensors
  ./mute.py "Mystic*.safetensors"
  ./mute.py input.safetensors --out muted.safetensors
"""

import argparse
import glob
import os
import sys
import torch
from safetensors.torch import load_file, save_file

# =========================
# MUTE RULE
# =========================
def should_mute(key: str) -> bool:
    """
    Matches exactly the keys reported as 'not loaded'
    """
    return (
        key.startswith("diffusion_model.layers.")
        and ".attention." in key
        and (
            key.endswith(".lora_A.weight")
            or key.endswith(".lora_B.weight")
        )
    )

# =========================
# INPUT FILES
# =========================
parser = argparse.ArgumentParser(
    description="Mute unsupported diffusion_model.layers.*.attention.* tensors"
)
parser.add_argument(
    "inputs",
    nargs="+",
    help="Input files or glob patterns",
)
parser.add_argument(
    "--out",
    default=None,
    help="Output file (single input only)",
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

# Optional single-output override
output_override = None
if args.out:
    if len(input_files) != 1:
        print("--out can only be used when exactly one input file is provided.")
        sys.exit(1)
    output_override = args.out

# =========================
# PROCESS
# =========================
for input_lora in input_files:
    base, ext = os.path.splitext(input_lora)
    output_lora = output_override or f"{base}_muted{ext}"

    print(f"\nLoading: {input_lora}")
    state = load_file(input_lora)

    new_state = {}
    muted_keys = []
    keep_keys = []

    for key, tensor in state.items():
        if should_mute(key):
            new_state[key] = torch.zeros_like(tensor)
            muted_keys.append(key)
        else:
            new_state[key] = tensor
            keep_keys.append(key)

    print(f"Saving: {output_lora}")
    save_file(new_state, output_lora)

    print("===================================")
    print(f"File               : {input_lora}")
    print(f"Total tensors      : {len(state)}")
    print(f"Muted tensors      : {len(muted_keys)}")
    print(f"Keep tensors       : {len(keep_keys)}")
    print("===================================")

    if muted_keys:
        print("Muted keys:")
        for k in muted_keys:
            print(" -", k)
    else:
        print("No keys were muted.")
    print("Keep keys:")
    for k in keep_keys:
        print(" -", k)
