#!/usr/bin/env python3
"""
Remove diffusion_model.layers.*.attention.* LoRA tensors
by deleting them from the safetensors file and saving a new one.

Usage:
  ./prune.py file1.safetensors file2.safetensors
  ./prune.py name*.safetensors
"""

from safetensors.torch import load_file, save_file
import glob
import os
import sys

# =========================
# REMOVE RULE
# =========================
def should_remove(key: str) -> bool:
    """
    Matches attention LoRA tensors to be removed entirely
    """
    return (
        key.startswith("diffusion_model.layers.")
        and (
            ".attention." in key
            # or ".layers.20." in key
            # or ".layers.1" in key
            # or ".layers.2." in key
            # or ".layers.3." in key
            # or ".layers.4." in key
            # or ".layers.5." in key
            # or ".layers.26." in key
            # or ".layers.27." in key
            # or ".layers.28." in key
            # or ".layers.29." in key
        )
        # and (
        #     key.endswith(".lora_A.weight")
        #     or key.endswith(".lora_B.weight")
        # )
    )

# =========================
# INPUT FILES
# =========================
if len(sys.argv) < 2:
    print("Usage: prune.py <file_or_pattern> [more_files_or_patterns...]")
    sys.exit(1)

input_patterns = sys.argv[1:]
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
        if should_remove(key):
            removed_keys.append(key)
            continue
        new_state[key] = tensor
        keep_keys.append(key)

    print(f"Saving: {output_lora}")
    save_file(new_state, output_lora)

    print("===================================")
    print(f"File               : {input_lora}")
    print(f"Total tensors      : {len(state)}")
    print(f"Removed tensors    : {len(removed_keys)}")
    print(f"Kept tensors       : {len(keep_keys)}")
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

