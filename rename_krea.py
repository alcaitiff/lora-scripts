#!/usr/bin/env python3
"""
Rename Kohya/diffusers-style LoRA keys (base_model.model.*) to
ComfyUI/Krea 2 native format (diffusion_model.*).

Some trainers (Kohya, AI Toolkit) prefix keys with "base_model.model."
instead of the "diffusion_model." prefix that ComfyUI expects for Flux/Krea 2
models. This script simply replaces the prefix across all keys, preserving
every tensor including extra layers like tmlp, tproj, txtmlp, first, etc.

Usage:
  ./rename_krea.py input.safetensors --out converted.safetensors
  ./rename_krea.py input.safetensors                           # auto-named
  ./rename_krea.py input.safetensors --dry-run                 # preview only
"""

import argparse
import os
import sys

try:
    from safetensors.torch import load_file, save_file
except ImportError:
    print("Error: safetensors not installed. Run: pip install safetensors")
    sys.exit(1)


def rename_keys(state_dict, dry_run=False):
    """Rename base_model.model. prefix to diffusion_model. across all keys."""
    new_sd = {}
    changes = []
    skipped = []

    for key, tensor in state_dict.items():
        if key.startswith("base_model.model."):
            new_key = key.replace("base_model.model.", "diffusion_model.", 1)
            changes.append((key, new_key))
            new_sd[new_key] = tensor
        else:
            skipped.append(key)
            new_sd[key] = tensor  # pass through unchanged

    if dry_run:
        print(f"\nWould rename {len(changes)} keys:")
        if changes:
            print("  Sample renames (first 5):")
            for old, new in changes[:5]:
                print(f"    {old}")
                print(f"  → {new}")
            if len(changes) > 5:
                print(f"    ... and {len(changes) - 5} more")
        if skipped:
            print(f"\n  {len(skipped)} keys left unchanged (no 'base_model.model.' prefix):")
            for k in skipped[:3]:
                print(f"    {k}")
            if len(skipped) > 3:
                print(f"    ... and {len(skipped) - 3} more")
        return None

    return new_sd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename base_model.model.* → diffusion_model.* for ComfyUI/Krea 2 compatibility"
    )
    parser.add_argument("input", help="Input .safetensors file")
    parser.add_argument("--out", default=None, help="Output .safetensors file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing output",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    base, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".safetensors"
    output_path = args.out or f"{base}_converted{ext}"

    print(f"Loading: {input_path}")
    state_dict = load_file(input_path, device="cpu")
    print(f"Loaded {len(state_dict)} tensors")

    # Check if conversion is needed
    has_base_model = any(k.startswith("base_model.model.") for k in state_dict)
    has_diffusion = any(k.startswith("diffusion_model.") for k in state_dict)

    if not has_base_model and has_diffusion:
        print("Already using 'diffusion_model.' prefix. No conversion needed.")
        if not args.dry_run:
            print("Nothing to do.")
        sys.exit(0)
    elif not has_base_model and not has_diffusion:
        print("Warning: keys use neither 'base_model.model.' nor 'diffusion_model.' prefix.")
        print("First few keys:")
        for k in list(state_dict.keys())[:5]:
            print(f"  {k}")

    new_sd = rename_keys(state_dict, dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    print(f"\nRenamed {sum(1 for k in state_dict if k.startswith('base_model.model.'))} keys")
    print(f"Kept {sum(1 for k in state_dict if not k.startswith('base_model.model.'))} keys unchanged")

    print(f"\nSaving: {output_path}")
    save_file(new_sd, output_path)

    original_bytes = sum(t.element_size() * t.numel() for t in state_dict.values())
    new_bytes = sum(t.element_size() * t.numel() for t in new_sd.values())
    print(f"Original size: {original_bytes / 1024 / 1024:.1f} MB")
    print(f"Output size:   {new_bytes / 1024 / 1024:.1f} MB")

    print("\nDone! Sample converted keys:")
    count = 0
    for old_key in list(state_dict.keys()):
        if old_key.startswith("base_model.model."):
            new_key = old_key.replace("base_model.model.", "diffusion_model.", 1)
            print(f"  {old_key}")
            print(f"  → {new_key}")
            count += 1
            if count >= 3:
                break
