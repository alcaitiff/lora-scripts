#!/usr/bin/env python3
"""
Convert KREA2 LoRA keys between naming conventions.

Standard convention (used by most KREA2 LoRAs):
  diffusion_model.blocks.{N}.attn.wq
  diffusion_model.blocks.{N}.mlp.up
  diffusion_model.txtfusion.layerwise_blocks.{N}.attn.wq

Alternate convention (used by some LoRAs like xray-krea2):
  transformer.transformer_blocks.{N}.attn.to_q
  transformer.transformer_blocks.{N}.ff.up
  transformer.text_fusion.layerwise_blocks.{N}.attn.to_q

This script converts the alternate → standard convention in-place.
"""

import argparse
import os
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


# Mapping: (pattern_regex, replacement)
# Ordered from most specific to least specific
KEY_RULES = [
    # Blocks: transformer.transformer_blocks.N → diffusion_model.blocks.N
    # attn: to_q/wk/wv/gate/out → wq/wk/wv/gate/wo
    (r'transformer\.transformer_blocks\.(\d+)\.attn\.to_q',
     r'diffusion_model.blocks.\1.attn.wq'),
    (r'transformer\.transformer_blocks\.(\d+)\.attn\.to_k',
     r'diffusion_model.blocks.\1.attn.wk'),
    (r'transformer\.transformer_blocks\.(\d+)\.attn\.to_v',
     r'diffusion_model.blocks.\1.attn.wv'),
    (r'transformer\.transformer_blocks\.(\d+)\.attn\.to_gate',
     r'diffusion_model.blocks.\1.attn.gate'),
    (r'transformer\.transformer_blocks\.(\d+)\.attn\.to_out\.0',
     r'diffusion_model.blocks.\1.attn.wo'),
    # Blocks: ff → mlp
    (r'transformer\.transformer_blocks\.(\d+)\.ff\.(up|down|gate)',
     r'diffusion_model.blocks.\1.mlp.\2'),
    
    # Text fusion layerwise: transformer.text_fusion → diffusion_model.txtfusion
    (r'transformer\.text_fusion\.layerwise_blocks\.(\d+)\.attn\.to_q',
     r'diffusion_model.txtfusion.layerwise_blocks.\1.attn.wq'),
    (r'transformer\.text_fusion\.layerwise_blocks\.(\d+)\.attn\.to_k',
     r'diffusion_model.txtfusion.layerwise_blocks.\1.attn.wk'),
    (r'transformer\.text_fusion\.layerwise_blocks\.(\d+)\.attn\.to_v',
     r'diffusion_model.txtfusion.layerwise_blocks.\1.attn.wv'),
    (r'transformer\.text_fusion\.layerwise_blocks\.(\d+)\.attn\.to_gate',
     r'diffusion_model.txtfusion.layerwise_blocks.\1.attn.gate'),
    (r'transformer\.text_fusion\.layerwise_blocks\.(\d+)\.attn\.to_out\.0',
     r'diffusion_model.txtfusion.layerwise_blocks.\1.attn.wo'),
    (r'transformer\.text_fusion\.layerwise_blocks\.(\d+)\.ff\.(up|down|gate)',
     r'diffusion_model.txtfusion.layerwise_blocks.\1.mlp.\2'),
    
    # Text fusion refiner: same pattern
    (r'transformer\.text_fusion\.refiner_blocks\.(\d+)\.attn\.to_q',
     r'diffusion_model.txtfusion.refiner_blocks.\1.attn.wq'),
    (r'transformer\.text_fusion\.refiner_blocks\.(\d+)\.attn\.to_k',
     r'diffusion_model.txtfusion.refiner_blocks.\1.attn.wk'),
    (r'transformer\.text_fusion\.refiner_blocks\.(\d+)\.attn\.to_v',
     r'diffusion_model.txtfusion.refiner_blocks.\1.attn.wv'),
    (r'transformer\.text_fusion\.refiner_blocks\.(\d+)\.attn\.to_gate',
     r'diffusion_model.txtfusion.refiner_blocks.\1.attn.gate'),
    (r'transformer\.text_fusion\.refiner_blocks\.(\d+)\.attn\.to_out\.0',
     r'diffusion_model.txtfusion.refiner_blocks.\1.attn.wo'),
    (r'transformer\.text_fusion\.refiner_blocks\.(\d+)\.ff\.(up|down|gate)',
     r'diffusion_model.txtfusion.refiner_blocks.\1.mlp.\2'),
]


def convert_key(key: str) -> str | None:
    """Convert a key from alternate to standard convention.
    Returns None if no conversion was needed."""
    for pattern, replacement in KEY_RULES:
        if re.search(pattern, key):
            return re.sub(pattern, replacement, key)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert KREA2 LoRA key naming conventions"
    )
    parser.add_argument("input", help="Input safetensors file")
    parser.add_argument("--out", default=None, help="Output file (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    sd = load_file(args.input)
    
    keys = list(sd.keys())
    converted = 0
    skipped = 0
    
    new_sd = {}
    
    for k in keys:
        new_key = convert_key(k)
        if new_key:
            if new_key in new_sd:
                print(f"  ⚠ Duplicate key: {new_key}")
            new_sd[new_key] = sd[k]
            converted += 1
            if args.dry_run:
                print(f"  {k:80s}  →  {new_key}")
        else:
            # Check if it's already in standard format
            if 'diffusion_model.' in k:
                skipped += 1
                new_sd[k] = sd[k]
            elif '.lora_A.weight' in k or '.lora_B.weight' in k:
                print(f"  ⚠ Unknown key pattern: {k}")
                new_sd[k] = sd[k]
                skipped += 1
            else:
                # Non-LoRA tensors (alpha, metadata, etc.)
                new_sd[k] = sd[k]
                skipped += 1
    
    print(f"\nConverted: {converted} keys")
    print(f"Skipped:   {skipped} keys")
    print(f"Total:     {len(new_sd)} tensors")
    
    if args.dry_run:
        print("\nDry run complete.")
        return
    
    output_path = args.out or args.input
    print(f"\nSaving: {output_path} ... ", end="", flush=True)
    save_file(new_sd, output_path)
    print("done.")


if __name__ == "__main__":
    main()
