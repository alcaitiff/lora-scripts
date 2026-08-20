#!/usr/bin/env python3
"""
Expand a KREA2 LoRA from low rank to higher rank via zero-padding.

For rank r → target R (where R > r):
  Pad lora_A with R-r zero rows:  A_new = [A; 0]   →  [R, d_in]
  Pad lora_B with R-r zero cols:  B_new = [B, 0]   →  [d_out, R]

This is functionally identical to the original (B_new @ A_new = B @ A),
just with extra zero dimensions to match the target rank shape.

Usage:
  python expand_lora_rank.py input.safetensors --rank 32
"""

import argparse
import os
import time
import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(
        description="Expand KREA2 LoRA rank via zero-padding (fast)"
    )
    parser.add_argument("input", help="Input LoRA safetensors file")
    parser.add_argument("--rank", type=int, required=True, help="Target rank")
    parser.add_argument("--out", default=None, help="Output file")
    args = parser.parse_args()

    output_path = args.out
    if output_path is None:
        base, ext = os.path.splitext(args.input)
        if not ext:
            ext = ".safetensors"
        output_path = f"{base}_rank{args.rank}{ext}"

    print(f"Loading: {args.input} ... ", end="", flush=True)
    t0 = time.time()
    sd = load_file(args.input)
    print(f"done ({time.time()-t0:.2f}s)")
    print(f"Tensors: {len(sd)}")
    print()

    new_sd = {}
    processed = set()
    target = args.rank

    lora_pairs = sum(1 for k in sd.keys() if ".lora_A.weight" in k)
    layer_idx = 0

    for k in list(sd.keys()):
        if ".lora_A.weight" in k:
            layer_idx += 1
            base = k.replace(".lora_A.weight", "")
            B_key = base + ".lora_B.weight"

            A = sd[k]
            B = sd.get(B_key)

            if B is None:
                print(f"  {layer_idx}/{lora_pairs}  ⚠ Missing lora_B for {base}, copy")
                new_sd[k] = A
                continue

            r = A.shape[0]  # current rank
            if target <= r:
                print(f"  {layer_idx}/{lora_pairs}  ⏭  {base}  (rank {r} >= {target})")
                new_sd[k] = A
                new_sd[B_key] = B
                processed.add(base)
                continue

            # Zero-pad
            d_in = A.shape[1]
            d_out = B.shape[0]
            pad_rows = target - r

            A_new = torch.cat([A, torch.zeros(pad_rows, d_in, dtype=A.dtype, device='cpu')], dim=0)
            B_new = torch.cat([B, torch.zeros(d_out, pad_rows, dtype=B.dtype, device='cpu')], dim=1)

            new_sd[base + ".lora_A.weight"] = A_new
            new_sd[base + ".lora_B.weight"] = B_new
            processed.add(base)

            print(f"  {layer_idx}/{lora_pairs}  ✅ {base:.<50} {r}→{target}")

        elif ".lora_B.weight" in k:
            base = k.replace(".lora_B.weight", "")
            if base not in processed:
                new_sd[k] = sd[k]
        else:
            new_sd[k] = sd[k]

    print(f"\nSaving: {output_path} ... ", end="", flush=True)
    t1 = time.time()
    save_file(new_sd, output_path)
    print(f"done ({time.time()-t1:.2f}s)")

    print()
    print("=" * 60)
    print(f"  Input   : {args.input}")
    print(f"  Output  : {output_path}")
    print(f"  Target rank: {target}")
    print(f"  Layers processed: {len(processed)}")
    print(f"  Saved tensors: {len(new_sd)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
