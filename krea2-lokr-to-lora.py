#!/usr/bin/env python3
"""
Convert a KREA2 LoKr safetensors file to a standard LoRA (lora_A / lora_B) via SVD.

The KREA2 format stores weight deltas as Kronecker products:
  delta = alpha * kron(w1, w2)

HOWEVER: KREA2 files store alpha as a sentinel value (9999220736 = 0x254000000)
across ALL layers — it is NOT a real multiplier. The actual delta is
just kron(w1, w2). But the raw kron product has very small magnitudes
(std ~0.0008) compared to working KREA2 LoRAs (std ~0.004).

This script applies a --scale factor (default 6.0) to amplify the delta
to match the typical magnitude of native KREA2 LoRAs, then reduces to
rank-N via low-rank SVD, producing standard LoRA A/B weights.

Usage:
  ./krea2-lokr-to-lora.py snofs_krea_v1_1.safetensors          # scale=6.0 (default)
  ./krea2-lokr-to-lora.py input.safetensors --scale 3.0        # custom scale
  ./krea2-lokr-to-lora.py input.safetensors --rank 64 --scale 6.0
  ./krea2-lokr-to-lora.py input.safetensors --device cpu       # fallback if OOM
  ./krea2-lokr-to-lora.py input.safetensors --auto-scale       # infer scale from data
"""

import torch
from safetensors import safe_open
from safetensors.torch import save_file
import argparse
import os
import sys
import math


def estimate_scale_factor(state, rank=32, device='cpu'):
    """
    Estimate the scale factor so the rank-N SVD reconstruction
    has delta std ≈ 0.004 (typical KREA2 LoRA magnitude).
    
    Samples 3 layers and does full kron + SVD to measure actual
    reconstruction magnitude, then computes the needed scale.
    """
    w1_keys = [k for k in state.keys() if k.endswith('.lokr_w1')]
    if not w1_keys:
        return 6.0
    
    sample = w1_keys[:3]
    recon_stds = []
    
    for w1k in sample:
        base = w1k[:-len('.lokr_w1')]
        w2k = f"{base}.lokr_w2"
        if w2k not in state:
            continue
        w1 = state[w1k].to(device).float()
        w2 = state[w2k].to(device).float()
        kron = torch.kron(w1, w2)
        m, n = kron.shape
        r = min(rank, min(m, n))
        try:
            U, S, V = torch.svd_lowrank(kron, q=r)
            S_sqrt = torch.sqrt(S + 1e-8)
            B = (U * S_sqrt.unsqueeze(0))
            A = (S_sqrt.unsqueeze(1) * V.t())
            recon = B @ A
            recon_stds.append(recon.std().item())
        except Exception:
            recon_stds.append(kron.std().item() * 0.5)
        finally:
            del kron, U, S, V, B, A
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    if not recon_stds:
        return 6.0
    
    median_std = sorted(recon_stds)[len(recon_stds)//2]
    TARGET = 0.004
    
    if median_std > 0:
        scale = TARGET / median_std
    else:
        scale = 6.0
    
    return max(0.1, min(100.0, scale))


def convert_krea2_lokr_to_lora(krea2_path, output_path=None, rank=32, device=None, scale=None, auto_scale=False):
    """
    Converts KREA2 LoKr → standard LoRA.
    
    Recomposes kron(w1, w2) for each layer, optionally scales it,
    then reduces to the target rank via SVD_lowrank.
    """
    if not os.path.isfile(krea2_path):
        raise FileNotFoundError(f"KREA2 file not found: {krea2_path}")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if output_path is None:
        base, ext = os.path.splitext(krea2_path)
        output_path = f"{base}_lora{ext}"

    # Load state dict
    state = {}
    with safe_open(krea2_path, framework="pt", device=device) as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)

    # Auto-scale if requested
    if auto_scale and scale is None:
        scale = estimate_scale_factor(state, rank=rank, device=device)
        print(f"Auto-scale: {scale:.4f} (target delta std = 0.004)")
    elif scale is None:
        scale = 6.0
        print(f"Using default scale: {scale:.2f}")
    else:
        print(f"Using scale: {scale:.2f}")

    print(f"Input:  {krea2_path}")
    print(f"Output: {output_path}")
    print(f"Target rank: {rank}\n")

    # Discover base keys from lokr_w1 / lokr_w2 pairs
    base_keys = set()
    for k in state.keys():
        if k.endswith(('.alpha', '.lokr_w1', '.lokr_w2')):
            base = '.'.join(k.split('.')[:-1])
            base_keys.add(base)

    new_state = {}
    converted_count = 0
    failed_count = 0

    for base in sorted(base_keys):
        alpha_key = f"{base}.alpha"
        w1_key    = f"{base}.lokr_w1"
        w2_key    = f"{base}.lokr_w2"

        if not all(k in state for k in [alpha_key, w1_key, w2_key]):
            print(f"  Skip incomplete: {os.path.basename(base)}")
            continue

        w1 = state[w1_key].to(device)
        w2 = state[w2_key].to(device)

        try:
            # Recompose delta via Kronecker product
            # NOTE: alpha is always 9999220736 (sentinel), never use it
            kron_product = torch.kron(w1, w2)

            # Apply scaling factor to match typical KREA2 LoRA magnitudes
            delta = kron_product * scale

            # Cast to float32 for SVD numerical stability
            delta_float = delta.to(torch.float32)

            # Low-rank SVD decomposition
            U, S, V = torch.svd_lowrank(delta_float, q=rank)

            # Truncate to target rank
            U  = U[:, :rank]
            S  = S[:rank]
            V  = V[:, :rank]

            S_sqrt = torch.sqrt(S + 1e-8)

            # B @ A approximates delta:
            #   B: out_dim × rank  (U scaled)
            #   A: rank × in_dim   (V.t() scaled)
            B = (U * S_sqrt.unsqueeze(0)).contiguous()
            A = (S_sqrt.unsqueeze(1) * V.t()).contiguous()

            # Cast back to bfloat16
            A = A.to(torch.bfloat16).contiguous()
            B = B.to(torch.bfloat16).contiguous()

            new_state[f"{base}.lora_A.weight"] = A.cpu()
            new_state[f"{base}.lora_B.weight"] = B.cpu()

            converted_count += 1
            out_dim, in_dim = B.shape[0], A.shape[1]
            print(f"  {os.path.basename(base):40}  {out_dim:>5}×{in_dim:<5}  rank={rank}")

        except RuntimeError as e:
            failed_count += 1
            print(f"  FAIL {os.path.basename(base)}: {e}")
            continue
        except Exception as e:
            failed_count += 1
            print(f"  ERROR {os.path.basename(base)}: {e}")
            continue

    if converted_count == 0:
        print("\nNo layers converted. Verify KREA2 keys or try --device cpu if OOM.")
        return

    save_file(new_state, output_path)

    # Summary
    total_elements = sum(t.numel() for t in new_state.values())
    file_size_mb = total_elements * 2 / (1024 * 1024)

    print(f"\n{'='*50}")
    print(f"Converted: {converted_count} layers")
    if failed_count:
        print(f"Failed:    {failed_count} layers")
    print(f"Scale:     {scale:.4f}")
    print(f"Output:    {output_path}")
    print(f"Elements:  {total_elements:,}")
    print(f"Size (bf16): ~{file_size_mb:.1f} MB")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert KREA2 LoKr (.lokr_w1/.lokr_w2) to standard LoRA (lora_A/lora_B)"
    )
    parser.add_argument(
        "krea2_file",
        type=str,
        help="Input KREA2 LoKr .safetensors file"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="Target LoRA rank (default: 32)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scaling factor for delta magnitude (default: 5.0, auto-tune with --auto-scale)"
    )
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Automatically estimate scale factor from layer statistics"
    )
    parser.add_argument(
        "--output", "--out",
        type=str,
        default=None,
        help="Output path (default: <input>_lora.safetensors)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help="Device for SVD computation (default: cuda if available, else cpu)"
    )

    args = parser.parse_args()

    try:
        convert_krea2_lokr_to_lora(
            krea2_path=args.krea2_file,
            output_path=args.output,
            rank=args.rank,
            device=args.device,
            scale=args.scale,
            auto_scale=args.auto_scale
        )
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
