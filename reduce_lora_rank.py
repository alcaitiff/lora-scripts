#!/usr/bin/env python3
"""
Fast LoRA rank reduction using a QR-decomposition trick.

*** WHY THIS IS FAST ***

The OLD code computed W = B @ A (a large [d_out × d_in] matrix) and then ran
full SVD on it — O(d_out × d_in × min(d_out, d_in)) per layer.

This NEW code NEVER materializes W. Instead:

    B = Q_B @ R_B          (QR decomposition of B)
    A = L_A @ Q_A          (LQ decomposition of A via QR on transpose)

    W = B @ A = Q_B @ (R_B @ L_A) @ Q_A = Q_B @ M @ Q_A

M = R_B @ L_A is only [R × R] — tiny! SVD of M costs O(R³) instead of
O(d_out × d_in × min(d_out, d_in)).

For a typical layer with R=64, d_in=768, d_out=768:
    Old: ~38 million FLOP + SVD of 768×768 matrix
    New: ~200K FLOP + SVD of 64×64 matrix
    >>> ~100-200x faster for common sizes.

Usage:
    python reduce_lora_rank.py model.safetensors --rank 8
    python reduce_lora_rank.py model.safetensors --rank 16 --out reduced.safetensors
    python reduce_lora_rank.py model.safetensors --rank 8 --device cpu
"""

import argparse
import os
import sys
import time
import torch
from safetensors.torch import load_file, save_file


def reduce_lora_fast(A, B, rank, device="cpu"):
    """
    Reduce LoRA rank using the QR trick.
    
    A: [R, d_in]   — lora_A.weight
    B: [d_out, R]  — lora_B.weight
    rank: target rank r
    
    Returns (A_new, B_new) where A_new is [r, d_in], B_new is [d_out, r].
    """
    orig_dtype = A.dtype
    orig_rank = A.shape[0]

    if rank >= orig_rank:
        return A.cpu().to(orig_dtype), B.cpu().to(orig_dtype)

    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)

    # ---------------------------------------------------------------
    # LQ decomposition of A via QR of A^T
    #   A^T = Q_qr @ R_qr   →   A = R_qr^T @ Q_qr^T = L_A @ Q_A
    # ---------------------------------------------------------------
    # A is [R, d_in]. A^T is [d_in, R].
    # Q_qr: [d_in, R] orthonormal columns. R_qr: [R, R] upper triangular.
    Q_qr, R_qr = torch.linalg.qr(A.T)
    L_A = R_qr.T   # [R, R] lower triangular
    Q_A = Q_qr.T   # [R, d_in] orthonormal rows (Q_A @ Q_A^T = I_R)

    # ---------------------------------------------------------------
    # QR decomposition of B
    #   B = Q_B @ R_B
    # ---------------------------------------------------------------
    # B is [d_out, R].
    # Q_B: [d_out, R] orthonormal columns. R_B: [R, R] upper triangular.
    Q_B, R_B = torch.linalg.qr(B)

    # ---------------------------------------------------------------
    # M = R_B @ L_A is tiny [R × R] — this is the key!
    # W = Q_B @ M @ Q_A
    # ---------------------------------------------------------------
    M = R_B @ L_A

    # SVD of the small matrix — FAST
    U_M, S, Vh_M = torch.linalg.svd(M, full_matrices=False)

    # Truncate
    U_r = U_M[:, :rank]    # [R, r]
    S_r = S[:rank]          # [r]
    Vh_r = Vh_M[:rank, :]   # [r, R]

    sqrt_S = torch.sqrt(S_r)

    # Reconstruct new LoRA factors
    A_new = torch.diag(sqrt_S) @ Vh_r @ Q_A   # [r, d_in]
    B_new = Q_B @ U_r @ torch.diag(sqrt_S)     # [d_out, r]

    return A_new.to(dtype=orig_dtype, device='cpu'), B_new.to(dtype=orig_dtype, device='cpu')


def format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds/60:.1f}m {seconds%60:.0f}s"


def main():
    parser = argparse.ArgumentParser(
        description="Reduce LoRA rank using fast QR-based SVD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python reduce_lora_rank.py model.safetensors --rank 8\n"
            "  python reduce_lora_rank.py model.safetensors --rank 16 --out reduced.safetensors\n"
            "  python reduce_lora_rank.py model.safetensors --rank 4 --device cpu\n"
        ),
    )
    parser.add_argument("input", help="Input LoRA safetensors file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output LoRA safetensors file (optional)")
    parser.add_argument("--out", default=None, help="Output LoRA safetensors file")
    parser.add_argument("--rank", type=int, required=True, help="Target rank")
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda', 'cpu', or auto-detect (default)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-layer timing")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA requested but not available, falling back to CPU")
        device = "cpu"

    print(f"Device    : {device}")
    print(f"Target rank: {args.rank}")
    print()

    compute_device = device

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------
    output_path = args.out or args.output
    if output_path is None:
        base, ext = os.path.splitext(args.input)
        if not ext:
            ext = ".safetensors"
        output_path = f"{base}_rank{args.rank}{ext}"

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"Loading: {args.input} ... ", end="", flush=True)
    t0 = time.time()
    sd = load_file(args.input)
    t_load = time.time() - t0
    print(f"done ({format_time(t_load)})")
    print(f"Tensors : {len(sd)}")
    print()

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------
    new_sd = {}
    processed = set()
    total_time = 0.0

    # Count total LoRA pairs for progress
    lora_pairs = 0
    for k in sd.keys():
        if ".lora_A.weight" in k:
            lora_pairs += 1

    layer_idx = 0
    for k in sd.keys():
        if ".lora_A.weight" in k:
            layer_idx += 1
            base = k.replace(".lora_A.weight", "")

            A = sd[k]
            B_key = base + ".lora_B.weight"
            if B_key not in sd:
                print(f"  ⚠ Missing lora_B for {base}, copying A as-is")
                new_sd[k] = A
                continue

            B = sd[B_key]
            orig_rank = A.shape[0]
            target = args.rank

            if target >= orig_rank:
                print(f"  {layer_idx}/{lora_pairs}  ⏭  {base}  (rank {orig_rank} already ≤ {target})")
                new_sd[k] = A
                new_sd[B_key] = B
                continue

            t1 = time.time()
            try:
                A_new, B_new = reduce_lora_fast(A, B, target, device=compute_device)
            except torch.cuda.OutOfMemoryError:
                print(f"  {layer_idx}/{lora_pairs}  ⚠ OOM on {base}, retrying on CPU...")
                A_new, B_new = reduce_lora_fast(A, B, target, device="cpu")
            t_layer = time.time() - t1
            total_time += t_layer

            new_sd[base + ".lora_A.weight"] = A_new
            new_sd[base + ".lora_B.weight"] = B_new
            processed.add(base)

            if args.verbose:
                old_size = B.numel() + A.numel()
                new_size = B_new.numel() + A_new.numel()
                saved = old_size - new_size
                pct = 100.0 * saved / old_size if old_size > 0 else 0
                print(
                    f"  {layer_idx}/{lora_pairs}  ✅ {base:.<50} "
                    f"{orig_rank}→{target}  "
                    f"{format_time(t_layer):>6}  "
                    f"params: {old_size:,} → {new_size:,} ({pct:.0f}% ↓)"
                )

        elif ".lora_B.weight" in k:
            base = k.replace(".lora_B.weight", "")
            if base not in processed:
                # Lone lora_B without matching lora_A — copy through
                new_sd[k] = sd[k]
        else:
            # Non-LoRA tensors (metadata, alpha values, etc.) — copy through
            new_sd[k] = sd[k]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print(f"\nSaving: {output_path} ... ", end="", flush=True)
    t1 = time.time()
    save_file(new_sd, output_path)
    t_save = time.time() - t1
    print(f"done ({format_time(t_save)})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  Input  : {args.input}")
    print(f"  Output : {output_path}")
    print(f"  Target rank : {args.rank}")
    print(f"  Layers processed : {len(processed)}")
    print(f"  Compute time     : {format_time(total_time)}")
    print(f"  Total time       : {format_time(t_load + total_time + t_save)}")
    if len(processed) > 0:
        avg = total_time / len(processed)
        print(f"  Avg per layer    : {format_time(avg)}")
    print(f"  Saved tensors: {len(new_sd)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
