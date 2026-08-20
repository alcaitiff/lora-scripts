#!/usr/bin/env python3
"""
Merge multiple KREA2 LoRAs (same rank) into one via stack-and-reduce.

For each layer tensor:
  1. Stack all A factors vertically, all B factors horizontally
  2. Scale by 1/sqrt(N) → gives true average: (1/N) * sum(B_i @ A_i)
  3. Reduce from stacked rank (N*R) back to target rank via fast QR trick

Usage:
  python merge_krea2_loras.py /path/to/folder/*.safetensors --rank 32
  python merge_krea2_loras.py --files lora1.sft lora2.sft lora3.sft --rank 32
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds/60:.1f}m {seconds%60:.0f}s"


def reduce_lora_fast(A, B, rank, device="cpu"):
    """
    Reduce stacked LoRA rank using the fast QR trick.
    
    A: [N*R, d_in] — stacked lora_A weights
    B: [d_out, N*R] — stacked lora_B weights
    rank: target rank (R')
    
    Returns (A_new, B_new): [rank, d_in], [d_out, rank]
    """
    if rank >= A.shape[0]:
        return A.cpu(), B.cpu()

    orig_dtype = A.dtype
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)

    # LQ decomposition of A: A = L_A @ Q_A  (Q_A orthonormal rows)
    # Done via QR of A^T
    Q_qr, R_qr = torch.linalg.qr(A.T)
    L_A = R_qr.T      # [N*R, N*R] lower triangular
    Q_A = Q_qr.T      # [N*R, d_in] orthonormal rows

    # QR decomposition of B: B = Q_B @ R_B
    Q_B, R_B = torch.linalg.qr(B)

    # M = R_B @ L_A  is [N*R × N*R]
    M = R_B @ L_A

    # SVD of the small matrix
    U_M, S, Vh_M = torch.linalg.svd(M, full_matrices=False)

    # Truncate to target rank
    U_r = U_M[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh_M[:rank, :]
    sqrt_S = torch.sqrt(S_r)

    # Reconstruct
    A_new = torch.diag(sqrt_S) @ Vh_r @ Q_A   # [rank, d_in]
    B_new = Q_B @ U_r @ torch.diag(sqrt_S)     # [d_out, rank]

    return A_new.to(dtype=orig_dtype, device='cpu'), \
           B_new.to(dtype=orig_dtype, device='cpu')


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple KREA2 LoRAs into one via stack-and-reduce"
    )
    parser.add_argument(
        "folder",
        help="Folder containing safetensors files to merge (all .safetensors in folder)"
    )
    parser.add_argument(
        "--rank", type=int, default=32,
        help="Target rank for merged LoRA (default: 32)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Output merged safetensors file (default: <folder>/merged_rank{R}.safetensors)"
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        help="Filenames to exclude (e.g., original_rank128.safetensors)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: 'cuda', 'cpu', or auto-detect"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan files and show what would be merged without processing"
    )
    args = parser.parse_args()

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = "cpu"

    # Collect files
    folder = Path(args.folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    exclude_names = set(args.exclude)
    all_files = sorted(folder.glob("*.safetensors"))
    
    # If _rank32 versions exist, prefer them and exclude originals
    # Auto-detect: for any "base_rank32.sft", exclude "base.sft" if it exists
    rank32_names = {f.stem.replace("_rank32", "") for f in all_files if "_rank32" in f.stem}
    auto_exclude = set()
    for f in all_files:
        stem = f.stem
        if stem in rank32_names and stem not in exclude_names:
            auto_exclude.add(f.name)
            print(f"  (auto-excluding original: {f.name} — _rank32 version exists)")

    files = [f for f in all_files if f.name not in exclude_names and f.name not in auto_exclude]
    
    print(f"Found {len(all_files)} .safetensors files in {folder}")
    if auto_exclude:
        print(f"Auto-excluded {len(auto_exclude)} original(s) with _rank32 counterpart")
    if exclude_names:
        print(f"User-excluded: {exclude_names}")
    print(f"Merging {len(files)} files\n")

    for f in files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {size_mb:>7.1f}MB  {f.name}")

    if args.dry_run:
        print("\nDry run complete.")
        return

    # Output path
    if args.out is None:
        out_name = f"merged_rank{args.rank}.safetensors"
        args.out = str(folder / out_name)

    # ---------------------------------------------------------------
    # Load all LoRAs
    # ---------------------------------------------------------------
    print(f"\nLoading {len(files)} LoRAs ... ", end="", flush=True)
    t0 = time.time()
    
    all_sd = []
    for f in files:
        sd = load_file(str(f))
        all_sd.append(sd)
    
    print(f"done ({format_time(time.time()-t0)})")

    # Collect all unique keys
    all_keys = set()
    for sd in all_sd:
        all_keys.update(sd.keys())
    
    a_keys = sorted([k for k in all_keys if ".lora_A.weight" in k])
    lone_keys = [k for k in all_keys if ".lora_A.weight" not in k and ".lora_B.weight" not in k]
    
    print(f"Total unique tensors: {len(all_keys)}")
    print(f"LoRA-A pairs: {len(a_keys)}")
    if lone_keys:
        print(f"Non-LoRA tensors (will copy through): {len(lone_keys)}")
    
    # ---------------------------------------------------------------
    # Merge each A/B pair
    # ---------------------------------------------------------------
    print(f"\nMerging layers ...\n")
    
    new_sd = {}
    total_pairs = len(a_keys)
    compute_device = device

    for layer_idx, a_key in enumerate(a_keys, 1):
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        
        # Collect all A_i and B_i for this key
        A_list = []
        B_list = []
        n_valid = 0
        
        for sd in all_sd:
            if a_key in sd and b_key in sd:
                A_list.append(sd[a_key])
                B_list.append(sd[b_key])
                n_valid += 1
        
        if n_valid < 2:
            base = a_key.replace(".lora_A.weight", "")
            print(f"  {layer_idx}/{total_pairs}  ⚠  {base:.<55} only in {n_valid} LoRA(s), skipping")
            if n_valid == 1:
                new_sd[a_key] = A_list[0]
                new_sd[b_key] = B_list[0]
            continue

        # Determine which rank to target
        orig_ranks = [A.shape[0] for A in A_list]
        max_rank = max(orig_ranks)
        target_rank = min(args.rank, max_rank)
        
        # Stack: vertically for A, horizontally for B
        A_stacked = torch.cat(A_list, dim=0)   # [N*r, d_in]
        B_stacked = torch.cat(B_list, dim=1)   # [d_out, N*r]
        
        # Determine original dtype (all should be the same)
        orig_dtype = A_list[0].dtype
        
        # Detect if all values are zero before any dtype promotion
        if all((t.abs().max().item() == 0 for t in A_list)) and all((t.abs().max().item() == 0 for t in B_list)):
            new_sd[a_key] = A_list[0]
            new_sd[b_key] = B_list[0]
            print(f"  {layer_idx}/{total_pairs}  \u23af  all zeros, copied first")
            continue
        
        # Stack: vertically for A, horizontally for B
        A_stacked = torch.cat(A_list, dim=0)   # [N*r, d_in]
        B_stacked = torch.cat(B_list, dim=1)   # [d_out, N*r]
        
        # Scale by 1/sqrt(N) for averaging.
        # Division promotes bfloat16 to float32; cast back to preserve dtype.
        scale = n_valid ** 0.5
        A_scaled = (A_stacked.float() / scale).to(dtype=orig_dtype)
        B_scaled = (B_stacked.float() / scale).to(dtype=orig_dtype)
        
        t1 = time.time()
        try:
            A_new, B_new = reduce_lora_fast(A_scaled, B_scaled, target_rank, compute_device)
        except torch.cuda.OutOfMemoryError:
            print(f"  {layer_idx}/{total_pairs}  \u26a0 OOM on layer, retrying CPU ... ", end="", flush=True)
            A_new, B_new = reduce_lora_fast(A_scaled, B_scaled, target_rank, "cpu")
        t_layer = time.time() - t1
        
        # Ensure output dtype matches the original LoRA dtype
        A_new = A_new.to(dtype=orig_dtype)
        B_new = B_new.to(dtype=orig_dtype)
        new_sd[a_key] = A_new
        new_sd[b_key] = B_new
        
        base = a_key.replace(".lora_A.weight", "")
        ranks_str = f"{min(orig_ranks)}–{max(orig_ranks)}→{target_rank}"
        print(f"  {layer_idx}/{total_pairs}  ✅ {base:.<55} {ranks_str:>12s}  {format_time(t_layer):>6s}")
    
    # Copy non-LoRA tensors from the first LoRA
    for k in lone_keys:
        # Find the first LoRA that has this key
        for sd in all_sd:
            if k in sd:
                new_sd[k] = sd[k]
                break
    
    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    print(f"\nSaving: {args.out} ... ", end="", flush=True)
    t1 = time.time()
    save_file(new_sd, args.out)
    print(f"done ({format_time(time.time()-t1)})")
    
    out_size = os.path.getsize(args.out) / (1024*1024)
    print(f"\n{'=' * 60}")
    print(f"  Input files    : {len(files)}")
    print(f"  Layers merged  : {total_pairs}")
    print(f"  Output rank    : {args.rank}")
    print(f"  Output size    : {out_size:.1f} MB")
    print(f"  Output file    : {args.out}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
