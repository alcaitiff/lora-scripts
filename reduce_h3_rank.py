#!/usr/bin/env python3
"""
Reduce MiniMax H3 LoRA rank using a QR-decomposition trick.

The OLD approach computed W = B @ A (large [d_out x d_in]) then ran full SVD.
This version never materializes W:

    B = Q_B @ R_B          (QR of B)
    A = L_A @ Q_A          (LQ of A via QR on transpose)

    W = B @ A = Q_B @ (R_B @ L_A) @ Q_A = Q_B @ M @ Q_A

M = R_B @ L_A is only [R x R] — tiny! SVD of M costs O(R^3).

Preserves ai-toolkit metadata and validates the input looks like an H3 LoRA.

Usage:
    python reduce_h3_rank.py model.safetensors --rank 16
    python reduce_h3_rank.py model.safetensors --rank 8 --out reduced.safetensors
    python reduce_h3_rank.py model.safetensors --rank 8 --device cpu
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def load_metadata(path):
    """Load safetensors header metadata without loading tensors."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return dict(f.metadata()) if f.metadata() else {}


def looks_like_h3(keys):
    """Return ('h3'|'krea2'|'unknown', reason) from key list."""
    if not any("lora_A.weight" in k for k in keys):
        return "unknown", "no .lora_A.weight tensors found"
    joined = "|" + "|".join(keys)
    h3_hits = sum(1 for m in ["qkv_proj", "token_refiner", ".fc1.", ".fc2."] if m in joined)
    krea2_hits = sum(1 for m in [".gate.", ".wk.", ".wv.", ".wo.", "adaln_proj"] if m in joined)
    if h3_hits >= 2 and krea2_hits == 0:
        return "h3", "qkv_proj/fc1/fc2/token_refiner layers detected"
    if krea2_hits >= 2 and h3_hits == 0:
        return "krea2", "gate/wk/wv/wo layers detected"
    if h3_hits > krea2_hits:
        return "h3", f"mixed markers (h3:{h3_hits}, krea2:{krea2_hits})"
    if krea2_hits > 0:
        return "krea2", f"mixed markers (h3:{h3_hits}, krea2:{krea2_hits})"
    return "unknown", "no recognizable layer markers"


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

    # LQ decomposition of A via QR of A^T: A = L_A @ Q_A
    Q_qr, R_qr = torch.linalg.qr(A.T)
    L_A = R_qr.T   # [R, R] lower triangular
    Q_A = Q_qr.T   # [R, d_in] orthonormal rows

    # QR decomposition of B: B = Q_B @ R_B
    Q_B, R_B = torch.linalg.qr(B)

    # M = R_B @ L_A is tiny [R x R] — the key trick
    M = R_B @ L_A

    U_M, S, Vh_M = torch.linalg.svd(M, full_matrices=False)

    U_r = U_M[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh_M[:rank, :]
    sqrt_S = torch.sqrt(S_r)

    A_new = torch.diag(sqrt_S) @ Vh_r @ Q_A   # [r, d_in]
    B_new = Q_B @ U_r @ torch.diag(sqrt_S)     # [d_out, r]

    return A_new.to(dtype=orig_dtype, device='cpu'), \
           B_new.to(dtype=orig_dtype, device='cpu')


def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds/60:.1f}m {seconds%60:.0f}s"


def main():
    parser = argparse.ArgumentParser(
        description="Reduce MiniMax H3 LoRA rank using fast QR-based SVD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python reduce_h3_rank.py model.safetensors --rank 16\n"
            "  python reduce_h3_rank.py model.safetensors --rank 8 --out reduced.safetensors\n"
            "  python reduce_h3_rank.py model.safetensors --rank 4 --device cpu\n"
        ),
    )
    parser.add_argument("input", help="Input H3 LoRA safetensors file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output LoRA safetensors file (optional)")
    parser.add_argument("--out", default=None, help="Output LoRA safetensors file")
    parser.add_argument("--rank", type=int, required=True, help="Target rank")
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda', 'cpu', or auto-detect (default)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-layer timing")

    args = parser.parse_args()

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

    output_path = args.out or args.output
    if output_path is None:
        base, ext = os.path.splitext(args.input)
        if not ext:
            ext = ".safetensors"
        output_path = f"{base}_rank{args.rank}{ext}"

    print(f"Loading: {args.input} ... ", end="", flush=True)
    t0 = time.time()
    sd = load_file(args.input)
    t_load = time.time() - t0
    print(f"done ({format_time(t_load)})")
    print(f"Tensors : {len(sd)}")

    fam, why = looks_like_h3(list(sd.keys()))
    if fam != "h3":
        print(f"⚠ Input looks like: {fam} ({why})")
        print("  Expected MiniMax H3 (diffusion_model.blocks.N.qkv_proj/fc1/fc2).")
        print("  Continuing anyway — verify layer names before use in ComfyUI.")
    print()

    new_sd = {}
    processed = set()

    lora_pairs = sum(1 for k in sd if ".lora_A.weight" in k)
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
                A_new, B_new = reduce_lora_fast(A, B, target, device=device)
            except torch.cuda.OutOfMemoryError:
                print(f"  {layer_idx}/{lora_pairs}  ⚠ OOM on {base}, retrying on CPU...")
                A_new, B_new = reduce_lora_fast(A, B, target, device="cpu")
            t_layer = time.time() - t1

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
                new_sd[k] = sd[k]
        else:
            new_sd[k] = sd[k]

    # Preserve metadata, note the rank change
    # (read the actual rank from the first lora_A we processed)
    orig_rank_meta = ""
    for base in processed:
        t = sd.get(base + ".lora_A.weight")
        if t is not None:
            orig_rank_meta = str(t.shape[0])
            break
    meta = load_metadata(args.input)
    meta["ss_output_name"] = Path(output_path).stem
    meta["name"] = Path(output_path).stem
    if "ss_base_model_version" not in meta:
        meta["ss_base_model_version"] = "minimax_h3"
    if orig_rank_meta:
        meta["ss_rank_reduced_from"] = orig_rank_meta
    meta["ss_rank_reduced_to"] = str(args.rank)

    print(f"\nSaving: {output_path} ... ", end="", flush=True)
    t1 = time.time()
    save_file(new_sd, output_path, metadata=meta)
    t_save = time.time() - t1
    print(f"done ({format_time(t_save)})")

    print()
    print("=" * 60)
    print(f"  Input  : {args.input}")
    print(f"  Output : {output_path}")
    print(f"  Target rank : {args.rank}")
    print(f"  Layers processed : {len(processed)}")
    print(f"  Saved tensors: {len(new_sd)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
