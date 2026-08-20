#!/usr/bin/env python3
"""
Merge multiple MiniMax H3 LoRAs into one via stack-and-reduce.

For each layer tensor:
  1. Stack all A factors vertically, all B factors horizontally
  2. Weight each by sqrt(w_i / sum(w)) → weighted average of (B_i @ A_i)
  3. Reduce from stacked rank (N*R) back to target rank via fast QR trick

Preserves ai-toolkit metadata (ss_base_model_version = minimax_h3) and updates
ss_output_name. Validates that inputs look like H3 LoRAs (diffusion_model
prefix, qkv_proj/fc1/fc2 layers) to avoid silently merging krea2 files.

Usage:
  python merge_h3_loras.py /path/to/h3_folder --rank 32
  python merge_h3_loras.py --files a.sft b.sft c.sft --rank 32 --out merged.safetensors
  python merge_h3_loras.py /path/to/h3_folder --weights 0.7,0.3 --rank 32
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


# Layers unique to H3 architecture (present in MiniMax H3 but NOT krea2)
H3_LAYER_MARKERS = ["qkv_proj", "token_refiner", ".fc1.", ".fc2.", "qkv_proj", "lora_down"]
# Layers unique to krea2 (present in krea2 loras but NOT H3)
KREA2_LAYER_MARKERS = [".gate.", ".wk.", ".wv.", ".wo.", "adaln_proj"]

# Mapping from Krea2-style key names to standard H3 key names
# Krea2: lora_unet_blocks_{N}_{layer}.lora_down.weight / lora_up.weight
# H3:    diffusion_model.blocks.{N}.{layer}.lora_A.weight / lora_B.weight
KREA2_LAYER_MAP = {
    "adaln_proj_linear": "adaln_proj.linear",
    "attn_out_proj": "attn.out_proj",
    "attn_qkv_proj": "attn.qkv_proj",
    "mlp_fc1": "mlp.fc1",
    "mlp_fc2": "mlp.fc2",
}


def krea2_to_h3_key(key):
    """Convert Krea2-style key to H3-style key.

    lora_unet_blocks_5_adaln_proj_linear.lora_down.weight
      -> diffusion_model.blocks.5.adaln_proj.linear.lora_A.weight

    lora_unet_token_refiner_blocks_0_mlp_fc1.lora_up.weight
      -> diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_B.weight

    Returns None if the key doesn't match Krea2 pattern.
    """
    import re
    # Pattern: lora_unet_blocks_{N}_{layer}.{suffix}
    m = re.match(
        r"lora_unet_blocks_(\d+)_([a-z0-9_]+)\.lora_(down|up)\.weight",
        key,
    )
    if m:
        block_n, layer_name, direction = m.groups()
        h3_layer = KREA2_LAYER_MAP.get(layer_name)
        if h3_layer:
            suffix = "lora_A.weight" if direction == "down" else "lora_B.weight"
            return f"diffusion_model.blocks.{block_n}.{h3_layer}.{suffix}"
    # Pattern: lora_unet_token_refiner_blocks_{N}_{layer}.{suffix}
    m = re.match(
        r"lora_unet_token_refiner_blocks_(\d+)_([a-z0-9_]+)\.lora_(down|up)\.weight",
        key,
    )
    if m:
        block_n, layer_name, direction = m.groups()
        h3_layer = KREA2_LAYER_MAP.get(layer_name)
        if h3_layer:
            suffix = "lora_A.weight" if direction == "down" else "lora_B.weight"
            return f"diffusion_model.token_refiner.blocks.{block_n}.{h3_layer}.{suffix}"
    return None


def normalize_keys(sd):
    """Normalize state dict keys to H3 standard format.

    Detects Krea2-style naming (lora_unet_blocks_X_...) and converts to
    diffusion_model.blocks.X....lora_A.weight / lora_B.weight format.

    Also handles alpha keys: .alpha -> lora_alpha (scalar metadata).
    Returns a new state dict with normalized keys.
    """
    import re
    new_sd = {}
    for key, val in sd.items():
        # Try Krea2 -> H3 conversion for weight tensors
        new_key = krea2_to_h3_key(key)
        if new_key:
            new_sd[new_key] = val
        elif key.endswith(".alpha"):
            # Convert alpha keys too
            base = key.rsplit(".", 1)[0]
            new_base = krea2_to_h3_key(base + ".lora_down.weight")
            if new_base:
                # lora_alpha lives alongside lora_A/lora_B
                alpha_key = new_base.replace(".lora_A.weight", ".lora_alpha")
                new_sd[alpha_key] = val
            else:
                new_sd[key] = val
        else:
            new_sd[key] = val
    return new_sd


def detect_key_style(sd):
    """Return 'krea2' or 'h3' based on key naming convention."""
    keys = list(sd.keys())
    has_krea2 = any("lora_unet_blocks_" in k for k in keys)
    has_h3 = any("diffusion_model.blocks." in k and "lora_A.weight" in k for k in keys)
    if has_krea2:
        return "krea2"
    if has_h3:
        return "h3"
    return "unknown"


def looks_like_h3(sd):
    """Return ('h3'|'krea2'|'unknown', reason) for a state dict."""
    keys = list(sd.keys())
    if not any("lora_A.weight" in k or "lora_down.weight" in k for k in keys):
        return "unknown", "no LoRA weight tensors found"
    joined = "|" + "|".join(keys)
    h3_hits = sum(1 for m in H3_LAYER_MARKERS if m in joined)
    krea2_hits = sum(1 for m in KREA2_LAYER_MARKERS if m in joined)
    if h3_hits >= 2 and krea2_hits == 0:
        return "h3", "qkv_proj/fc1/fc2/token_refiner layers detected"
    if krea2_hits >= 2 and h3_hits == 0:
        return "krea2", "gate/wk/wv/wo layers detected"
    if h3_hits > krea2_hits:
        return "h3", f"mixed markers (h3:{h3_hits}, krea2:{krea2_hits})"
    if krea2_hits > 0:
        return "krea2", f"mixed markers (h3:{h3_hits}, krea2:{krea2_hits})"
    # Check for Krea2 H3 naming
    if any("lora_unet_blocks_" in k for k in keys):
        return "h3", "Krea2-style H3 LoRA (lora_unet_blocks_X pattern)"
    return "unknown", "no recognizable layer markers"


def load_metadata(path):
    """Load safetensors header metadata without loading tensors."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return dict(f.metadata()) if f.metadata() else {}


def reduce_lora_fast(A, B, rank, device="cpu"):
    """
    Reduce stacked LoRA rank using the fast QR trick.

    A: [N*R, d_in] — stacked lora_A weights
    B: [d_out, N*R] — stacked lora_B weights
    rank: target rank (R')

    Returns (A_new, B_new): [rank, d_in], [d_out, rank]

    Strategy: decompose the larger dimension first to preserve all singular values.
    - If A is tall (N*R > d_in): QR-decompose B, then SVD of (R_B @ A)
    - Otherwise: QR-decompose A, then SVD of (R_A.T @ B)
    """
    stacked_rank = A.shape[0]
    if rank >= stacked_rank:
        return A.cpu(), B.cpu()

    orig_dtype = A.dtype
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)

    n_r, d_in = A.shape  # [N*R, d_in]
    d_out = B.shape[0]   # [d_out, N*R]

    if n_r >= d_in:
        # A is tall or square: QR-decompose B to reduce columns
        # B = Q_B @ R_B  where Q_B: [d_out, N*R], R_B: [N*R, N*R]
        Q_B, R_B = torch.linalg.qr(B)
        # M = R_B @ A  is [N*R, d_in]
        M = R_B @ A
        # SVD of M
        U_M, S, Vh_M = torch.linalg.svd(M, full_matrices=False)
        # Truncate to target rank
        U_r = U_M[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh_M[:rank, :]
        sqrt_S = torch.sqrt(S_r)
        # Reconstruct: A_new = sqrt(S) @ Vh @ I = sqrt(S) @ Vh
        #              B_new = Q_B @ U @ sqrt(S)
        A_new = torch.diag(sqrt_S) @ Vh_r    # [rank, d_in]
        B_new = Q_B @ U_r @ torch.diag(sqrt_S)  # [d_out, rank]
    else:
        # A is wide: QR-decompose A to reduce rows
        # A.T = Q @ R  => A = R.T @ Q.T
        Q_qr, R_qr = torch.linalg.qr(A.T)
        L_A = R_qr.T      # [N*R, N*R]
        Q_A = Q_qr.T      # [N*R, d_in]
        # QR of B
        Q_B, R_B = torch.linalg.qr(B)
        # M = R_B @ L_A  is [N*R × N*R]
        M = R_B @ L_A
        U_M, S, Vh_M = torch.linalg.svd(M, full_matrices=False)
        U_r = U_M[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh_M[:rank, :]
        sqrt_S = torch.sqrt(S_r)
        A_new = torch.diag(sqrt_S) @ Vh_r @ Q_A   # [rank, d_in]
        B_new = Q_B @ U_r @ torch.diag(sqrt_S)     # [d_out, rank]

    return A_new.to(dtype=orig_dtype, device='cpu'), \
           B_new.to(dtype=orig_dtype, device='cpu')


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple MiniMax H3 LoRAs into one via stack-and-reduce"
    )
    parser.add_argument(
        "folder", nargs="?",
        help="Folder containing safetensors files to merge (all .safetensors in folder)"
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Explicit list of LoRA files (overrides folder mode)"
    )
    parser.add_argument(
        "--rank", type=int, default=32,
        help="Target rank for merged LoRA (default: 32, H3 native)"
    )
    parser.add_argument(
        "--weights", default=None,
        help="Comma-separated merge weights, one per input file (default: equal)"
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

    # ---------------------------------------------------------------
    # Resolve input files
    # ---------------------------------------------------------------
    if args.files:
        files = [Path(f) for f in args.files]
    elif args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        exclude_names = set(args.exclude)
        files = [f for f in sorted(folder.glob("*.safetensors"))
                 if f.name not in exclude_names]
    else:
        parser.error("provide either a folder or --files")

    if len(files) < 2:
        raise SystemExit("Need at least 2 LoRA files to merge.")

    # Weights
    if args.weights:
        w = [float(x) for x in args.weights.split(",")]
        if len(w) != len(files):
            raise SystemExit(f"--weights has {len(w)} entries but {len(files)} files")
        if any(x < 0 for x in w):
            raise SystemExit("--weights must be non-negative")
        total = sum(w)
        if total <= 0:
            raise SystemExit("--weights must sum to > 0")
        weights = [x / total for x in w]
    else:
        weights = [1.0 / len(files)] * len(files)

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Merging {len(files)} H3 LoRAs -> target rank {args.rank} (device: {device})")
    print(f"Weights: {', '.join(f'{x:.3f}' for x in weights)}\n")

    for f, w in zip(files, weights):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {size_mb:>7.1f}MB  w={w:.3f}  {f.name}")

    if args.dry_run:
        print("\nDry run complete.")
        return

    # ---------------------------------------------------------------
    # Load + validate
    # ---------------------------------------------------------------
    print(f"\nLoading {len(files)} LoRAs ... ", end="", flush=True)
    t0 = time.time()
    all_sd = []
    families = []
    styles = []
    for f in files:
        sd = load_file(str(f))
        fam, why = looks_like_h3(sd)
        style = detect_key_style(sd)
        families.append(fam)
        styles.append(style)
        if style == "krea2":
            print(f"\n  {f.name}: Krea2 naming detected, normalizing to H3 format")
            sd = normalize_keys(sd)
        all_sd.append(sd)
        if fam != "h3":
            print(f"\n  ⚠ {f.name}: {fam} ({why})")
    print(f"done ({format_time(time.time()-t0)})")

    if any(fam != "h3" for fam in families):
        print("\n⚠ Some inputs don't look like H3 LoRAs. Continuing anyway, but")
        print("  verify layer names before using the merged file in ComfyUI.")

    # ---------------------------------------------------------------
    # Collect keys
    # ---------------------------------------------------------------
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

        A_list, B_list, w_list = [], [], []
        for sd, w in zip(all_sd, weights):
            if a_key in sd and b_key in sd:
                A_list.append(sd[a_key])
                B_list.append(sd[b_key])
                w_list.append(w)

        n_valid = len(A_list)
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

        orig_dtype = A_list[0].dtype

        # Fast path: all-zero tensors
        if all(t.abs().max().item() == 0 for t in A_list) and \
           all(t.abs().max().item() == 0 for t in B_list):
            new_sd[a_key] = A_list[0]
            new_sd[b_key] = B_list[0]
            print(f"  {layer_idx}/{total_pairs}  ⏭  all zeros, copied first")
            continue

        # Stack + weighted scale. Effective contribution of LoRA i is
        # B_i @ A_i scaled by w_i. Scale each side by sqrt(w_i) so the
        # product carries exactly w_i (with weights already normalized).
        # Each LoRA i contributes orig_ranks[i] rows/cols, so tile the weight.
        A_stacked = torch.cat(A_list, dim=0)   # [N*r, d_in]
        B_stacked = torch.cat(B_list, dim=1)   # [d_out, N*r]
        sqrt_blocks = []
        for w_i, r_i in zip(w_list, orig_ranks):
            sqrt_blocks.extend([w_i ** 0.5] * r_i)
        sqrt_w = torch.tensor(sqrt_blocks, dtype=torch.float32)
        A_stacked = (A_stacked.float() * sqrt_w[:, None]).to(dtype=orig_dtype)
        B_stacked = (B_stacked.float() * sqrt_w[None, :]).to(dtype=orig_dtype)

        t1 = time.time()
        try:
            A_new, B_new = reduce_lora_fast(A_stacked, B_stacked, target_rank, compute_device)
        except torch.cuda.OutOfMemoryError:
            print(f"  {layer_idx}/{total_pairs}  ⚠ OOM on layer, retrying CPU ... ", end="", flush=True)
            A_new, B_new = reduce_lora_fast(A_stacked, B_stacked, target_rank, "cpu")
        t_layer = time.time() - t1

        A_new = A_new.to(dtype=orig_dtype)
        B_new = B_new.to(dtype=orig_dtype)
        new_sd[a_key] = A_new
        new_sd[b_key] = B_new

        base = a_key.replace(".lora_A.weight", "")
        ranks_str = f"{min(orig_ranks)}–{max(orig_ranks)}→{target_rank}"
        print(f"  {layer_idx}/{total_pairs}  ✅ {base:.<55} {ranks_str:>12s}  {format_time(t_layer):>6s}")

    # Copy non-LoRA tensors from the first LoRA that has them
    for k in lone_keys:
        for sd in all_sd:
            if k in sd:
                new_sd[k] = sd[k]
                break

    # ---------------------------------------------------------------
    # Output path + metadata
    # ---------------------------------------------------------------
    if args.out is None:
        folder = files[0].parent
        out_name = f"merged_rank{args.rank}.safetensors"
        args.out = str(folder / out_name)

    out_stem = Path(args.out).stem
    meta = load_metadata(str(files[0]))
    meta["ss_output_name"] = out_stem
    meta["name"] = out_stem
    if "ss_base_model_version" not in meta:
        meta["ss_base_model_version"] = "minimax_h3"
    meta["ss_merged_from"] = ",".join(f.name for f in files)
    meta["ss_merge_weights"] = ",".join(f"{w:.4f}" for w in weights)
    meta["ss_merge_rank"] = str(args.rank)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    print(f"\nSaving: {args.out} ... ", end="", flush=True)
    t1 = time.time()
    save_file(new_sd, args.out, metadata=meta)
    print(f"done ({format_time(time.time()-t1)})")

    out_size = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"  Input files    : {len(files)}")
    print(f"  Weights        : {', '.join(f'{x:.3f}' for x in weights)}")
    print(f"  Layers merged  : {total_pairs}")
    print(f"  Output rank    : {args.rank}")
    print(f"  Output size    : {out_size:.1f} MB")
    print(f"  Output file    : {args.out}")
    print(f"  Metadata       : ss_base_model_version={meta.get('ss_base_model_version')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
