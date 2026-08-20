#!/usr/bin/env python3
"""
Calculate cosine similarity between two LoRA safetensors files.

Reports:
  - Overall weighted cosine similarity (weighted by tensor size)
  - Per-layer cosine similarity with detailed breakdown
  - Top-N most similar and most dissimilar layers
  - Distribution of similarity scores

Usage:
  python lora_similarity.py --lora_a A.safetensors --lora_b B.safetensors
  python lora_similarity.py --lora_a A.safetensors --lora_b B.safetensors --csv
  python lora_similarity.py --lora_a A.safetensors --lora_b B.safetensors --top 20
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import torch
from safetensors.torch import load_file


# ── Helpers ──

def strip_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened tensors.

    Promotes to float32 for accurate computation (avoids bfloat16/fp16 issues
    where dot(x, x) != ‖x‖² due to low precision).
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    num = torch.dot(a_flat, b_flat)
    den = torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat)
    if den == 0:
        return 0.0
    return max(-1.0, min(1.0, (num / den).item()))


def l2_norm(t: torch.Tensor) -> float:
    return torch.linalg.norm(t.flatten()).item()


def normalized_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized L2 difference: ‖a - b‖ / (‖a‖ + ‖b‖).  0 = identical, 1+ = very different."""
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    diff_norm = torch.linalg.norm(a_f - b_f)
    sum_norm = torch.linalg.norm(a_f) + torch.linalg.norm(b_f)
    if sum_norm == 0:
        return 0.0
    return (diff_norm / sum_norm).item()


def format_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def group_key(key: str) -> str:
    """Group a LoRA key into a block category for summary."""
    if "double_blocks." in key:
        parts = key.split("double_blocks.")[1]
        bnum = parts.split(".")[0]
        return f"double_blocks.{bnum}"
    if "single_blocks." in key:
        parts = key.split("single_blocks.")[1]
        bnum = parts.split(".")[0]
        return f"single_blocks.{bnum}"
    # Transformer / SDXL style
    if "lora_unet_" in key:
        # e.g. lora_unet_input_blocks_1_1_proj_in.lora_down.weight
        parts = key.split("lora_unet_")[1]
        seg = parts.split(".")[0]  # input_blocks_1_1_proj_in
        return f"unet.{seg}"
    if "lora_te_" in key:
        parts = key.split("lora_te_")[1]
        seg = parts.split(".")[0]
        seg = seg.rsplit("_", 1)[0] if seg.endswith("_lora_down") or seg.endswith("_lora_up") else seg
        return f"te.{seg}"
    if "lora." in key:
        # Kohya-style: something like lora.unet.input_blocks.1.1.proj_in.lora_down.weight
        parts = key.split("lora.")[1]
        seg = parts.rsplit(".lora_down", 1)[0] if ".lora_down" in parts else parts.rsplit(".lora_up", 1)[0]
        seg = seg.rsplit(".", 1)[0] if seg.endswith("weight") else seg
        return f"kohya.{seg}"
    return "other"


def short_key(key: str) -> str:
    """Shorten key for display."""
    # Remove lora_down / lora_up / weight / bias suffixes for cleaner group display
    s = key
    for suffix in [".lora_down.weight", ".lora_up.weight", ".weight", ".bias",
                   "_lora_down.weight", "_lora_up.weight"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    # Collapse path separators
    return s


# ── Argument Parsing ──

def parse_args():
    parser = argparse.ArgumentParser(
        prog="lora_similarity.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Calculate cosine similarity between two LoRA safetensors files.\n"
            "Only common keys (tensors with the same name and shape) are compared."
        ),
        epilog=(
            "EXAMPLES:\n"
            "  Basic similarity:\n"
            "    python lora_similarity.py --lora_a style1.safetensors --lora_b style2.safetensors\n\n"
            "  Output as CSV:\n"
            "    python lora_similarity.py --lora_a a.safetensors --lora_b b.safetensors --csv\n\n"
            "  Show top 20 most similar/dissimilar layers:\n"
            "    python lora_similarity.py --lora_a a.safetensors --lora_b b.safetensors --top 20\n\n"
            "  Save results to JSON:\n"
            "    python lora_similarity.py --lora_a a.safetensors --lora_b b.safetensors --json-out results.json\n"
        )
    )

    parser.add_argument("--lora_a", required=True, help="First LoRA safetensors file")
    parser.add_argument("--lora_b", required=True, help="Second LoRA safetensors file")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top similar/dissimilar layers to show (default: 10)")
    parser.add_argument("--csv", action="store_true",
                        help="Output per-layer results as CSV instead of table")
    parser.add_argument("--json-out", default=None,
                        help="Write full results as JSON to this file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Minimum cosine similarity to report a layer as 'similar' (optional)")
    parser.add_argument("--decimals", type=int, default=5,
                        help="Decimal places for numeric output (default: 5)")

    args = parser.parse_args()

    for f in [args.lora_a, args.lora_b]:
        if not os.path.isfile(f):
            parser.error(f"File not found: {f}")

    return args


# ── Main ──

def main():
    args = parse_args()
    D = args.decimals
    name_a = strip_ext(args.lora_a)
    name_b = strip_ext(args.lora_b)

    print(f"\n{'='*70}")
    print(f"  LoRA Similarity Analysis")
    print(f"{'='*70}")
    print(f"  A: {args.lora_a}")
    print(f"  B: {args.lora_b}")
    print()

    # ── Load ──
    print("  Loading LoRAs...")
    lora_a = load_file(args.lora_a, device="cpu")
    lora_b = load_file(args.lora_b, device="cpu")

    size_a = os.path.getsize(args.lora_a)
    size_b = os.path.getsize(args.lora_b)
    print(f"  LoRA A size: {format_size(size_a)} ({len(lora_a)} tensors)")
    print(f"  LoRA B size: {format_size(size_b)} ({len(lora_b)} tensors)")
    print()

    # ── Find common keys by both name AND shape ──
    common_keys = []
    skipped_shape = []
    only_a = []
    only_b = []

    for key in lora_a:
        if key in lora_b:
            if lora_a[key].shape == lora_b[key].shape:
                common_keys.append(key)
            else:
                skipped_shape.append((key, lora_a[key].shape, lora_b[key].shape))
        else:
            only_a.append(key)
    for key in lora_b:
        if key not in lora_a:
            only_b.append(key)

    if not common_keys:
        print("  ❌ No common keys with matching shapes found between the two LoRAs.")
        print("     They may target different model architectures.")
        sys.exit(1)

    print(f"  Common keys (same name + shape): {len(common_keys)}")
    if skipped_shape:
        print(f"  Skipped (same name, different shape): {len(skipped_shape)}")
    if only_a:
        print(f"  Only in A: {len(only_a)}")
    if only_b:
        print(f"  Only in B: {len(only_b)}")
    print()

    # ── Compute per-layer similarity ──
    results = []
    for key in common_keys:
        ta = lora_a[key]
        tb = lora_b[key]
        cosim = cosine_similarity(ta, tb)
        ndiff = normalized_diff(ta, tb)
        elems = ta.numel()
        results.append({
            "key": key,
            "shape": list(ta.shape),
            "elements": elems,
            "cosine_similarity": cosim,
            "normalized_diff": ndiff,
            "norm_a": l2_norm(ta),
            "norm_b": l2_norm(tb),
            "group": group_key(key),
            "short": short_key(key),
        })

    # Sort by cosine similarity ascending → most different first
    results.sort(key=lambda r: r["cosine_similarity"])

    # ── Aggregate ──
    total_elements = sum(r["elements"] for r in results)
    weighted_cosim = sum(r["cosine_similarity"] * r["elements"] for r in results) / total_elements if total_elements else 0.0
    mean_cosim = sum(r["cosine_similarity"] for r in results) / len(results)
    min_cosim = results[0]["cosine_similarity"]
    max_cosim = results[-1]["cosine_similarity"]

    # Per-group stats
    groups = defaultdict(list)
    for r in results:
        groups[r["group"]].append(r)

    # ── Output ──

    # Summary
    print(f"  {'─'*66}")
    print(f"  SUMMARY")
    print(f"  {'─'*66}")
    print(f"  Overall weighted cosine similarity   : {weighted_cosim:.{D}f}")
    print(f"  Mean cosine similarity               : {mean_cosim:.{D}f}")
    print(f"  Min cosine similarity                : {min_cosim:.{D}f}  ({results[0]['short']})")
    print(f"  Max cosine similarity                : {max_cosim:.{D}f}  ({results[-1]['short']})")
    print(f"  Compared tensors                     : {len(results)}")
    print(f"  Total elements compared               : {total_elements:,}")
    print()

    # Histogram-like distribution
    buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    print(f"  SIMILARITY DISTRIBUTION")
    print(f"  {'─'*66}")
    for lo, hi in buckets:
        count = sum(1 for r in results if lo <= r["cosine_similarity"] < hi)
        bar = "█" * count if count <= 50 else "█" * 50 + f" {count}"
        if count > 0:
            print(f"  [{lo:.1f}–{hi:.1f})  {bar}")
    print()

    # Per-block group averages
    print(f"  PER-BLOCK AVERAGES (sorted by similarity)")
    print(f"  {'─'*66}")
    block_stats = []
    for gname, grp in groups.items():
        g_wcosim = sum(r["cosine_similarity"] * r["elements"] for r in grp) / sum(r["elements"] for r in grp)
        g_mean = sum(r["cosine_similarity"] for r in grp) / len(grp)
        block_stats.append((g_wcosim, g_mean, gname, len(grp)))
    block_stats.sort(key=lambda x: x[0])  # sort by weighted cosim ascending
    for wcosim, mean, gname, count in block_stats:
        bar_len = int(wcosim * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {gname:<40s} {bar} {wcosim:.{D}f}  ({count} tensors)")
    print()

    # ── Top most dissimilar / similar ──
    n = min(args.top, len(results))

    print(f"  TOP {n} MOST DISSIMILAR (lowest cosine similarity)")
    print(f"  {'─'*66}")
    for i in range(n):
        r = results[i]
        print(f"  {i+1:>3}.  {r['short']:<55s}  cos={r['cosine_similarity']:.{D}f}  "
              f"diff={r['normalized_diff']:.{D}f}  "
              f"‖A‖={r['norm_a']:.{D}f}  ‖B‖={r['norm_b']:.{D}f}")
    print()

    print(f"  TOP {n} MOST SIMILAR (highest cosine similarity)")
    print(f"  {'─'*66}")
    for i in range(n):
        r = results[-(i + 1)]
        print(f"  {i+1:>3}.  {r['short']:<55s}  cos={r['cosine_similarity']:.{D}f}  "
              f"diff={r['normalized_diff']:.{D}f}  "
              f"‖A‖={r['norm_a']:.{D}f}  ‖B‖={r['norm_b']:.{D}f}")
    print()

    # ── Threshold filter ──
    if args.threshold is not None:
        similar = [r for r in results if r["cosine_similarity"] >= args.threshold]
        print(f"  COSINE SIMILARITY ≥ {args.threshold}: {len(similar)}/{len(results)} layers")
        if similar:
            for r in similar:
                print(f"    {r['short']:<55s}  cos={r['cosine_similarity']:.{D}f}")
        print()

    # ── CSV output ──
    if args.csv:
        csv_file = f"{name_a}_vs_{name_b}_similarity.csv"
        with open(csv_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "shape", "elements", "cosine_similarity",
                        "normalized_diff", "norm_a", "norm_b", "group"])
            for r in results:
                w.writerow([
                    r["key"],
                    "×".join(str(s) for s in r["shape"]),
                    r["elements"],
                    f"{r['cosine_similarity']:.{D}f}",
                    f"{r['normalized_diff']:.{D}f}",
                    f"{r['norm_a']:.{D}f}",
                    f"{r['norm_b']:.{D}f}",
                    r["group"],
                ])
        print(f"  📄 CSV written to: {csv_file}")
        print()

    # ── JSON output ──
    if args.json_out:
        out = {
            "lora_a": args.lora_a,
            "lora_b": args.lora_b,
            "lora_a_tensors": len(lora_a),
            "lora_b_tensors": len(lora_b),
            "common_keys": len(common_keys),
            "skipped_shape": len(skipped_shape),
            "only_in_a": len(only_a),
            "only_in_b": len(only_b),
            "overall_weighted_cosine_similarity": round(weighted_cosim, D),
            "mean_cosine_similarity": round(mean_cosim, D),
            "min_cosine_similarity": round(min_cosim, D),
            "max_cosine_similarity": round(max_cosim, D),
            "per_layer": [
                {
                    "key": r["key"],
                    "shape": r["shape"],
                    "elements": r["elements"],
                    "cosine_similarity": round(r["cosine_similarity"], D),
                    "normalized_diff": round(r["normalized_diff"], D),
                    "norm_a": round(r["norm_a"], D),
                    "norm_b": round(r["norm_b"], D),
                    "group": r["group"],
                }
                for r in results
            ],
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  📄 JSON written to: {args.json_out}")
        print()

    # ── Skipped shape mismatches ──
    if skipped_shape:
        print(f"  SKIPPED (shape mismatch): {len(skipped_shape)}")
        for key, sha, shb in skipped_shape[:10]:  # show first 10
            print(f"    {key:<55s}  A={list(sha)}  B={list(shb)}")
        if len(skipped_shape) > 10:
            print(f"    ... and {len(skipped_shape) - 10} more")
        print()

    # ── Only-in-A / Only-in-B ──
    for label, keys_list in [("A", only_a), ("B", only_b)]:
        if keys_list:
            print(f"  ONLY IN {label}: {len(keys_list)}")
            for k in keys_list[:6]:
                print(f"    {k}")
            if len(keys_list) > 6:
                print(f"    ... and {len(keys_list) - 6} more")
            print()

    print(f"  {'='*70}")
    print(f"  Done.")
    print(f"  {'='*70}\n")


if __name__ == "__main__":
    main()
