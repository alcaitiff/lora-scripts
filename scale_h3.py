#!/usr/bin/env python3
"""
Scale the effective strength of a MiniMax H3 LoRA.

IMPORTANT: LoRA weight change is delta_W = B @ A (times the scale applied at
load time). To multiply the *effective* strength by `factor`, only ONE factor
needs scaling:

    B' = factor * B   (or A' = factor * A)   ->  delta_W' = factor * delta_W

Scaling BOTH A and B by `factor` would give factor^2, which is almost never
what you want. The krea2-era scale script did that; this one does it right by
default and offers the legacy behavior via --both.

Preserves ai-toolkit metadata (ss_base_model_version = minimax_h3).

Usage:
  ./scale_h3.py 2.0 lora.safetensors                 # 2x effective strength
  ./scale_h3.py 0.5 lora.safetensors --out halved.safetensors
  ./scale_h3.py 10.0 --dry-run lora.safetensors
  ./scale_h3.py 2.0 --both lora.safetensors          # legacy factor^2 scaling
  ./scale_h3.py 3.0 *.safetensors
"""

from safetensors.torch import load_file, save_file
from safetensors import safe_open
import argparse
import glob
import os
import sys


def load_metadata(path):
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


parser = argparse.ArgumentParser(
    description="Scale effective strength of MiniMax H3 LoRA files by a factor",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "Scaling math: delta_W = B @ A. Default scales B only (or A only with\n"
        "--side A), giving exactly `factor` x effective strength. Use --both\n"
        "for legacy factor^2 behavior.\n\n"
        "Examples:\n"
        "  ./scale_h3.py 2.0 lora.safetensors\n"
        "  ./scale_h3.py 0.5 lora.safetensors --out halved.safetensors\n"
        "  ./scale_h3.py 10.0 --dry-run lora.safetensors\n"
        "  ./scale_h3.py 3.0 *.safetensors\n"
    ),
)
parser.add_argument("factor", type=float, help="Scaling factor (e.g. 3.0 for 3x strength)")
parser.add_argument("inputs", nargs="+", help="Input .safetensors files or glob patterns")
parser.add_argument("--out", default=None, help="Output file (single input only)")
parser.add_argument("--side", choices=["B", "A"], default="B",
                    help="Which factor to scale (default: B)")
parser.add_argument("--both", action="store_true",
                    help="Legacy: scale both A and B by factor (effective factor^2)")
parser.add_argument("--dry-run", action="store_true",
                    help="Show what would be done without writing output files")
args = parser.parse_args()

if args.factor <= 0:
    print("Factor must be positive.")
    sys.exit(1)

# Resolve input files
input_files = []
for pattern in args.inputs:
    matches = glob.glob(pattern)
    if not matches:
        print(f"Warning: no files matched '{pattern}'")
    input_files.extend(matches)

if not input_files:
    print("No input files to process.")
    sys.exit(1)

# Deduplicate, preserve order
seen = set()
input_files = [f for f in input_files if not (f in seen or seen.add(f))]

output_override = None
if args.out:
    if len(input_files) != 1:
        print("--out can only be used when exactly one input file is provided.")
        sys.exit(1)
    output_override = args.out

if args.both:
    mode_desc = f"BOTH A and B x{args.factor} (effective {args.factor**2:.4g}^2)"
else:
    mode_desc = f"{args.side} x{args.factor} (effective x{args.factor})"

print(f"Scale mode: {mode_desc}\n")

for input_path in input_files:
    base, ext = os.path.splitext(input_path)
    if output_override:
        output_path = output_override
    else:
        factor_str = str(args.factor).rstrip("0").rstrip(".")
        suffix = "both" if args.both else ("a" if args.side == "A" else "b")
        output_path = f"{base}_x{factor_str}{suffix}{ext}"

    print(f"Loading: {input_path}")
    state = load_file(input_path)
    keys = list(state.keys())

    fam, why = looks_like_h3(keys)
    if fam != "h3":
        print(f"⚠ Input looks like: {fam} ({why})")
        print("  Expected MiniMax H3 (diffusion_model.blocks.N.qkv_proj/fc1/fc2).")
        print("  Continuing anyway — verify layer names before use in ComfyUI.")

    n_tensors = len(state)
    n_elements = sum(t.numel() for t in state.values())
    n_lora = sum(1 for k in keys if "lora_A.weight" in k or "lora_B.weight" in k)

    # Stats on lora_B only (the side we scale by default) before
    b_mean_before = sum(t.abs().mean().item() for k, t in state.items()
                        if k.endswith("lora_B.weight")) / max(n_lora // 2, 1)

    if args.dry_run:
        print(f"Would scale {n_tensors} tensors ({n_elements:,} elements, {n_lora} LoRA factors) by {args.factor}x")
        print(f"  mode: {mode_desc}")
        eff = args.factor if not args.both else args.factor ** 2
        print(f"  lora_B abs mean: {b_mean_before:.6f} -> {b_mean_before * eff:.6f} (effective x{eff:.4g})")
        print(f"  output:   {output_path}")
        continue

    # Scale
    scaled = {}
    for k, t in state.items():
        if args.both:
            if "lora_A.weight" in k or "lora_B.weight" in k:
                t = t * args.factor
        else:
            if k.endswith(f"lora_{args.side}.weight"):
                t = t * args.factor
        scaled[k] = t

    # Preserve metadata
    meta = load_metadata(input_path)
    meta["ss_output_name"] = os.path.splitext(os.path.basename(output_path))[0]
    meta["name"] = os.path.splitext(os.path.basename(output_path))[0]
    if "ss_base_model_version" not in meta:
        meta["ss_base_model_version"] = "minimax_h3"
    meta["ss_scale_factor"] = str(args.factor)
    meta["ss_scale_mode"] = mode_desc

    print(f"Saving: {output_path}")
    save_file(scaled, output_path, metadata=meta)

    b_mean_after = sum(t.abs().mean().item() for k, t in scaled.items()
                       if k.endswith("lora_B.weight")) / max(n_lora // 2, 1)
    print(f"  tensors:      {n_tensors}")
    print(f"  elements:     {n_elements:,}")
    print(f"  LoRA factors: {n_lora}")
    print(f"  lora_B abs mean: {b_mean_before:.6f} -> {b_mean_after:.6f}")
    print(f"  dtype:        {next(iter(state.values())).dtype}")
    print()
