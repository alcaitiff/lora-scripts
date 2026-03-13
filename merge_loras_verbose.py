import argparse
import os
import sys
import math
import torch
from safetensors.torch import load_file, save_file


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def strip_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def default_output_name(lora_a, alpha_a, lora_b, alpha_b):
    return f"{strip_ext(lora_a)}_{alpha_a}_{strip_ext(lora_b)}_{alpha_b}.safetensors"


def l2(t):
    return torch.linalg.norm(t).item()


# -------------------------------------------------
# Argument parsing
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        prog="merge_loras_verbose.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Model-agnostic LoRA merger with very verbose, per-layer diagnostics.\n"
            "Supports rank mismatches using LoRA-correct projection math.\n"
        ),
        epilog=(
            "USAGE:\n"
            "  python merge_loras_verbose.py --lora_a A.safetensors "
            "--lora_b B.safetensors [options]\n\n"
            "EXAMPLES:\n"
            "  Basic merge with defaults:\n"
            "    python merge_loras_verbose.py \\\n"
            "      --lora_a base.safetensors \\\n"
            "      --lora_b style.safetensors\n\n"
            "  Explicit weights and output name:\n"
            "    python merge_loras_verbose.py \\\n"
            "      --lora_a l1.safetensors \\\n"
            "      --alpha_a 0.8 \\\n"
            "      --lora_b l2.safetensors \\\n"
            "      --alpha_b 0.2 \\\n"
            "      --out merged.safetensors\n\n"
            "DEFAULT OUTPUT NAMING:\n"
            "  If --out is omitted, the output file is automatically named as:\n"
            "    <loraA>_<alphaA>_<loraB>_<alphaB>.safetensors\n\n"
            "RANK MISMATCH HANDLING:\n"
            "  • Rank mismatches are merged using LoRA projection math.\n"
            "  • Smaller-rank tensors are scaled by sqrt(R_big / R_small).\n"
            "  • Larger-rank capacity is preserved.\n\n"
            "NOTES:\n"
            "  • Both LoRAs must target the same base architecture.\n"
            "  • Very different file sizes are normal and fully supported.\n"
            "  • No layers are silently dropped."
        )
    )

    parser.add_argument("--lora_a", required=True, help="First LoRA safetensors file")
    parser.add_argument("--lora_b", required=True, help="Second LoRA safetensors file")
    parser.add_argument("--alpha_a", type=float, default=0.7, help="Weight for LoRA A")
    parser.add_argument("--alpha_b", type=float, default=0.3, help="Weight for LoRA B")
    parser.add_argument("--out", default=None, help="Output safetensors file")

    args = parser.parse_args()

    if not os.path.isfile(args.lora_a):
        parser.error(f"File not found: {args.lora_a}")
    if not os.path.isfile(args.lora_b):
        parser.error(f"File not found: {args.lora_b}")

    if not (0.0 <= args.alpha_a <= 1.0):
        parser.error("--alpha_a must be between 0.0 and 1.0")
    if not (0.0 <= args.alpha_b <= 1.0):
        parser.error("--alpha_b must be between 0.0 and 1.0")
    if args.alpha_a + args.alpha_b == 0.0:
        parser.error("alpha_a + alpha_b must be > 0.0")

    if args.out is None:
        args.out = default_output_name(
            args.lora_a, args.alpha_a,
            args.lora_b, args.alpha_b
        )

    return args


# -------------------------------------------------
# Rank-aware merge (ORIGINAL CORRECT MATH)
# -------------------------------------------------

def merge_rank_mismatch_original_math(ta, tb, alpha_a, alpha_b):
    """
    EXACT original math:
        merged = ta
        merged += tb * alpha_b * rank_scale
        merged *= alpha_a

    Handles both lora_A and lora_B layouts.
    """

    norm_a = l2(ta)
    norm_b = l2(tb)

    merged = ta.clone()

    # lora_A case: rank on dim 0
    if ta.shape[1] == tb.shape[1]:
        R_big = ta.shape[0]
        R_small = tb.shape[0]
        rank_scale = math.sqrt(R_big / R_small)

        merged[:R_small, :] += tb * alpha_b * rank_scale
        merged *= alpha_a

    # lora_B case: rank on dim 1
    elif ta.shape[0] == tb.shape[0]:
        R_big = ta.shape[1]
        R_small = tb.shape[1]
        rank_scale = math.sqrt(R_big / R_small)

        merged[:, :R_small] += tb * alpha_b * rank_scale
        merged *= alpha_a

    else:
        raise RuntimeError("Incompatible rank layout")

    merged_norm = l2(merged)

    # Observational diagnostics
    contrib_a_norm = norm_a * alpha_a
    contrib_b_norm = norm_b * alpha_b * rank_scale
    total = contrib_a_norm + contrib_b_norm + 1e-8

    debug = {
        "norm_a": norm_a,
        "norm_b": norm_b,
        "contrib_a_norm": contrib_a_norm,
        "contrib_b_norm": contrib_b_norm,
        "merged_norm": merged_norm,
        "pct_a": 100.0 * contrib_a_norm / total,
        "pct_b": 100.0 * contrib_b_norm / total,
        "rank_scale": rank_scale,
    }

    return merged, debug


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    args = parse_args()

    print("\n🔹 Loading LoRAs")
    print(f"  A: {args.lora_a}")
    print(f"  B: {args.lora_b}")
    print(f"  Output: {args.out}\n")

    lora_a = load_file(args.lora_a)
    lora_b = load_file(args.lora_b)

    keys = sorted(set(lora_a) | set(lora_b))
    merged = {}

    print("🔹 Beginning layer-by-layer merge\n")

    for key in keys:
        print(f"▶ Layer: {key}")

        if key in lora_a and key in lora_b:
            ta = lora_a[key]
            tb = lora_b[key]
            print("  • Present in BOTH")

            # Same shape: trivial merge
            if ta.shape == tb.shape:
                na = l2(ta)
                nb = l2(tb)

                out = ta * args.alpha_a + tb * args.alpha_b

                ca = na * args.alpha_a
                cb = nb * args.alpha_b
                nm = l2(out)
                total = ca + cb + 1e-8

                print("  • Same shape merge")
                print(f"      ‖A‖            : {na:.6f}")
                print(f"      ‖B‖            : {nb:.6f}")
                print(f"      ‖A×αₐ‖         : {ca:.6f}")
                print(f"      ‖B×αᵦ‖         : {cb:.6f}")
                print(f"      ‖Merged‖       : {nm:.6f}")
                print(
                    f"      Contribution   : "
                    f"A {100*ca/total:.2f}% | B {100*cb/total:.2f}%"
                )

                merged[key] = out
                print("  ✅ MERGED\n")
                continue

            # Rank mismatch — ensure ta is larger tensor
            if ta.numel() < tb.numel():
                ta, tb = tb, ta
                alpha_a, alpha_b = args.alpha_b, args.alpha_a
            else:
                alpha_a, alpha_b = args.alpha_a, args.alpha_b

            try:
                out, dbg = merge_rank_mismatch_original_math(
                    ta, tb, alpha_a, alpha_b
                )
                merged[key] = out

                print("  ⚠ Rank mismatch → original LoRA projection")
                print(f"      ‖A‖            : {dbg['norm_a']:.6f}")
                print(f"      ‖B‖            : {dbg['norm_b']:.6f}")
                print(f"      ‖A×αₐ‖         : {dbg['contrib_a_norm']:.6f}")
                print(f"      ‖B×αᵦ×scale‖   : {dbg['contrib_b_norm']:.6f}")
                print(f"      ‖Merged‖       : {dbg['merged_norm']:.6f}")
                print(f"      Rank scale     : {dbg['rank_scale']:.6f}")
                print(
                    f"      Contribution   : "
                    f"A {dbg['pct_a']:.2f}% | B {dbg['pct_b']:.2f}%"
                )
                print("  ✅ MERGED (rank-aware)\n")

            except RuntimeError:
                print("  ❌ Incompatible layout → copied larger tensor\n")
                merged[key] = ta.clone()

        elif key in lora_a:
            merged[key] = lora_a[key].clone()
            print("  • Only in A → copied\n")

        else:
            merged[key] = lora_b[key].clone()
            print("  • Only in B → copied\n")

    save_file(merged, args.out)

    print("================================")
    print("✅ Merge completed successfully")
    print(f"📦 Output: {args.out}")
    print("================================")


if __name__ == "__main__":
    main()
