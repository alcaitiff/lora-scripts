#!/usr/bin/env python3
"""
Convert a MiniMax H3 LoRA to ComfyUI key naming (diffusers2 format).

ComfyUI's LoRA adapter (comfy/weight_adapter/lora.py) matches the diffusers2
pattern:

    {model_key}.lora_A.weight     (down / in side)
    {model_key}.lora_B.weight     (up / out side)

where {model_key} is the full model key as ComfyUI holds it (wrapped as
self.diffusion_model), e.g.

    diffusion_model.blocks.9.mlp.fc2.weight

This script handles four input styles and normalizes them all:

1. Official MiniMax checkpoints (minimax_h3_turbo_4step / _8step) ship WITHOUT
   the "diffusion_model." prefix: keys look like
   blocks.0.adaln_proj.linear.lora_A.weight.  → prefix added.

2. musubi-tuner checkpoints ship in kohya format:
       lora_unet_blocks_9_mlp_fc2.lora_down.weight   [rank, in]
       lora_unet_blocks_9_mlp_fc2.lora_up.weight     [out, rank]
       lora_unet_blocks_9_mlp_fc2.alpha              (scalar)
   → rewritten to
       diffusion_model.blocks.9.mlp.fc2.lora_A.weight
       diffusion_model.blocks.9.mlp.fc2.lora_B.weight
       diffusion_model.blocks.9.mlp.fc2.alpha
   The underscore path becomes dotted, lora_down→lora_A, lora_up→lora_B.
   No transposition is needed: musubi-tuner already stores down as
   [rank, in] and up as [out, rank], exactly what ComfyUI expects (up @ down).

3. Official FL2VA turbo (minimax_h3_fl2v_turbo_4step_v0.1) ships in diffusers
   PEFT format with UNFUSED attention:
       transformer_blocks.9.attn.to_q.lora_A.default.weight
       token_refiner.refiner_blocks.1.ff.net.2.lora_B.default.weight
   → renamed to ComfyUI paths:
       transformer_blocks.N      -> blocks.N
       token_refiner.refiner_blocks.N -> token_refiner.blocks.N
       attn.to_out.0             -> attn.out_proj
       ff.net.0.proj             -> mlp.fc1
       ff.net.2                  -> mlp.fc2
   The attention is the tricky one: ComfyUI's H3 uses a FUSED qkv_proj
   ([3*inner, hidden]) while this LoRA stores to_q/to_k/to_v separately.
   The three A tensors are concatenated (rank 3x) and the three B tensors
   block-diagonalized, which reproduces vstack([dq, dk, dv]) EXACTLY (no SVD,
   no loss). Order is q, k, v, matching ComfyUI's qkv_proj split.

4. Already-converted keys (diffusion_model. prefix) are passed through
   untouched.

Alpha tensors are kept and renamed to {model_key}.alpha — ComfyUI reads them
in comfy/lora.py load_lora() and applies the kohya alpha/rank scaling. If a
LoRA was trained with alpha == rank (musubi-tuner default, e.g. 32/32) the
scale is 1.0 either way, but keeping them stays faithful when alpha != rank.

Metadata is preserved; tensors are moved by reference, never copied or
re-typed (use --bf16 to cast to bfloat16 for consistency with the rest of
the h3 folder).

Usage:
  ./convert_h3_lora_to_comfy.py MysticXXX_MMH3-step00001700.safetensors
  ./convert_h3_lora_to_comfy.py minimax_h3_turbo_4step.safetensors --out comfy.safetensors
  ./convert_h3_lora_to_comfy.py --in-place MysticXXX_MMH3-step00001700.safetensors
  ./convert_h3_lora_to_comfy.py --dry-run MysticXXX_MMH3-step00001700.safetensors
"""

from safetensors.torch import load_file, save_file
from safetensors import safe_open
import argparse
import glob
import os
import re
import sys

import torch

PREFIX = "diffusion_model."
MUSUBI_PREFIX = "lora_unet_"

# Official FL2VA turbo LoRA (zoe-diffusion / PEFT format):
#   transformer_blocks.N.{layer}.lora_[AB].default.weight
#   token_refiner.refiner_blocks.N.{layer}.lora_[AB].default.weight
# Groups: 1 = transformer block idx, 2 = token_refiner block idx,
#         3 = layer, 4 = A/B side.
DIFFUSERS_LORA_RE = re.compile(
    r"^(?:transformer_blocks\.(\d+)|token_refiner\.refiner_blocks\.(\d+))"
    r"\.(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|ff\.net\.0\.proj|ff\.net\.2)"
    r"\.lora_([AB])\.default\.weight$"
)

# diffusers PEFT layer -> ComfyUI H3 layer (1:1 renames; to_q/to_k/to_v are
# handled separately as a fused qkv_proj merge).
DIFFUSERS_LAYER_MAP = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
}

# (source suffix, target suffix) — checked in order
SUFFIX_MAP = [
    (".lora_down.weight", ".lora_A.weight"),
    (".lora_up.weight", ".lora_B.weight"),
    (".alpha", ".alpha"),
]

# H3 module names that contain underscores in the model's state-dict keys
# (out_proj, qkv_proj, token_refiner, ...). kohya-style naming — which is what
# musubi-tuner emits — flattens every dot of the path to an underscore, so a
# generic _ -> . split would wrongly turn blocks.N.attn.out_proj into
# blocks.N.attn.out.proj. Protect these compounds first, split, then restore.
H3_UNDERSCORE_MODULES = [
    "token_refiner",
    "audio_patch_proj",
    "condition_proj",
    "adaln_proj",
    "sigma_shift_audio",
    "sigma_shift_video",
    "final_norm",
    "final_layer",
    "out_proj",
    "qkv_proj",
    "proj_in",
    "proj_out",
    "q_norm",
    "k_norm",
    "img_pos",
    "img_update",
    "audio_pos",
    "audio_update",
    "audio_out",
    "freq_dim",
]


def musubi_layer_to_model(layer):
    """Reverse kohya's underscore flattening: lora_unet_<layer> -> dotted model path."""
    for name in H3_UNDERSCORE_MODULES:
        layer = layer.replace(name, name.replace("_", "\x00"))
    layer = layer.replace("_", ".")
    return layer.replace("\x00", "_")


def load_metadata(path):
    with safe_open(path, framework="pt") as f:
        return dict(f.metadata()) if f.metadata() else {}


def looks_like_h3(keys):
    """Return ('h3'|'krea2'|'unknown', reason) from key list."""
    lora_markers = ["lora_A.weight", "lora_B.weight", "lora_down.weight",
                    "lora_up.weight", "lora_A.default.weight", "lora_B.default.weight"]
    lora_tensors = [k for k in keys if any(m in k for m in lora_markers)]
    if not lora_tensors:
        return "unknown", "no lora tensors found"
    joined = "|" + "|".join(keys)
    h3_hits = sum(1 for m in ["qkv_proj", "token_refiner", "fc1", "fc2",
                              "refiner_blocks", "ff.net", "to_out.0"] if m in joined)
    krea2_hits = sum(1 for m in [".gate.", ".wk.", ".wv.", ".wo."] if m in joined)
    if h3_hits >= 2 and krea2_hits == 0:
        return "h3", "qkv_proj/fc1/fc2/token_refiner/refiner_blocks layers detected"
    if krea2_hits >= 2 and h3_hits == 0:
        return "krea2", "gate/wk/wv/wo layers detected"
    if h3_hits > krea2_hits:
        return "h3", f"mixed markers (h3:{h3_hits}, krea2:{krea2_hits})"
    if krea2_hits > 0:
        return "krea2", f"mixed markers (h3:{h3_hits}, krea2:{krea2_hits})"
    return "unknown", "no recognizable layer markers"


def convert_key(key):
    """Map a raw H3 LoRA key to ComfyUI diffusers2 naming.

    Returns (new_key, group). group is None for 1:1 keys, or a dict for
    diffusers to_q/to_k/to_v keys that must be merged into a fused qkv_proj:
    {'target': 'diffusion_model.blocks.N.attn.qkv_proj',
     'component': 'to_q', 'side': 'A'}.
    Returns (None, None) for unrecognized keys (should be skipped).
    """
    if key.startswith(PREFIX):
        return key, None  # already ComfyUI-ready
    if key.startswith(MUSUBI_PREFIX):
        # musubi-tuner kohya format: lora_unet_<layer>.<suffix>
        body = key[len(MUSUBI_PREFIX):]
        for src, dst in SUFFIX_MAP:
            if body.endswith(src):
                layer = body[:-len(src)]
                return PREFIX + musubi_layer_to_model(layer) + dst, None
        return None, None  # lora_unet_ key with an unknown suffix
    m = DIFFUSERS_LORA_RE.match(key)
    if m:
        main_idx, tr_idx, layer, side = m.groups()
        if tr_idx is not None:
            base = f"diffusion_model.token_refiner.blocks.{tr_idx}"
        else:
            base = f"diffusion_model.blocks.{main_idx}"
        if layer in DIFFUSERS_LAYER_MAP:
            return f"{base}.{DIFFUSERS_LAYER_MAP[layer]}.lora_{side}.weight", None
        # attn.to_q / to_k / to_v -> fused qkv_proj (merged later)
        comp = layer.rsplit(".", 1)[1]  # to_q / to_k / to_v
        return f"{base}.attn.qkv_proj.lora_{side}.weight", {
            "target": f"{base}.attn.qkv_proj",
            "component": comp,
            "side": side,
        }
    # bare official key (blocks.N.xxx.lora_A.weight)
    return PREFIX + key, None


def merge_qkv_groups(qkv_groups):
    """Merge diffusers to_q/to_k/to_v lora pairs into fused qkv_proj loras.

    Exact merge, no SVD: A_fused = cat([A_q, A_k, A_v]) (rank 3x), B_fused =
    block_diag(B_q, B_k, B_v), so B_fused @ A_fused == vstack([dq, dk, dv]),
    which is precisely the delta ComfyUI's fused qkv_proj expects (q first,
    then k, then v — matching qkv_proj(x).split in the model).

    qkv_groups: target key -> {'to_qA': t, 'to_kA': t, 'to_vA': t, ...}
    Returns (converted_tensors, merged_group_count).
    """
    merged = {}
    for target in sorted(qkv_groups):
        parts = qkv_groups[target]
        comps = []
        missing = []
        for comp in ("to_q", "to_k", "to_v"):
            a = parts.get(comp + "A")
            b = parts.get(comp + "B")
            if a is not None and b is not None:
                comps.append((comp, a, b))
            else:
                missing.append(comp)
        if missing:
            print(f"  ⚠ qkv group incomplete for {target}: missing {missing}")
        if not comps:
            continue
        a_fused = torch.cat([c[1] for c in comps], dim=0)
        b_fused = torch.block_diag(*[c[2] for c in comps])
        merged[target + ".lora_A.weight"] = a_fused
        merged[target + ".lora_B.weight"] = b_fused
    return merged, len(qkv_groups)


def main():


    parser = argparse.ArgumentParser(
        description="Convert MiniMax H3 LoRA keys to ComfyUI diffusers2 naming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Accepts 4 input styles and normalizes all to diffusion_model.* lora_A/B:\n"
            "  1. official turbo keys  (bare blocks.N.xxx.lora_A/lora_B)      -> prefix\n"
            "  2. musubi-tuner keys    (lora_unet_*.lora_down/up + .alpha)    -> rename\n"
            "  3. FL2VA turbo PEFT keys (transformer_blocks.N.*.lora_[AB].default.weight)\n"
            "                            -> rename; attn to_q/to_k/to_v merged into\n"
            "                               fused qkv_proj (cat A's, block_diag B's, exact)\n"
            "  4. already-converted keys (diffusion_model. prefix)            -> unchanged\n\n"
            "Examples:\n"
            "  ./convert_h3_lora_to_comfy.py MysticXXX_MMH3-step00001700.safetensors\n"
            "  ./convert_h3_lora_to_comfy.py minimax_h3_fl2v_turbo_4step_v0.1.safetensors\n"
            "  ./convert_h3_lora_to_comfy.py --in-place MysticXXX_MMH3-step00001700.safetensors\n"
            "  ./convert_h3_lora_to_comfy.py --dry-run --bf16 minimax_h3_fl2v_turbo_4step_v0.1.safetensors\n"
        ),
    )
    parser.add_argument("inputs", nargs="+", help="Input .safetensors files or glob patterns")
    parser.add_argument("--out", default=None, help="Output file (single input only)")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite the input file with the converted version")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing output files")
    parser.add_argument("--bf16", action="store_true",
                        help="Cast all tensors to bfloat16 (matches the bf16 files in the h3 folder)")
    args = parser.parse_args()

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

    if args.in_place and args.out:
        print("--in-place and --out are mutually exclusive.")
        sys.exit(1)

    for input_path in input_files:
        base, ext = os.path.splitext(input_path)
        if output_override:
            output_path = output_override
        elif args.in_place:
            output_path = input_path
        else:
            output_path = f"{base}_comfy{ext}"

        print(f"Loading: {input_path}")
        state = load_file(input_path)
        keys = list(state.keys())

        fam, why = looks_like_h3(keys)
        if fam != "h3":
            print(f"⚠ Input looks like: {fam} ({why})")
            print("  Expected MiniMax H3 (blocks.N.qkv_proj/fc1/fc2/token_refiner).")
            print("  Continuing anyway — verify layer names before use in ComfyUI.")

        # Classify how each key will be handled.
        counts = {"unchanged": 0, "musubi": 0, "diffusers": 0, "prefixed": 0, "skipped": 0}
        skipped_keys = []
        qkv_group_count = 0
        for k in keys:
            if DIFFUSERS_LORA_RE.match(k):
                counts["diffusers"] += 1
            else:
                nk, _ = convert_key(k)
                if nk is None:
                    counts["skipped"] += 1
                    skipped_keys.append(k)
                elif nk == k:
                    counts["unchanged"] += 1
                elif k.startswith(MUSUBI_PREFIX):
                    counts["musubi"] += 1
                else:
                    counts["prefixed"] += 1
        # qkv merge groups: one per fused qkv_proj target
        qkv_targets = {}
        for k in keys:
            m = DIFFUSERS_LORA_RE.match(k)
            if m and m.group(3).startswith("attn.to_"):
                main_idx, tr_idx, _, _ = m.groups()
                if tr_idx is not None:
                    qkv_targets[f"diffusion_model.token_refiner.blocks.{tr_idx}.attn.qkv_proj"] = True
                else:
                    qkv_targets[f"diffusion_model.blocks.{main_idx}.attn.qkv_proj"] = True
        qkv_group_count = len(qkv_targets)

        dtypes = {t.dtype for t in state.values()}
        print(f"  tensors:            {len(keys)}")
        print(f"  already ComfyUI:    {counts['unchanged']}")
        print(f"  musubi->diffusers2: {counts['musubi']}  (lora_unet_* -> diffusion_model.*)")
        print(f"  diffusers->comfy:   {counts['diffusers']}  (PEFT .default keys, {qkv_group_count} qkv merges)")
        print(f"  bare keys prefixed: {counts['prefixed']}")
        if counts["skipped"]:
            print(f"  ⚠ skipped:          {counts['skipped']} -> {skipped_keys[:5]}")
        if args.bf16:
            print(f"  dtype:              {dtypes} -> torch.bfloat16")
        else:
            print(f"  dtype:              {dtypes} (preserved)")

        if args.dry_run:
            print(f"  would write:        {output_path}")
            continue

        # Rename all keys; collect qkv groups for the fused merge.
        converted = {}
        qkv_groups = {}
        for k, t in state.items():
            nk, group = convert_key(k)
            if nk is None:
                continue  # skipped keys are dropped
            if group is None:
                converted[nk] = t
            else:
                qkv_groups.setdefault(group["target"], {})[
                    group["component"] + group["side"]] = t

        if qkv_groups:
            merged, n_groups = merge_qkv_groups(qkv_groups)
            converted.update(merged)
            print(f"  merged {n_groups} qkv groups -> fused qkv_proj "
                  f"(cat A's, block_diag B's, exact)")

        if args.bf16:
            converted = {k: t.to(torch.bfloat16) for k, t in converted.items()}

        meta = load_metadata(input_path)
        meta.setdefault("ss_base_model_version", "minimax_h3")
        meta.setdefault("ss_output_name", os.path.splitext(os.path.basename(output_path))[0])
        meta.setdefault("name", os.path.splitext(os.path.basename(output_path))[0])

        print(f"Saving: {output_path}")
        save_file(converted, output_path, metadata=meta)

        # Report a few converted names so the user can eyeball the result.
        sample = [k for k in converted if k.endswith("lora_A.weight")][:3]
        for k in sample:
            print(f"  e.g. {k}")
        print()


if __name__ == "__main__":
    main()
