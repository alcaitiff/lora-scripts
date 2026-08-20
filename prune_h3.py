#!/usr/bin/env python3
"""
Prune MiniMax H3 LoRA tensors by key substring / block / layer type.

Understands H3 key formats:
  diffusion_model.blocks.<N>.{attn.qkv_proj|attn.out_proj|mlp.fc1|mlp.fc2|adaln_proj.linear}
  diffusion_model.token_refiner.blocks.<N>.{attn.qkv_proj|attn.out_proj|mlp.fc1|mlp.fc2}

Modes:
- Remove-matching (default): drops any key containing a --match / --blocks / --layers token.
- Keep-only: if any --match token starts with '!', keeps only keys matching at least
  one '!' token, removing everything else.

METADATA: intentionally STRIPPED on output. A pruned LoRA is a new artifact; the
ai-toolkit provenance tags (ss_*, sshs_*) from training are removed. The output
file contains only tensors.

Usage:
  ./prune_h3.py --match ".attn." file.safetensors
  ./prune_h3.py --blocks 4 7 10-13 file.safetensors
  ./prune_h3.py --layers attn file.safetensors
  ./prune_h3.py --layers mlp --blocks 8-12 file.safetensors
  ./prune_h3.py --layers adaln file.safetensors
  ./prune_h3.py --match "!qkv_proj" file.safetensors     # keep-only
  ./prune_h3.py --match ".attn." --dry-run "*.safetensors"
"""

from safetensors.torch import load_file, save_file
import argparse
import glob
import os
import sys

# H3 layer types -> key substrings
LAYER_MATCHES = {
    "attn": [".attn.qkv_proj.", ".attn.out_proj."],
    "mlp": [".mlp.fc1.", ".mlp.fc2."],
    "adaln": [".adaln_proj."],
    "token_refiner": ["token_refiner."],
    "qkv": [".attn.qkv_proj."],
    "out_proj": [".attn.out_proj."],
    "fc1": [".mlp.fc1."],
    "fc2": [".mlp.fc2."],
}


def split_match_tokens(tokens):
    include = []
    exclude = []
    for tok in tokens or []:
        if tok.startswith("!"):
            if tok == "!":
                raise ValueError("Invalid --match token '!': empty include substring")
            include.append(tok[1:])
        else:
            exclude.append(tok)
    return include, exclude


def should_remove(key: str, include_substrings, exclude_substrings) -> bool:
    if include_substrings:
        if not any(s in key for s in include_substrings):
            return True
    return any(s in key for s in exclude_substrings)


def is_block_token(tok: str) -> bool:
    if tok.isdigit():
        return True
    if "-" in tok:
        parts = tok.split("-", 1)
        return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()
    return False


def parse_block_tokens(tokens):
    out = []
    for tok in tokens:
        if "-" in tok:
            parts = tok.split("-", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid block range: {tok}")
            start, end = int(parts[0]), int(parts[1])
            if end < start:
                raise ValueError(f"Invalid block range: {tok}")
            out.extend(range(start, end + 1))
        else:
            out.append(int(tok))
    return out


def looks_like_h3(keys):
    """Return ('h3'|'krea2'|'unknown', reason) from key list."""
    if not any("lora_A.weight" in k for k in keys):
        return "unknown", "no .lora_A.weight tensors found"
    joined = "|" + "|".join(keys)
    h3_hits = sum(1 for m in ["qkv_proj", "token_refiner", ".fc1.", ".fc2."] if m in joined)
    krea2_hits = sum(1 for m in [".gate.", ".wk.", ".wv.", ".wo."] if m in joined)
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
    description="Prune MiniMax H3 LoRA tensors by key substring / block / layer type",
)
parser.add_argument(
    "--match", nargs="+", required=False,
    help="Substrings to match against tensor keys. Prefix with '!' to keep only matching keys.",
)
parser.add_argument(
    "--blocks", nargs="+",
    help="Block indexes or ranges (e.g. 4 7 10-13) matching .blocks.<N>. and token_refiner .blocks.<N>.",
)
parser.add_argument(
    "--layers", type=str, default=None,
    help=f"H3 layer types to remove, comma-separated: {', '.join(sorted(LAYER_MATCHES.keys()))} "
         f"(attn=qkv+out_proj, mlp=fc1+fc2, adaln=adaln_proj.linear, token_refiner=all refiner blocks)",
)
parser.add_argument(
    "inputs", nargs="*", help="Input .safetensors files or glob patterns",
)
parser.add_argument("--dry-run", action="store_true",
                    help="Show which keys would be removed without writing output files")
parser.add_argument("--out", default=None, help="Output file (single input only)")
args = parser.parse_args()


# ---- infer inputs from --blocks / --match tokens if positional args were swallowed ----
if args.blocks and not args.inputs:
    inferred, remaining = [], []
    for token in args.blocks:
        if is_block_token(token):
            remaining.append(token)
            continue
        if any(ch in token for ch in "*?["):
            m = glob.glob(token)
            if m:
                inferred.extend(m)
                continue
        inferred.append(token)
    if inferred:
        args.inputs = inferred
    args.blocks = remaining

if not args.inputs and args.match:
    inferred, remaining = [], []
    for token in args.match:
        if any(ch in token for ch in "*?["):
            m = glob.glob(token)
            if m:
                inferred.extend(m)
                continue
        if os.path.exists(token):
            inferred.append(token)
        else:
            remaining.append(token)
    if inferred:
        args.inputs = inferred
        args.match = remaining

if not args.inputs:
    print("No input files provided.")
    sys.exit(1)

# ---- build match lists ----
match_list = list(args.match or [])
try:
    include_matches, exclude_matches = split_match_tokens(match_list)
except ValueError as e:
    print(str(e))
    sys.exit(1)

layer_matches = []
if args.layers:
    for layer in args.layers.split(","):
        layer = layer.strip()
        if layer not in LAYER_MATCHES:
            print(f"Unknown layer type: '{layer}'. Choices: {', '.join(sorted(LAYER_MATCHES.keys()))}")
            sys.exit(1)
        layer_matches.extend(LAYER_MATCHES[layer])

block_matches = []
if args.blocks:
    try:
        block_ids = parse_block_tokens(args.blocks)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    for i in sorted(set(block_ids)):
        block_matches.append(f".blocks.{i}.")

exclude_matches = list(exclude_matches) + layer_matches + block_matches
if not include_matches and not exclude_matches:
    print("No match criteria provided. Use --match, --blocks, and/or --layers.")
    sys.exit(1)

# ---- resolve files ----
input_files = []
for pattern in args.inputs:
    matches = glob.glob(pattern)
    if not matches:
        print(f"Warning: no files matched '{pattern}'")
    input_files.extend(matches)
seen = set()
input_files = [f for f in input_files if not (f in seen or seen.add(f))]
if not input_files:
    print("No input files to process.")
    sys.exit(1)

output_override = None
if args.out:
    if len(input_files) != 1:
        print("--out can only be used when exactly one input file is provided.")
        sys.exit(1)
    output_override = args.out

# ---- process ----
for input_lora in input_files:
    base, ext = os.path.splitext(input_lora)
    output_lora = output_override or f"{base}_pruned{ext}"
    mode = "keep-only" if include_matches else "remove-matching"

    print(f"\nLoading: {input_lora}")
    state = load_file(input_lora)

    fam, why = looks_like_h3(list(state.keys()))
    if fam != "h3":
        print(f"⚠ Input looks like: {fam} ({why})")
        print("  Expected MiniMax H3 (diffusion_model.blocks.N.qkv_proj/fc1/fc2).")
        print("  Continuing anyway.")

    new_state = {}
    removed_keys = []
    for key, tensor in state.items():
        if should_remove(key, include_matches, exclude_matches):
            removed_keys.append(key)
            continue
        new_state[key] = tensor

    if args.dry_run:
        print("Dry run: no output file written.")
    else:
        print(f"Saving: {output_lora}")
        # NOTE: no metadata arg -> metadata intentionally stripped
        save_file(new_state, output_lora)

    print("===================================")
    print(f"File            : {input_lora}")
    print(f"Total tensors   : {len(state)}")
    print(f"Removed tensors : {len(removed_keys)}")
    print(f"Kept tensors    : {len(new_state)}")
    print(f"Match mode      : {mode}")
    print(f"Metadata        : STRIPPED (by design)")
    if args.dry_run:
        print("Mode            : dry-run (no file written)")
    print("===================================")

    if removed_keys:
        print(f"Removed keys ({len(removed_keys)}):")
        for k in removed_keys:
            print(" -", k)
    else:
        print("No keys were removed.")
