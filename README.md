# LoRA Scripts

A small collection of one-off utilities for inspecting and transforming LoRA `.safetensors` files.

## Prerequisites

- Python 3.9+ (3.10+ recommended)
- Install deps:

```bash
pip install torch safetensors
```

Notes:
- Some scripts can use CUDA if available. CPU fallback is supported unless noted.
- All scripts operate on `.safetensors` files unless otherwise stated.
- Most scripts support `--out` to override the output filename.

---

## Scripts

### `lokr-to-lora.py`
Converts a LoKr `.safetensors` into a standard LoRA format (Flux-compatible), using SVD and writing LoRA A/B weights.

Usage:
```bash
python lokr-to-lora.py path/to/input_lokr.safetensors \
  --rank 16 \
  --out path/to/output_lora.safetensors \
  --device cuda
```

Arguments:
- `lokr_file` (positional): Input LoKr `.safetensors` file.
- `--out` / `--output`: Output path. Default is `<input>_converted_lora.safetensors`.
- `--rank`: Target LoRA rank. Default `16`.
- `--device`: `cpu` or `cuda`. Default auto-detects.

Output:
- Writes a new LoRA safetensors file with `.lora_A.weight` and `.lora_B.weight` keys.

---

### `reduce_lora_rank.py`
Reduces LoRA rank using SVD. Uses GPU if available.

Usage:
```bash
python reduce_lora_rank.py input.safetensors --rank 8 --out output.safetensors
```

Arguments:
- `input`: Input LoRA `.safetensors`.
- `output` (optional positional): Output file.
- `--out`: Output file (preferred).
- `--rank`: Target rank (required).

Notes:
- Uses CUDA if available; otherwise CPU.
- Preserves original tensor dtype on output.

---

### `merge_loras_verbose.py`
Merges two LoRAs with detailed per-layer diagnostics. Supports rank mismatches.

Usage:
```bash
python merge_loras_verbose.py \
  --lora_a A.safetensors \
  --lora_b B.safetensors \
  --alpha_a 0.7 \
  --alpha_b 0.3 \
  --out merged.safetensors
```

Arguments:
- `--lora_a`: First LoRA file (required).
- `--lora_b`: Second LoRA file (required).
- `--alpha_a`: Weight for LoRA A (default `0.7`).
- `--alpha_b`: Weight for LoRA B (default `0.3`).
- `--out`: Output file. If omitted, auto-named based on inputs and weights.

Notes:
- If ranks differ, the smaller rank is scaled by `sqrt(R_big/R_small)`.
- Prints norms and contribution estimates for each layer.

---

### `lora_shapes.py`
Prints per-layer tensor shapes and element counts for a LoRA file.

Usage:
```bash
python lora_shapes.py path/to/lora.safetensors --out report.txt
```

Output:
- Groups keys by `double_blocks.<id>` and `single_blocks.<id>`.
- Includes a quick summary of `qkv`-related keys.
- With `--out`, also writes the report to a text file.

---

### `mute.py`
Zeros out attention LoRA tensors that match specific keys, writing a new file with `_muted` suffix.

Usage:
```bash
python mute.py file1.safetensors file2.safetensors
python mute.py "Mystic*.safetensors"
python mute.py input.safetensors --out muted.safetensors
```

Behavior:
- Matches keys like `diffusion_model.layers.*.attention.*` ending with `.lora_A.weight` or `.lora_B.weight`.
- Writes `<input>_muted.safetensors` for each input file.
- `--out` can be used for a single input file to override the output name.

---

### `prune.py`
Removes tensors whose keys contain any of the provided substrings, writing a new file with `_pruned` suffix.

Usage:
```bash
python prune.py --match ".attention." ".layers.20." file1.safetensors file2.safetensors
python prune.py --match ".attention." "to_k" "name*.safetensors"
python prune.py --match ".attention." --dry-run "name*.safetensors"
python prune.py --match ".attention." --blocks 4 7 10-13 lora.safetensors
python prune.py --blocks 4 7 10-13 lora.safetensors
python prune.py --match ".attention." input.safetensors --out pruned.safetensors
python prune.py --match "!to_k" "!to_v" input.safetensors  # keep-only mode
```

Behavior:
- Default: removes any key that contains one of the `--match` substrings.
- If any `--match` substring starts with `!`, the script switches to keep-only mode and removes keys that do **not** match any `!` substring (non-`!` substrings are still removed as usual).
- Writes `<input>_pruned.safetensors` for each input file.
- With `--dry-run`, prints what would be removed and does not write output.
- If inputs are omitted, the script will try to infer any existing files/globs that were accidentally included in `--match`.
- `--blocks` adds match substrings like `block.<n>.`, `blocks.<n>.`, and `transformer_blocks.<n>.` and works alongside `--match`.
- `--out` can be used for a single input file to override the output name.

---

### `prune_and_scale.py`
Prunes attention layers and scales LoRA layers by layer index ranges.

Usage:
```bash
python prune_and_scale.py input.safetensors \
  --scale-range 0 5 0.7 \
  --scale-range 6 10 1.2 \
  --out output_pruned_scaled.safetensors \
  --debug
```

Arguments:
- `input`: Input `.safetensors` file.
- `--scale-range START END MULT`: Required; may be repeated.
- `--out` / `--output`: Output file. Default: `<input>_pruned_scaled.safetensors`.
- `--debug`: Print per-layer before/after stats.

Behavior:
- Prunes any key that contains `.attention.`.
- Scales tensors whose key includes `.layers.<index>.`.

---

### `rename-from-modeldiff.py`
Converts ModelDiff-style `lora_unet_*` keys to the `diffusion_model.*` format.

Usage:
```bash
python rename-from-modeldiff.py path/to/input.safetensors --out converted.safetensors
```

Output:
- Writes `<input>_converted.safetensors`.
- Only keys starting with `lora_unet_` are converted.
- Use `--out` to override the output name.

---

### `rename.py`
Remaps FLUX-style LoRA keys (`transformer_blocks` / `single_transformer_blocks`) to
`diffusion_model.double_blocks.*` / `diffusion_model.single_blocks.*`.

```bash
python rename.py path/to/input.safetensors --out renamed.safetensors
```

Notes:
- Uses `safetensors` when available; otherwise falls back to `torch.load` (unsafe for unknown `.pt` files).
- Prints sample remapped keys for quick verification.

---

### `lora_expand.py`
Fixes broken `qkv.lora_B.weight` tensors that are too small by tripling the first dimension.

```bash
python lora_expand.py path/to/input.safetensors --out expanded.safetensors
```

Behavior:
- If a `qkv.lora_B.weight` tensor has shape `[4096, ...]`, it is expanded to `[12288, ...]` by concatenation.

---

## Tips

- Prefer working on copies of your files. Most scripts already write a new output file.
- If a script fails due to memory on GPU, retry with CPU options (where available).
