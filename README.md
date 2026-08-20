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

## MiniMax H3 scripts

These handle ai-toolkit MiniMax H3 LoRAs (`diffusion_model.blocks.N.qkv_proj/fc1/fc2`
keys, bf16, `ss_base_model_version = minimax_h3` metadata). All preserve and
update the safetensors metadata that ComfyUI and ai-toolkit read, and validate
inputs so krea2 LoRAs (different layer names: `attn.gate/wk/wv/wo`) are flagged
rather than silently mangled.

### `convert_h3_lora_to_comfy.py`
Normalizes MiniMax H3 LoRA keys to ComfyUI's diffusers2 naming so the LoRA
adapter can find its target layers. ComfyUI matches the suffix pattern
(`{model_key}.lora_A.weight` / `.lora_B.weight`) against full model keys that
start with `diffusion_model.`; anything else logs "lora key not loaded".

Handles four input styles:

1. **Official turbo checkpoints** (`minimax_h3_turbo_4step` / `_8step`) ship
   with bare keys like `blocks.0.adaln_proj.linear.lora_A.weight` → the
   `diffusion_model.` prefix is prepended.
2. **musubi-tuner checkpoints** ship in kohya format:
   `lora_unet_blocks_9_mlp_fc2.lora_down.weight` / `.lora_up.weight` +
   `.alpha`. These are rewritten to
   `diffusion_model.blocks.9.mlp.fc2.lora_A.weight` / `.lora_B.weight`
   (underscore path → dotted, `lora_down`→`lora_A`, `lora_up`→`lora_B`).
   Compound module names (`out_proj`, `qkv_proj`, `token_refiner`, ...) are
   protected, so `attn_out_proj` stays `attn.out_proj`, not `attn.out.proj`.
   No transposition is needed: musubi-tuner stores down as `[rank, in]` and
   up as `[out, rank]`, exactly what ComfyUI expects (`up @ down`).
3. **Official FL2VA turbo** (`minimax_h3_fl2v_turbo_4step_v0.1`) ships in
   diffusers PEFT format with UNFUSED attention:
   `transformer_blocks.9.attn.to_q.lora_A.default.weight`,
   `token_refiner.refiner_blocks.1.ff.net.2.lora_B.default.weight`. Renamed
   to ComfyUI paths (`transformer_blocks.N`→`blocks.N`,
   `token_refiner.refiner_blocks.N`→`token_refiner.blocks.N`,
   `attn.to_out.0`→`attn.out_proj`, `ff.net.0.proj`→`mlp.fc1`,
   `ff.net.2`→`mlp.fc2`). The unfused `to_q/to_k/to_v` are merged into the
   fused `attn.qkv_proj`: A tensors concatenated (rank 3x), B tensors
   block-diagonalized — this reproduces `vstack([dq, dk, dv])` bitwise
   (verified), matching ComfyUI's `qkv_proj` split order (q, k, v).
4. **Already-converted keys** (`diffusion_model.` prefix) pass through
   untouched.

Alpha tensors are kept and renamed to `{model_key}.alpha` — ComfyUI reads
them and applies kohya's alpha/rank scaling. If alpha == rank (musubi-tuner
default, e.g. 32/32) the scale is 1.0 either way. Metadata is preserved and
tagged with `ss_base_model_version = minimax_h3`. Tensors are passed by
reference (no copy, no dtype promotion) unless `--bf16` is used to cast for
consistency with the bf16 files in the h3 folder.

Usage:
```bash
python convert_h3_lora_to_comfy.py MysticXXX_MMH3-step00001700.safetensors
python convert_h3_lora_to_comfy.py minimax_h3_fl2v_turbo_4step_v0.1.safetensors
python convert_h3_lora_to_comfy.py --in-place MysticXXX_MMH3-step00001700.safetensors
python convert_h3_lora_to_comfy.py --bf16 lora.safetensors --out converted.safetensors
python convert_h3_lora_to_comfy.py --dry-run minimax_h3_turbo_8step.safetensors
```

### `merge_h3_loras.py`
Merges multiple H3 LoRAs into one via stack-and-reduce (fast QR trick). Supports
weighted merging and mixed ranks.

Usage:
```bash
python merge_h3_loras.py /path/to/h3_folder --rank 32
python merge_h3_loras.py --files a.sft b.sft c.sft --rank 32 --out merged.safetensors
python merge_h3_loras.py /path/to/h3_folder --weights 0.7,0.3 --rank 32
```

Arguments:
- `folder` (positional) or `--files`: Input LoRA files.
- `--rank`: Target rank. Default `32` (H3 native).
- `--weights`: Comma-separated merge weights (default: equal).
- `--out`: Output file. Default `<folder>/merged_rank{R}.safetensors`.
- `--exclude`: Filenames to exclude (folder mode).
- `--device`: `cpu` or `cuda` (auto-detect).
- `--dry-run`: Scan and report without writing.

### `reduce_h3_rank.py`
Reduces H3 LoRA rank using the fast QR-based SVD (never materializes the full
weight matrix). Preserves metadata and dtype.

Usage:
```bash
python reduce_h3_rank.py model.safetensors --rank 16
python reduce_h3_rank.py model.safetensors --rank 8 --out reduced.safetensors
```

Arguments:
- `input`: Input H3 LoRA `.safetensors`.
- `--rank`: Target rank (required).
- `--out`: Output file.
- `--device`: `cpu` or `cuda`.

### `scale_h3.py`
Scales the **effective strength** of an H3 LoRA. Since `delta_W = B @ A`, scaling
one factor by `factor` gives exactly `factor` x strength. The krea2-era script
scaled every tensor (factor² in effect); this one does it right by default.

Usage:
```bash
./scale_h3.py 2.0 lora.safetensors          # 2x effective strength
./scale_h3.py 0.5 lora.safetensors --out halved.safetensors
./scale_h3.py 10.0 --dry-run lora.safetensors
./scale_h3.py 3.0 *.safetensors
./scale_h3.py 2.0 --both lora.safetensors   # legacy factor² behavior
```

Arguments:
- `factor`: Positive scaling factor.
- `inputs`: Files or globs.
- `--side B|A`: Which factor to scale (default `B`).
- `--both`: Legacy mode — scale both A and B (effective factor²).
- `--dry-run`: Report only.

---

### `prune_h3.py`
Prunes MiniMax H3 LoRA tensors by key substring, block index, or layer type.
Understands H3 key formats (`diffusion_model.blocks.<N>.{attn.qkv_proj|attn.out_proj|mlp.fc1|mlp.fc2|adaln_proj.linear}`
and `diffusion_model.token_refiner.blocks.<N>.*`). **Metadata is intentionally STRIPPED**
on output — a pruned LoRA is a new artifact, so ai-toolkit provenance tags are dropped.

Usage:
```bash
./prune_h3.py --match ".attn." file.safetensors
./prune_h3.py --blocks 4 7 10-13 file.safetensors
./prune_h3.py --layers attn file.safetensors          # qkv+out_proj
./prune_h3.py --layers mlp,adaln file.safetensors
./prune_h3.py --layers token_refiner file.safetensors
./prune_h3.py --match "!qkv_proj" file.safetensors    # keep-only mode
./prune_h3.py --match ".attn." --dry-run "*.safetensors"
```

Arguments:
- `--match`: Substrings to remove; prefix with `!` for keep-only (whitelist).
- `--blocks`: Block indexes or ranges (e.g. `4 7 10-13`).
- `--layers`: Comma-separated H3 layer types: `attn` (qkv+out_proj), `mlp` (fc1+fc2),
  `adaln`, `token_refiner`, `qkv`, `out_proj`, `fc1`, `fc2`.
- `--dry-run`: Report only. `--out`: Output file (single input only).

---

### `set_lora_metadata.py`
Replaces or edits safetensors LoRA metadata without touching a single tensor (tensors are re-serialized bit-identical).

Usage:
```bash
python set_lora_metadata.py lora.safetensors --set name="My Custom" --set ss_output_name=custom_v1
python set_lora_metadata.py lora.safetensors --del ss_learning_rate --del ss_epoch
python set_lora_metadata.py lora.safetensors --json meta.json --out out.safetensors
python set_lora_metadata.py lora.safetensors --clear --set name=Clean --dry-run
```

Arguments:
- `--set KEY=VALUE`: Add or overwrite a metadata key (repeatable).
- `--del KEY`: Remove a metadata key (repeatable).
- `--json FILE`: Load a full metadata dict from JSON (values stringified; dicts/lists become JSON strings).
- `--clear`: Start from an empty header (drop ALL existing metadata) instead of inheriting it.
- `--out`: Output path (default: `<stem>_meta.safetensors` next to input).
- `--dry-run`: Print the resulting diff without writing a file.

---

### `set_lora_metadata_silly.py`
The troll edition: wipes ALL real training info from a LoRA and replaces it with the legendary collab of Rick Sanchez x Alcaitiff x Chuck Norris. Negative steps, infinite datasets, negative loss, and a certified Chuck Norris fact. Tensors stay bit-identical; only the header gets lobotomized.

Usage:
```bash
python set_lora_metadata_silly.py lora.safetensors
python set_lora_metadata_silly.py lora.safetensors --out prank.safetensors
python set_lora_metadata_silly.py lora.safetensors --dry-run --joke-seed 7
```

Arguments:
- `--out`: Output path (default: `<stem>_silly.safetensors`).
- `--dry-run`: Preview the silly header without writing.
- `--joke-seed N`: Pick which jokes land in the header deterministically.

---

### `lora_similarity.py`
Calculates cosine similarity between two LoRA safetensors files, per-layer and overall.

Usage:
```bash
python lora_similarity.py --lora_a A.safetensors --lora_b B.safetensors
python lora_similarity.py --lora_a A.safetensors --lora_b B.safetensors --csv --top 20
python lora_similarity.py --lora_a A.safetensors --lora_b B.safetensors --json-out results.json
```

Arguments:
- `--lora_a` / `--lora_b`: Input LoRA files (required).
- `--top N`: Show top N most similar/dissimilar layers (default 10).
- `--csv`: Output per-layer results as CSV.
- `--json-out FILE`: Write full results as JSON.
- `--threshold FLOAT`: Only show layers with cosine similarity ≥ this value.
- `--decimals N`: Decimal places for numeric output (default 5).

Output:
- Overall weighted cosine similarity (weighted by element count).
- Per-block-group averages with visual bars.
- Similarity distribution histogram.
- Top most similar and most dissimilar layers.
- Lists keys only found in one LoRA and keys skipped due to shape mismatch.

Notes:
- Only compares tensors with the same key name AND same shape.
- Cosine similarity is computed on the flattened weight vector (standard LoRA weight similarity).
- Layers with cos ≥ 0.99 are nearly interchangeable; cos < 0.8 indicates meaningfully different learned features.

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
