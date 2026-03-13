import argparse
import os
import torch
from collections import OrderedDict

def remap_flux_lora_keys(old_state_dict):
    new_state_dict = OrderedDict()

    for old_key, value in old_state_dict.items():
        if not old_key.endswith('.weight'):
            new_state_dict[old_key] = value
            continue

        parts = old_key.split('.')
        if len(parts) < 4:
            print(f"Skipping unusual key: {old_key}")
            continue

        new_key_base = None

        # ── Double blocks ──
        if parts[0] == 'transformer_blocks':
            block_id = parts[1]
            sub = '.'.join(parts[2:])  # attn.to_out.0.lora_A.default.weight etc.

            if 'attn.to_out.0' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.img_attn.proj"
            elif 'attn.to_q' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.txt_attn.qkv"
            elif 'attn.to_k' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.txt_attn.qkv"
            elif 'attn.to_v' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.txt_attn.qkv"
            elif 'attn.add_q_proj' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.img_attn.qkv"
            elif 'attn.add_k_proj' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.img_attn.qkv"
            elif 'attn.add_v_proj' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.img_attn.qkv"
            elif 'attn.to_add_out' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.img_attn.proj"

            # FF parts
            elif 'ff.linear_in' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.txt_mlp.0"
            elif 'ff.linear_out' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.txt_mlp.2"
            elif 'ff_context.linear_in' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.img_mlp.0"
            elif 'ff_context.linear_out' in sub:
                new_key_base = f"diffusion_model.double_blocks.{block_id}.img_mlp.2"

        # ── Single blocks ──
        elif parts[0] == 'single_transformer_blocks':
            block_id = parts[1]
            sub = '.'.join(parts[2:])

            if 'attn.to_out' in sub:
                new_key_base = f"diffusion_model.single_blocks.{block_id}.linear2"
            elif 'attn.to_qkv_mlp_proj' in sub:
                new_key_base = f"diffusion_model.single_blocks.{block_id}.linear1"

        if new_key_base is None:
            print(f"Warning: unmatched key → {old_key}")
            continue

        # Determine A or B and build final suffix
        if old_key.endswith('.lora_A.default.weight'):
            suffix = ".lora_A.weight"
        elif old_key.endswith('.lora_B.default.weight'):
            suffix = ".lora_B.weight"
        else:
            print(f"Unexpected suffix in {old_key}")
            continue

        final_key = new_key_base + suffix
        new_state_dict[final_key] = value

    return new_state_dict


# ── CLI usage ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap Flux LoRA keys")
    parser.add_argument("input", help="Input .safetensors file")
    parser.add_argument("--out", default=None, help="Output .safetensors file")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        raise SystemExit(1)

    base, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".safetensors"
    output_path = args.out or f"{base}_renamed{ext}"

    print("Loading LoRA...")

    try:
        from safetensors.torch import load_file, save_file
        state_dict = load_file(input_path, device="cpu")
        print(f"Loaded {len(state_dict)} keys using safetensors.")
    except ImportError:
        print("safetensors library not found.")
        print("Install it with:   pip install safetensors")
        raise SystemExit(1)
    except Exception as e:
        print(f"Failed to load with safetensors: {e}")
        print("Falling back to torch.load (unsafe for .pt files only)...")
        state_dict = torch.load(input_path, map_location="cpu", weights_only=False)

    # Unwrap if needed (some trainers wrap the dict)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # rare ai-toolkit style wrapper
    elif len(state_dict) > 0 and isinstance(next(iter(state_dict.values())), dict):
        # sometimes it's {"lora": {...}} or similar — adjust if needed
        pass

    print("Remapping keys...")
    new_sd = remap_flux_lora_keys(state_dict)

    # Optional: print a few new keys to verify
    print("Sample remapped keys:")
    for k in list(new_sd.keys())[:8]:
        print("   ", k)

    print(f"Saving to {output_path}")
    save_file(new_sd, output_path)
    print("Done! You can now try loading this LoRA in ComfyUI / Forge etc.")
