#!/usr/bin/env python3
"""
Attribution-based audio scaling for MiniMax H3 LoRAs (Musubi format).

The LoRA has NO audio-specific modules: every module (blocks.N x {attn_qkv_proj,
attn_out_proj, mlp_fc1, mlp_fc2}) is shared between the audio and video streams.
So "the audio component" cannot be deleted by name. Instead this script runs a
real forward + backward through the frozen base transformer with the LoRA
applied, using Musubi's own `joint_velocity_loss`:

  * audio-only loss  (video_weight=0, audio_weight=1) -> per-module grad norms
  * video-only loss  (video_weight=1, audio_weight=0) -> per-module grad norms

  dominance_i = audio_norm_i / (audio_norm_i + video_norm_i)

A module with dominance 1.0 is pure-audio; 0.0 is pure-video. Then each module
is scaled by

  scale_i = 1 - (1 - audio_scale) * dominance_i

so --audio-scale 0.5 halves the audio contribution (pure-audio modules x0.5,
pure-video modules untouched) and --audio-scale 0.0 zeroes it (pure-audio
modules x0.0). Per scale_h3.py precedent only ONE factor (lora_up, the B side)
is scaled, so delta_W = up @ down scales linearly.

Caches must be Musubi MMH3 caches (video+audio latents + text embeds + audio
loss mask), e.g. /media/queen/Workspace/ai-toolkit/datasets/xxx_video_h3/cache.

Usage:
  PYTHONPATH=/home/alan/Workspace/musubi-tuner/src python h3_audio_scale.py \
      --lora /media/queen/Backup/CustomModels/Lora/h3/MysticXXX_MMH3-step00007900.safetensors \
      --audio-scale 0.5 \
      --out /media/queen/Backup/CustomModels/Lora/h3/MysticXXX_MMH3-step00007900_audio50.safetensors \
      --attribution-json /home/alan/Workspace/lora-scripts/attr_MysticXXX_MMH3.json

  # second LoRA reuses the saved attribution (no model reload):
  python h3_audio_scale.py \
      --lora /media/queen/Backup/CustomModels/Lora/h3/MysticXXX_MMH3-step00007900.safetensors \
      --audio-scale 0.0 \
      --out /media/queen/Backup/CustomModels/Lora/h3/MysticXXX_MMH3-step00007900_noaudio.safetensors \
      --attribution-json /home/alan/Workspace/lora-scripts/attr_MysticXXX_MMH3.json

  # attribution only, no LoRA written:
  python h3_audio_scale.py --lora ... --audio-scale 1.0 --attribution-only --attribution-json attr.json
"""

import argparse
import glob
import json
import os
import random
import sys
import time

import torch

MUSUBI_SRC = "/home/alan/Workspace/musubi-tuner/src"
if MUSUBI_SRC not in sys.path:
    sys.path.insert(0, MUSUBI_SRC)

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.minimax_h3.backend import create_training_backend
from musubi_tuner.minimax_h3.training import joint_velocity_loss, prepare_joint_noisy_inputs
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.utils.model_utils import remove_dtype_suffix

DEFAULT_DIT = "/home/alan/Workspace/Fizgig/models/minimax_h3_fl2va_bf16.safetensors"
DEFAULT_CACHE_DIR = "/media/queen/Workspace/ai-toolkit/datasets/xxx_video_h3/cache"


def load_metadata(path: str) -> dict:
    with safe_open(path, framework="pt") as f:
        return dict(f.metadata()) if f.metadata() else {}


def build_batch(latent_path: str, te_path: str) -> dict:
    """Replicate Musubi bucket.py key handling for one MMH3 cached item."""
    sd = {**load_file(latent_path), **load_file(te_path)}
    batch: dict[str, list] = {}
    varlen_keys = set()
    for key, tensor in sd.items():
        is_varlen = key.startswith("varlen_")
        content_key = key[len("varlen_"):] if is_varlen else key
        if not content_key.endswith("_mask"):
            content_key = remove_dtype_suffix(content_key)
            if content_key.startswith("latents_"):
                content_key = content_key.rsplit("_", 1)[0]
        batch.setdefault(content_key, []).append(tensor)
        if is_varlen:
            varlen_keys.add(content_key)
    out = {}
    for key, tensors in batch.items():
        if key in varlen_keys:
            out[key] = tensors  # keep as list; _one_conditioning_item unwraps
        else:
            out[key] = torch.stack(tensors)  # [1, ...]
    return out


def pick_items(cache_dir: str, count: int, seed: int) -> list[tuple[str, str]]:
    latent_files = sorted(glob.glob(os.path.join(cache_dir, "*_mmh3.safetensors")))
    pairs = []
    for lf in latent_files:
        tokens = os.path.basename(lf).split("_")
        if len(tokens) < 3:
            continue
        frame_tag = tokens[-3]
        item_key = "_".join(tokens[:-3])
        te_path = os.path.join(cache_dir, f"{item_key}_{frame_tag}_mmh3_te.safetensors")
        if os.path.isfile(te_path):
            pairs.append((lf, te_path))
    if not pairs:
        raise FileNotFoundError(f"no MMH3 cache pairs found under {cache_dir}")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:count]


def module_grad_norm(lora) -> float:
    """Frobenius norm of combined lora_down/lora_up gradients."""
    total = 0.0
    for name, param in lora.named_parameters():
        if param.grad is not None:
            total += float(param.grad.float().square().sum().item())
    return total ** 0.5


def run_attribution(args, transformer, network, backend) -> dict:
    device = torch.device(args.device)
    pairs = pick_items(args.cache_dir, args.items, args.seed)
    print(f"[attribution] using {len(pairs)} cached item(s), {args.sigmas} sigma(s) each")

    # network module name -> accumulated audio/video grad norms
    acc = {lora.lora_name: {"audio": 0.0, "video": 0.0} for lora in network.unet_loras}
    passes = 0

    for latent_path, te_path in pairs:
        batch = build_batch(latent_path, te_path)
        video_latents = batch["latents"].to(device=device, dtype=torch.bfloat16)
        audio_latents = batch["latents_audio"].to(device=device, dtype=torch.bfloat16)
        audio_mask = batch.get("audio_loss_mask")
        if audio_mask is not None:
            audio_mask = audio_mask.to(device=device)
        print(f"[attribution] item {os.path.basename(latent_path)} "
              f"video {tuple(video_latents.shape)} audio {tuple(audio_latents.shape)}")

        for _ in range(args.sigmas):
            base_sigma = torch.full((1,), random.uniform(0.02, 0.98), device=device)
            video_noise = torch.randn_like(video_latents)
            audio_noise = torch.randn_like(audio_latents)
            video_latents.requires_grad_(True)
            audio_latents.requires_grad_(True)

            inputs = prepare_joint_noisy_inputs(
                video_latents, audio_latents, video_noise, audio_noise, base_sigma,
                video_shift=args.h3_shift_video, audio_shift=args.h3_shift_audio,
            )

            # ---- audio-only loss ----
            network.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                prediction = backend.predict_training(
                    transformer, batch,
                    inputs.video, inputs.audio,
                    inputs.video_timestep, inputs.audio_timestep,
                    conditioning="prompt",
                )
            loss = joint_velocity_loss(
                prediction, inputs,
                video_mask=None, audio_mask=audio_mask,
                balance="modality", video_weight=0.0, audio_weight=1.0,
            ).loss
            loss.backward(retain_graph=True)  # same graph feeds the video pass below
            for lora in network.unet_loras:
                acc[lora.lora_name]["audio"] += module_grad_norm(lora)

            # ---- video-only loss ----
            network.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                prediction = backend.predict_training(
                    transformer, batch,
                    inputs.video, inputs.audio,
                    inputs.video_timestep, inputs.audio_timestep,
                    conditioning="prompt",
                )
            loss = joint_velocity_loss(
                prediction, inputs,
                video_mask=None, audio_mask=audio_mask,
                balance="modality", video_weight=1.0, audio_weight=0.0,
            ).loss
            loss.backward()
            for lora in network.unet_loras:
                acc[lora.lora_name]["video"] += module_grad_norm(lora)

            network.zero_grad(set_to_none=True)
            video_latents.requires_grad_(False)
            audio_latents.requires_grad_(False)
            passes += 1
            torch.cuda.synchronize(device)

    attribution = {}
    for name, norms in acc.items():
        attribution[name] = {
            "audio_norm": norms["audio"] / passes,
            "video_norm": norms["video"] / passes,
        }
    return attribution


def summarize(attribution: dict, audio_scale: float, top: int = 15) -> None:
    rows = []
    for name, n in attribution.items():
        a, v = n["audio_norm"], n["video_norm"]
        dominance = a / (a + v + 1e-12)
        rows.append((name, a, v, dominance))
    print(f"\n[attribution] audio-dominant modules (top {top}):")
    for name, a, v, d in sorted(rows, key=lambda r: -r[3])[:top]:
        print(f"  {name:55s} audio={a:10.3f} video={v:10.3f} dominance={d:5.2f}")
    print(f"[attribution] video-dominant modules (top {top}):")
    for name, a, v, d in sorted(rows, key=lambda r: r[3])[:top]:
        print(f"  {name:55s} audio={a:10.3f} video={v:10.3f} dominance={d:5.2f}")

    # per-block audio energy (lora_unet_blocks_{N}_{module})
    by_block: dict[int, float] = {}
    for name, n in attribution.items():
        try:
            block = int(name.split("_")[3])
        except (IndexError, ValueError):
            continue
        by_block[block] = by_block.get(block, 0.0) + n["audio_norm"]
    total = sum(by_block.values()) or 1.0
    print("\n[attribution] audio gradient energy by block:")
    for block, energy in sorted(by_block.items(), key=lambda kv: -kv[1]):
        print(f"  block {block:3d}: {energy / total * 100:5.1f}%  ({energy:.1f})")

    if audio_scale < 1.0:
        eff = [1.0 - (1.0 - audio_scale) * (n["audio_norm"] / (n["audio_norm"] + n["video_norm"] + 1e-12))
               for n in attribution.values()]
        print(f"\n[scale] audio_scale={audio_scale}: per-module scale range "
              f"min={min(eff):.3f} max={max(eff):.3f} mean={sum(eff)/len(eff):.3f}")


def scale_lora(state: dict, attribution: dict, audio_scale: float) -> dict:
    """Scale lora_up (B side) per module; keep everything else identical."""
    scaled = {}
    n_scaled = 0
    for key, tensor in state.items():
        out = tensor
        if key.endswith(".lora_up.weight"):
            name = key[: -len(".lora_up.weight")]
            n = attribution.get(name)
            if n is not None:
                a, v = n["audio_norm"], n["video_norm"]
                dominance = a / (a + v + 1e-12)
                factor = 1.0 - (1.0 - audio_scale) * dominance
                out = tensor * factor
                n_scaled += 1
        scaled[key] = out
    print(f"[scale] scaled {n_scaled} lora_up tensors")
    return scaled


def main():
    parser = argparse.ArgumentParser(
        description="Attribution-based audio scaling for MiniMax H3 LoRAs (Musubi format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lora", required=True, help="Source Musubi-format LoRA .safetensors")
    parser.add_argument("--audio-scale", type=float, required=True,
                        help="Target audio strength (1.0 = unchanged, 0.5 = half, 0.0 = zeroed)")
    parser.add_argument("--out", default=None, help="Output LoRA path (required unless --attribution-only)")
    parser.add_argument("--attribution-only", action="store_true",
                        help="Run attribution and print the ranking, but do not write a LoRA")
    parser.add_argument("--attribution-json", default=None,
                        help="Save attribution here after computing; if it exists, reuse it (skips model load)")
    parser.add_argument("--dit", default=DEFAULT_DIT, help="Base transformer checkpoint")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="MMH3 latent cache directory")
    parser.add_argument("--items", type=int, default=3, help="Number of cached items to attribute over (default 3)")
    parser.add_argument("--sigmas", type=int, default=4, help="Sigma samples per item (default 4)")
    parser.add_argument("--blocks-to-swap", type=int, default=28,
                        help="Transformer blocks swapped to CPU (default 28, matches training)")
    parser.add_argument("--use-pinned-memory", action="store_true", default=True,
                        help="Use pinned host memory for block swap (default True)")
    parser.add_argument("--h3-shift-video", type=float, default=12.0)
    parser.add_argument("--h3-shift-audio", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.audio_scale <= 1.0:
        parser.error("--audio-scale must be in [0.0, 1.0]")
    if args.audio_scale >= 1.0 and not args.attribution_only:
        parser.error("--audio-scale 1.0 writes an unchanged LoRA; use --attribution-only instead")
    if not args.attribution_only and not args.out:
        parser.error("--out is required unless --attribution-only")

    attribution = None
    if args.attribution_json and os.path.isfile(args.attribution_json):
        with open(args.attribution_json) as f:
            attribution = json.load(f)
        print(f"[attribution] reused saved attribution from {args.attribution_json}")

    if attribution is None:
        device = torch.device(args.device)
        print(f"[load] creating training backend for {args.dit}")
        t0 = time.time()
        backend = create_training_backend(
            model=args.dit,
            device="cpu" if args.blocks_to_swap > 0 else str(device),
            dtype="bfloat16",
            mode="fl2va",
            attention_mode="torch",   # --sdpa
            split_attention=False,
            adaln_rank=16,            # --h3_adaln_rank 16 (matches training)
            convrot_int8=True,
            convrot_int8_bwd="bf16",
            convrot_int8_fwd="bf16",
            quantization_device=str(device),
            int8_convrot=False,
        )
        transformer = backend.get_training_transformer()
        transformer.eval()
        transformer.requires_grad_(False)

        if args.blocks_to_swap > 0:
            swap_config = BlockSwapConfig(
                device=device,
                supports_backward=True,
                use_pinned_memory=args.use_pinned_memory,
                h2d_only=True,          # frozen-base LoRA training: H2D-only streaming
                ring_size=2,
                granularity="block",
            )
            transformer.enable_block_swap(args.blocks_to_swap, swap_config)
            transformer.move_to_device_except_swap_blocks(device)
        transformer.enable_gradient_checkpointing(activation_cpu_offloading=True)
        transformer.prepare_block_swap_before_forward()
        print(f"[load] transformer ready in {time.time() - t0:.1f}s")

        print(f"[lora] building network from {args.lora}")
        weights_sd = load_file(args.lora)
        network = lora_minimax_h3.create_arch_network_from_weights(1.0, weights_sd, unet=transformer)
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.load_weights(args.lora)
        for p in network.parameters():
            p.requires_grad_(True)
        network.to(device)
        print(f"[lora] {len(network.unet_loras)} LoRA modules applied")

        attribution = run_attribution(args, transformer, network, backend)

        if args.attribution_json:
            with open(args.attribution_json, "w") as f:
                json.dump(attribution, f, indent=2)
            print(f"[attribution] saved to {args.attribution_json}")

        summarize(attribution, args.audio_scale)

        del transformer, network
        torch.cuda.empty_cache()
    else:
        summarize(attribution, args.audio_scale)

    if args.attribution_only:
        print("[done] attribution only; no LoRA written")
        return

    print(f"[lora] scaling {args.lora} with audio_scale={args.audio_scale} -> {args.out}")
    state = load_file(args.lora)
    scaled = scale_lora(state, attribution, args.audio_scale)
    meta = load_metadata(args.lora)
    meta["ss_audio_scale"] = str(args.audio_scale)
    meta["ss_audio_attribution"] = "musubi joint_velocity_loss grad-norm dominance"
    meta["ss_audio_items"] = str(args.items)
    meta["ss_audio_sigmas"] = str(args.sigmas)
    meta["ss_audio_seed"] = str(args.seed)
    if args.attribution_json:
        meta["ss_audio_attribution_file"] = os.path.basename(args.attribution_json)
    save_file(scaled, args.out, metadata=meta)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
