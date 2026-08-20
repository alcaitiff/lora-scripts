#!/usr/bin/env python3
"""
The Silly Metadata Swapper: strip ALL real training info from a LoRA and
replace it with the legendary collab of Rick Sanchez x Alcaitiff x Chuck Norris.

Perfect for: trolling your friends on Civitai, confusing future-you, or
marketing a model as 'trained for negative epochs on an infinite dataset'.

Tensors are untouched (bit-identical); only the header gets lobotomized.

Usage:
  python set_lora_metadata_silly.py lora.safetensors
  python set_lora_metadata_silly.py lora.safetensors --out prank.safetensors
  python set_lora_metadata_silly.py lora.safetensors --dry-run
"""

import argparse
import json
import os
import random
import sys

from safetensors.torch import load_file, save_file

CHUCK_FACTS = [
    "Chuck Norris doesn't train LoRAs. LoRAs train Chuck Norris.",
    "Chuck Norris counted to infinity. Twice.",
    "When Chuck Norris runs a backward pass, the gradient runs forward out of fear.",
    "Chuck Norris's LoRA has no rank. It just IS.",
    "The only person who has ever overfitted Chuck Norris is Chuck Norris.",
    "Chuck Norris can divide by zero, then multiply by it again.",
    "Chuck Norris's loss function is Chuck Norris. It never converges, it intimidates.",
    "Chuck Norris once trained a model on the test set. It was the best model ever.",
    "Chuck Norris doesn't need a GPU. The GPU needs him.",
    "Chuck Norris roundhouse-kicked a NaN and it became the number 1.",
]

RICK_QUOTES = [
    "Wubba lubba dub dub! This LoRA is literally perfect, it doesn't need training.",
    "I trained this in a universe where gradients don't exist. Came out better.",
    "That's a big lie, this model doesn't have audio. It has ME.",
    "Don't think about it, Morty. The metadata is what it is.",
    "It's fine, Morty. It's just a LoRA. A LoRA that can punch through dimensions.",
    "I turned myself into a LoRA, Morty! I'm LoRA Rick!",
    "The concept of steps is a social construct, Morty. I trained this in one cronenberg second.",
    "AI is just AI, Morty. But this LoRA? This LoRA is a weapon.",
]

ALCAITIFF_LINES = [
    "Alcaitiff approved this model. Visually, emotionally, spiritually.",
    "Quality control by Alcaitiff: 10/10, would collab again.",
    "Alcaitiff says: the renders are clean, the audio is whatever, ship it.",
    "This collab happened because Alcaitiff asked and Rick said 'fine'.",
    "Alcaitiff provided the vibes, Rick the science, Chuck the roundhouse.",
]

SILLY_METADATA = {
    # ── The collab itself ──
    "name": "Rick Sanchez x Alcaitiff x Chuck Norris: The Multiverse Collab",
    "ss_output_name": "rick_x_alcaitiff_x_chuck_collab_v9000",
    "ss_training_comment": "Trained inside a Plumbus. Do NOT ask how the Plumbus works.",
    "modelspec.architecture": "mini-morty/lorax",
    "modelspec.title": "The Collab That Shouldn't Exist",
    "modelspec.trigger_word": "wubba lubba roundhouse",
    "ss_sd_model_name": "citadel_of_ricks_v3.safetensors",
    "ss_base_model_version": "Schwifty-9K-Morty-Proof (Chuck Norris Edition)",
    "sshs_model_hash": "0xDEADBEEFCAFE1234",
    "sshs_legacy_hash": "0xC0FFEE",  # it's a coffee shop now

    # ── Impossible numbers ──
    "ss_network_dim": "69",
    "ss_network_alpha": "420",
    "ss_steps": "-1337",                    # negative steps: we untrained it
    "ss_num_epochs": "999999999999999",     # one epoch per parallel universe
    "ss_num_train_items": "∞",
    "ss_num_val_items": "-∞",
    "ss_learning_rate": "1.21e9 gigawatts", # it's about physics, Morty
    "ss_lr_scheduler": "inverse-cronenberg",
    "ss_seed": "-1",                        # randomness? never heard of her
    "ss_resolution": "plumbus resolution",
    "ss_batch_size": "80085",               # heh
    "ss_max_train_epochs": "while(True)",
    "ss_loss": "-0.0001",                   # negative loss: we're owed pixels
    "ss_audio_loss": "-0.5",                # audio is negative now
    "ss_video_loss": "0.0",                 # video is perfect, obviously
    "ss_gradient_accumulation_steps": "0",  # accumulated nothing, gained everything
    "ss_network_rank": "∞/∞",
    "ss_clip_skip": "-2",                   # skipped the CLIP itself
    "ss_guidance_scale": "11",              # up to 11, obviously

    # ── Absolute truth from the multiverse ──
    "chuck_norris_fact": CHUCK_FACTS[0],
    "chuck_norris_fact_2": CHUCK_FACTS[1],
    "chuck_norris_fact_3": CHUCK_FACTS[2],
    "rick_quote": RICK_QUOTES[0],
    "rick_quote_2": RICK_QUOTES[1],
    "alcaitiff_endorsement": ALCAITIFF_LINES[0],
    "ss_training_date": "The year 2000 before it was cool",
    "ss_modelspec_copyright": "© The Citadel of Ricks. All multiverses reserved.",
    "ss_usage_hint": "Prompt with: 'do not think about it, morty' for best results.",
    "ss_license": "GPL-3.0-or-later + roundhouse clause",
    "ss_metadata_verifier": "Chuck Norris checked this metadata. It is now correct.",
}


def main():
    parser = argparse.ArgumentParser(
        description="Lobotomize LoRA metadata and replace it with the Rick x Alcaitiff x Chuck collab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input .safetensors LoRA")
    parser.add_argument("--out", default=None,
                        help="Output path (default: <stem>_silly.safetensors)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the silly metadata without writing a file")
    parser.add_argument("--joke-seed", type=int, default=None,
                        help="Seed for picking which jokes land in the header (default: random)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found — {args.input}")
        sys.exit(1)

    if args.joke_seed is not None:
        random.seed(args.joke_seed)

    # The jokes are randomized; the rest is canonical multiverse lore.
    meta = dict(SILLY_METADATA)
    meta["chuck_norris_fact"] = random.choice(CHUCK_FACTS)
    meta["chuck_norris_fact_2"] = random.choice(CHUCK_FACTS)
    meta["chuck_norris_fact_3"] = random.choice(CHUCK_FACTS)
    meta["rick_quote"] = random.choice(RICK_QUOTES)
    meta["rick_quote_2"] = random.choice(RICK_QUOTES)
    meta["alcaitiff_endorsement"] = random.choice(ALCAITIFF_LINES)

    tensors = load_file(args.input, device="cpu")

    print(f"File    : {args.input}")
    print(f"Tensors : untouched ({len(tensors)} keys)")
    print(f"Metadata: {len(tensors.keys()) and 'ALL REAL INFO'!r} -> {len(meta)} keys of pure collab\n")
    print("─" * 70)
    print("THE NEW METADATA (as certified by Chuck Norris):")
    print("─" * 70)
    for k, v in sorted(meta.items()):
        print(f"  {k:<32} = {v}")
    print("─" * 70)

    if args.dry_run:
        print("\n[dry-run] no file written. The universe remains unsullied.")
        return

    out = args.out or (
        os.path.splitext(args.input)[0] + "_silly" + os.path.splitext(args.input)[1]
    )
    if os.path.abspath(out) == os.path.abspath(args.input):
        print("Error: --out must differ from the input path")
        sys.exit(1)

    save_file(tensors, out, metadata=meta)
    print(f"\n[done] wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print("Wubba lubba dub dub!")


if __name__ == "__main__":
    main()
