import argparse
import os
from safetensors.torch import load_file, save_file

def parse_args():
    parser = argparse.ArgumentParser(description="Convert modeldiff-style LoRA keys")
    parser.add_argument("input", help="Input .safetensors file")
    parser.add_argument("--out", default=None, help="Output .safetensors file")
    return parser.parse_args()

def convert_key(k):
    if not k.startswith("lora_unet_"):
        return None

    k = k.replace("lora_unet_", "diffusion_model.", 1)

    if ".lora_down.weight" in k:
        k = k.replace(".lora_down.weight", ".lora_A.weight")
    elif ".lora_up.weight" in k:
        k = k.replace(".lora_up.weight", ".lora_B.weight")

    parts = k.split(".")
    path = parts[1]

    path = path.replace("double_blocks_", "double_blocks.")
    path = path.replace("single_blocks_", "single_blocks.")
    path = path.replace("_img_", ".img_")
    path = path.replace("_txt_", ".txt_")
    path = path.replace("_linear", ".linear")
    path = path.replace("_proj", ".proj")
    path = path.replace("_qkv", ".qkv")
    path = path.replace("_mlp_", ".mlp.")

    return "diffusion_model." + path + "." + parts[-2] + "." + parts[-1]

def main():
    args = parse_args()

    input_file = args.input
    if not os.path.exists(input_file):
        print("File not found:", input_file)
        raise SystemExit(1)

    output_file = args.out
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + "_converted.safetensors"

    sd = load_file(input_file)
    new_sd = {}

    for k, v in sd.items():
        new_k = convert_key(k)
        if new_k:
            new_sd[new_k] = v

    save_file(new_sd, output_file)

    print(f"Converted {len(new_sd)} keys")
    print("Saved to:", output_file)

if __name__ == "__main__":
    main()
