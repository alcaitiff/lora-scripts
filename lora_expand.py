import argparse
import os
import torch
from safetensors.torch import load_file, save_file

def parse_args():
    parser = argparse.ArgumentParser(description="Expand broken qkv.lora_B weights")
    parser.add_argument("input", help="Input .safetensors file")
    parser.add_argument("--out", default=None, help="Output .safetensors file")
    return parser.parse_args()

def main():
    args = parse_args()

    inp = args.input
    if not os.path.isfile(inp):
        print(f"File not found: {inp}")
        raise SystemExit(1)

    base, ext = os.path.splitext(inp)
    if not ext:
        ext = ".safetensors"
    out = args.out or f"{base}_expanded{ext}"

    sd = load_file(inp)

    for k in list(sd.keys()):
        if "qkv.lora_B.weight" in k:
            B = sd[k]

            if B.shape[0] == 4096:  # broken ones
                sd[k] = torch.cat([B, B, B], dim=0)
                print("Fixed:", k, "->", sd[k].shape)

    save_file(sd, out)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
