import argparse
import os
import torch
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser(description="Reduce LoRA rank using GPU SVD")
parser.add_argument("input", help="Input LoRA safetensors file")
parser.add_argument("output", nargs="?", default=None, help="Output LoRA safetensors file")
parser.add_argument("--out", default=None, help="Output LoRA safetensors file")
parser.add_argument("--rank", type=int, required=True, help="Target rank")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Target rank:", args.rank)

def reduce_lora(A, B, rank):

    orig_dtype = A.dtype

    # move to GPU and convert to float32 for SVD
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)

    W = B @ A

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    sqrtS = torch.sqrt(S)

    B_new = U @ torch.diag(sqrtS)
    A_new = torch.diag(sqrtS) @ Vh

    # convert back to original dtype and CPU
    return A_new.to(orig_dtype).cpu(), B_new.to(orig_dtype).cpu()


output_path = args.out or args.output
if output_path is None:
    base, ext = os.path.splitext(args.input)
    if not ext:
        ext = ".safetensors"
    output_path = f"{base}_rank{args.rank}{ext}"

sd = load_file(args.input)
new_sd = {}

processed = set()

for k in sd.keys():

    if ".lora_A.weight" in k:

        base = k.replace(".lora_A.weight", "")

        A = sd[k]
        B = sd[base + ".lora_B.weight"]

        print("Reducing:", base)

        A_new, B_new = reduce_lora(A, B, args.rank)

        new_sd[base + ".lora_A.weight"] = A_new
        new_sd[base + ".lora_B.weight"] = B_new

        processed.add(base)

    elif ".lora_B.weight" in k:
        base = k.replace(".lora_B.weight", "")
        if base not in processed:
            continue
    else:
        new_sd[k] = sd[k]

save_file(new_sd, output_path)

print("Saved reduced LoRA:", output_path)
