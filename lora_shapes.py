# lora_shapes.py
# Run it like: python lora_shapes.py your_lora_file.safetensors [--out report.txt]

import argparse
from safetensors.torch import load_file

parser = argparse.ArgumentParser(description="Print LoRA tensor shapes")
parser.add_argument("input", help="Input .safetensors file")
parser.add_argument("--out", default=None, help="Output text file")
args = parser.parse_args()

lora_path = args.input

try:
    state_dict = load_file(lora_path, device='cpu')
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

lines = []

def emit(line=""):
    print(line)
    lines.append(line)

emit(f"\nLoRA file: {lora_path}")
emit(f"Total parameters: {len(state_dict)}\n")

# Group by block number for easier reading
from collections import defaultdict
blocks = defaultdict(list)

for key, tensor in state_dict.items():
    shape_str = f"{tensor.shape}  ({tensor.numel():>10,} elements)"
    blocks[key.split('.')[2]].append((key, shape_str))   # split('.')[2] = block number

# Print in a clean, sorted way
for block_num in sorted(blocks.keys(), key=int):
    emit(f"double_blocks.{block_num}:")
    for full_key, shape_info in sorted(blocks[block_num]):
        emit(f"  {full_key:.<65} {shape_info}")
    emit()

# Also print single_blocks if they exist
single_blocks = defaultdict(list)
for key, tensor in state_dict.items():
    if 'single_blocks' in key:
        block_num = key.split('.')[2]
        shape_str = f"{tensor.shape}  ({tensor.numel():>10,} elements)"
        single_blocks[block_num].append((key, shape_str))

if single_blocks:
    emit("\nSingle blocks:")
    for block_num in sorted(single_blocks.keys(), key=int):
        emit(f"single_blocks.{block_num}:")
        for full_key, shape_info in sorted(single_blocks[block_num]):
            emit(f"  {full_key:.<65} {shape_info}")
        emit()

# Quick summary of problematic qkv ones
emit("\nQuick check - qkv weights:")
for key in state_dict:
    if 'qkv' in key:
        tensor = state_dict[key]
        emit(f"{key:.<70} shape = {tensor.shape}   elements = {tensor.numel():,}")

if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\nSaved report: {args.out}")
