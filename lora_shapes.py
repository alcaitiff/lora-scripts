# lora_shapes.py
# Run it like: python lora_shapes.py your_lora_file.safetensors

import sys
from safetensors.torch import load_file

if len(sys.argv) != 2:
    print("Usage: python lora_shapes.py <path_to_lora.safetensors>")
    sys.exit(1)

lora_path = sys.argv[1]

try:
    state_dict = load_file(lora_path, device='cpu')
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

print(f"\nLoRA file: {lora_path}")
print(f"Total parameters: {len(state_dict)}\n")

# Group by block number for easier reading
from collections import defaultdict
blocks = defaultdict(list)

for key, tensor in state_dict.items():
    shape_str = f"{tensor.shape}  ({tensor.numel():>10,} elements)"
    blocks[key.split('.')[2]].append((key, shape_str))   # split('.')[2] = block number

# Print in a clean, sorted way
for block_num in sorted(blocks.keys(), key=int):
    print(f"double_blocks.{block_num}:")
    for full_key, shape_info in sorted(blocks[block_num]):
        print(f"  {full_key:.<65} {shape_info}")
    print()

# Also print single_blocks if they exist
single_blocks = defaultdict(list)
for key, tensor in state_dict.items():
    if 'single_blocks' in key:
        block_num = key.split('.')[2]
        shape_str = f"{tensor.shape}  ({tensor.numel():>10,} elements)"
        single_blocks[block_num].append((key, shape_str))

if single_blocks:
    print("\nSingle blocks:")
    for block_num in sorted(single_blocks.keys(), key=int):
        print(f"single_blocks.{block_num}:")
        for full_key, shape_info in sorted(single_blocks[block_num]):
            print(f"  {full_key:.<65} {shape_info}")
        print()

# Quick summary of problematic qkv ones
print("\nQuick check - qkv weights:")
for key in state_dict:
    if 'qkv' in key:
        tensor = state_dict[key]
        print(f"{key:.<70} shape = {tensor.shape}   elements = {tensor.numel():,}")
