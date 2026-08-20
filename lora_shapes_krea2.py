# lora_shapes_krea2.py
# LoRA shape inspector for KREA2 / Flux-style key structures.
# Run it like: python lora_shapes_krea2.py your_lora_file.safetensors [--out report.txt]

import argparse
import re
import sys
from collections import defaultdict

try:
    from safetensors.torch import load_file
except ImportError:
    print("Error: safetensors not installed. Run: pip install safetensors")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Print LoRA tensor shapes (KREA2 / Flux style)")
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


def format_tensor_info(key, tensor):
    """Format shape information, including the actual contents of alpha tensors."""
    info = f"{list(tensor.shape)}  ({tensor.numel():>10,} elements)"
    if key.endswith((".alpha", ".lora_alpha")):
        dtype = str(tensor.dtype).replace("torch.", "")
        if tensor.numel() == 0:
            value = "<EMPTY>"
        elif tensor.numel() == 1:
            value = repr(tensor.item())
        else:
            values = tensor.detach().flatten()[:8].tolist()
            suffix = "..." if tensor.numel() > 8 else ""
            value = f"{values}{suffix}"
        info += f"  dtype={dtype}  value={value}"
    return info

emit(f"LoRA file: {lora_path}")
emit(f"Total parameters: {len(state_dict)}")
emit()

# ---------------------------------------------------------------------------
# Group keys by section prefix + block number.
# We regex: ^(.*?)\.(\d+)\.(.+)$
#   group(1) = section prefix  (e.g. "diffusion_model.blocks")
#   group(2) = block number    (e.g. "0", "27")
#   group(3) = rest            (e.g. "attn.wq.lora_A.weight")
# ---------------------------------------------------------------------------
pattern = re.compile(r'^(.*?)\.(\d+)\.(.+)$')

sections = defaultdict(lambda: defaultdict(list))  # sections[prefix][block_num] -> [(full_key, shape_str)]

for key, tensor in state_dict.items():
    m = pattern.match(key)
    if not m:
        # Keys that don't match the pattern (e.g. non-block keys)
        shape_str = format_tensor_info(key, tensor)
        sections["__unmatched__"]["__none__"].append((key, shape_str))
        continue
    prefix = m.group(1)
    block_num = m.group(2)
    rest = m.group(3)
    shape_str = format_tensor_info(key, tensor)
    sections[prefix][block_num].append((key, shape_str))

# ---------------------------------------------------------------------------
# Print each section in order, sorted by block number (numeric).
# ---------------------------------------------------------------------------
sort_key_sections = sorted(sections.keys())  # alphabetical by section prefix

for section in sort_key_sections:
    if section == "__unmatched__":
        continue  # print at end

    blocks_dict = sections[section]
    # Compute number of blocks for header
    num_blocks = len(blocks_dict)
    short_name = section.split('.')[-1] if '.' in section else section
    emit(f"{short_name} ({num_blocks} blocks):")

    for block_num in sorted(blocks_dict.keys(), key=lambda x: int(x) if x.isdigit() else x):
        emit(f"  {section}.{block_num}:")
        for full_key, shape_info in sorted(blocks_dict[block_num]):
            emit(f"    {full_key:.<70} {shape_info}")
        emit()

# ---------------------------------------------------------------------------
# Print any keys that didn't match the pattern
# ---------------------------------------------------------------------------
if "__unmatched__" in sections and sections["__unmatched__"]["__none__"]:
    emit("\nOther keys (no block number):")
    for full_key, shape_info in sorted(sections["__unmatched__"]["__none__"]):
        emit(f"  {full_key:.<70} {shape_info}")
    emit()

# ---------------------------------------------------------------------------
# Print total element count per LoRA rank to verify rank
# ---------------------------------------------------------------------------
emit("Summary:")
total_elements = sum(t.numel() for t in state_dict.values())
emit(f"  Total elements: {total_elements:,}")

# Detect rank from lora_A weights (assumes all A matrices have same rank)
lora_ranks = set()
for key, tensor in state_dict.items():
    if 'lora_A' in key and tensor.ndim >= 2:
        lora_ranks.add(tensor.shape[0])  # rank is first dim for lora_A
if lora_ranks:
    ranks_str = ", ".join(str(r) for r in sorted(lora_ranks))
    emit(f"  LoRA rank(s): {ranks_str}")

# Modules affected
module_types = set()
for key in state_dict:
    if 'attn' in key:
        module_types.add('attn')
    if 'mlp' in key:
        module_types.add('mlp')
emit(f"  Modules: {', '.join(sorted(module_types))}")

if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\nSaved report: {args.out}")
