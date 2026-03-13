import torch
from safetensors.torch import load_file, save_file

inp = "Klein9B-Mystic2Realism.safetensors"
out = "Klein9B-Mystic2RealismExpanded.safetensors"

sd = load_file(inp)

for k in list(sd.keys()):
    if "qkv.lora_B.weight" in k:
        B = sd[k]

        if B.shape[0] == 4096:  # broken ones
            sd[k] = torch.cat([B, B, B], dim=0)
            print("Fixed:", k, "->", sd[k].shape)

save_file(sd, out)
