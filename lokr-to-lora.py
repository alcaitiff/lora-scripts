import torch
from safetensors import safe_open
from safetensors.torch import save_file
import argparse
import os

def convert_lokr_to_lora(lokr_path, output_path=None, rank=16, device=None):
    """
    Converts LoKr → standard LoRA (Flux-compatible).
    Casts to float32 for SVD, then back to bfloat16 for output.
    """
    if not os.path.isfile(lokr_path):
        raise FileNotFoundError(f"LoKr file not found: {lokr_path}")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if output_path is None:
        base, ext = os.path.splitext(lokr_path)
        output_path = f"{base}_converted_lora{ext}"

    print(f"Input:  {lokr_path}")
    print(f"Output: {output_path}")
    print(f"Target rank: {rank}\n")

    # Load state dict (safetensors loads as original dtype, usually bfloat16)
    state = {}
    with safe_open(lokr_path, framework="pt", device=device) as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)

    base_keys = set()
    for k in state.keys():
        if k.endswith(('.alpha', '.lokr_w1', '.lokr_w2')):
            base = '.'.join(k.split('.')[:-1])
            base_keys.add(base)

    new_state = {}
    converted_count = 0

    for base in sorted(base_keys):
        alpha_key = base + '.alpha'
        w1_key   = base + '.lokr_w1'
        w2_key   = base + '.lokr_w2'

        if not all(k in state for k in [alpha_key, w1_key, w2_key]):
            print(f"  Skipping incomplete: {base}")
            continue

        alpha = state[alpha_key].item()
        w1    = state[w1_key].to(device)   # bfloat16
        w2    = state[w2_key].to(device)   # bfloat16

        try:
            kron_product = torch.kron(w1, w2)          # bfloat16
            delta = alpha * kron_product               # bfloat16

            # Critical fix: cast to float32 for SVD compatibility
            delta_float = delta.to(torch.float32)

            # Low-rank SVD (more memory-friendly than full svd)
            U, S, Vh = torch.svd_lowrank(delta_float, q=rank)

            # Truncate
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]

            S_sqrt = torch.sqrt(S + 1e-8)

            # LoRA convention: delta ≈ B @ A  (B: out×rank, A: rank×in)
            A = Vh.t() * S_sqrt.unsqueeze(1)          # rank × in
            B = U * S_sqrt.unsqueeze(0)               # out × rank

            # Cast back to bfloat16 for Flux compatibility & smaller file
            A = A.to(torch.bfloat16).contiguous()
            B = B.to(torch.bfloat16).contiguous()

            new_state[base + '.lora_A.weight'] = A.cpu()
            new_state[base + '.lora_B.weight'] = B.cpu()

            converted_count += 1
            print(f"  Converted {base:60}  shapes: A={A.shape}  B={B.shape}  (bf16)")

        except RuntimeError as e:
            print(f"  Failed for {base}: {e}")
            continue
        except Exception as e:
            print(f"  Unexpected error for {base}: {e}")
            continue

    if converted_count == 0:
        print("No layers converted. Verify LoKr keys or try --device cpu if OOM.")
        return

    save_file(new_state, output_path)
    print(f"\nSuccess! Converted {converted_count} groups.")
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoKr → LoRA converter (Flux fix)")
    parser.add_argument("lokr_file", type=str, help="Input LoKr .safetensors")
    parser.add_argument("--output", "--out", type=str, default=None, help="Output path (alias: --out)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda'])

    args = parser.parse_args()

    try:
        convert_lokr_to_lora(
            lokr_path=args.lokr_file,
            output_path=args.output,
            rank=args.rank,
            device=args.device
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
