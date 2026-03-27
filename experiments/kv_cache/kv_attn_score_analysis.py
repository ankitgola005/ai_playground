# kv_attn_decode_vs_prefill_updated.py
import torch
from typing import List

from ai_playground.models.attention import MultiHeadAttention
from ai_playground.inference.cache.kv_cache import KVCache


# -----------------------------
# Helper: initialize random inputs
# -----------------------------
def generate_inputs(B: int, T: int, C: int, device="cuda", dtype=torch.float32):
    return torch.randn(B, T, C, device=device, dtype=dtype)


# -----------------------------
# Benchmark function
# -----------------------------
def benchmark_decode_vs_prefill(
    attn: MultiHeadAttention,
    seq_lens: List[int],
    use_cache: bool = True,
    device="cuda",
    dtype=torch.float32,
):
    results = {}

    for T in seq_lens:
        B, C = 1, attn.embed_dim
        x = generate_inputs(B, T, C, device=device, dtype=dtype)

        # -----------------------------
        # Prefill: full sequence at once
        # -----------------------------
        prefill_cache = (
            KVCache(
                B=B,
                H=attn.n_kv_head,
                head_dim=attn.head_dim,
                max_len=T,
                device=device,
                dtype=dtype,
            )
            if use_cache
            else None
        )
        with torch.no_grad():
            prefill_out, _ = attn(x, past_key_value=prefill_cache, use_cache=use_cache)

        # -----------------------------
        # Decode: token by token
        # -----------------------------
        decode_cache = (
            KVCache(
                B=B,
                H=attn.n_kv_head,
                head_dim=attn.head_dim,
                max_len=T,
                device=device,
                dtype=dtype,
            )
            if use_cache
            else None
        )
        decode_out_tokens = []
        for t in range(T):
            x_t = x[:, t : t + 1, :]  # single token
            out_t, _ = attn(x_t, past_key_value=decode_cache, use_cache=use_cache)
            decode_out_tokens.append(out_t)

        decode_out = torch.cat(decode_out_tokens, dim=1)

        # -----------------------------
        # Compare prefill vs decode
        # -----------------------------
        diff = (prefill_out - decode_out).abs().max().item()
        print(f"Sequence length {T}: max diff decode vs prefill = {diff:.6f}")
        assert diff < 1e-4, f"Decode mismatch! Max diff: {diff:.6f}"

        results[T] = diff

    return results


# -----------------------------
# Main
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Example MultiHeadAttention initialization
    embed_dim = 32
    n_head = 4
    n_kv_head = 2
    block_size = 64

    attn = MultiHeadAttention(
        embed_dim=embed_dim,
        n_head=n_head,
        n_kv_head=n_kv_head,
        block_size=block_size,
        use_flash_attention=False,
        attn_droupout=0.0,
        residual_droupout=0.0,
    ).to(device=device, dtype=dtype)

    seq_lens = [1, 2, 4, 8, 16, 32, 64]  # example sequence lengths
    results = benchmark_decode_vs_prefill(
        attn, seq_lens, use_cache=True, device=device, dtype=dtype
    )
    print("All decode vs prefill checks passed!")


if __name__ == "__main__":
    main()
