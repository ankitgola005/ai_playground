# Experiment 6: KV Cache Allocation Strategies under Fragmentation

## Experimental Setup

| Component  | Details                                                         |
| ---------- | --------------------------------------------------------------- |
| Simulation | Autoregressive KV cache growth (token-by-token)                 |
| Data       | Synthetic KV tensors                                            |
| Strategies | Naive (`torch.cat`), Chunked (doubling), Preallocated, Paged KV |
| Stressors  | Memory prefill, transient attention-like allocations            |

> **Objective:** Evaluate how different KV cache allocation strategies behave under **GPU memory pressure and fragmentation**, focusing on **failure modes, allocator behavior, and stability over time**.

---

## Steps to Reproduce

From the experiment folder:

```bash
python -u kv_fragmentation.py
```

**Key parameters:**

- heads = 32
- head_dim = 1024
- max steps = 100000

**Additional stress applied:**

- GPU memory prefilled
- Large temporary tensors: attention-like simulation

## Results

| Allocation Strategy       | Max Steps Run | Available Memory at Failed Step |
| ------------------------- | ------------- | ------------------------------- |
| Naive (torch.cat)         | 22,610        | 2.76 GiB                        |
| Chunked (doubling buffer) | 16,385        | 3.53 GiB                        |
| Preallocated              | 0             | 5.53 GiB                        |
| Paged KV                  | 44,801        | 2.38 MiB                        |

### Observations

**Fragmentation vs Capacity**

- Naive fails despite available memory, indicating fragmentation-driven failure. OOM occurs even when large total memory is free, due to lack of contiguous blocks.
- Chunked Growth is Surprisingly Worse. Chunked fails earlier than naive. Requires large reallocations during doubling steps. These allocations are harder to satisfy under fragmentation.
- Prealloc fails immediately. Even with sufficient total memory, large upfront allocation is impossible post-fragmentation. Preallocation only works if done before fragmentation occurs
- Paged KV is Significantly More Stable. Failure occurs only at true capacity limit, free memory at time of failure is in MiB.
- KV cache performance is governed by allocator dynamics, not just memory size.
