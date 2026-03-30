import torch
import random
import os
import matplotlib.pyplot as plt

from ai_playground.inference.cache import KVCache, PagedKVCache


# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 4
HEADS = 32
HEAD_DIM = 1024
DTYPE = torch.float32

MAX_STEPS = 10000
LOG_INTERVAL = 100

MAX_LEN = 5120
BLK_SIZE = 64


# KV IMPLEMENTATIONS
class CatKV:
    def __init__(self, *_):
        self.k = None
        self.v = None

    def append(self, k, v):
        if self.k is None or self.v is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)

    def __len__(self):
        return 0 if self.k is None else self.k.shape[2]

    def kv_mem_mb(self):
        if self.k is None or self.v is None:
            return 0
        return (self.k.numel() + self.v.numel()) * self.k.element_size() / 1024**2


class PreallocKVWrapper(KVCache):
    def kv_mem_mb(self):
        return (self.k.numel() + self.v.numel()) * self.k.element_size() / 1024**2


class PagedKVWrapper(PagedKVCache):
    def kv_mem_mb(self):
        total = 0
        for k, v in self.iter_kv():
            total += (k.numel() + v.numel()) * k.element_size()
        return total / 1024**2


KV_IMPLS = {
    "prealloc": PreallocKVWrapper,
    "cat": CatKV,
    "paged": PagedKVWrapper,
}


# STRESS
def fragment():
    junk = []
    for _ in range(30):
        size = random.randint(128, 512)
        junk.append(torch.randn(size, size, device=DEVICE))
    for i in range(0, len(junk), 2):
        junk[i] = None


def temp_alloc(seq_len):
    try:
        x = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=DEVICE)
        del x
    except:
        pass


def get_mem():
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return alloc, reserved, reserved - alloc


def run(name, kv_cls):
    print(f"\nRunning {name}")

    torch.cuda.empty_cache()

    results = {
        "steps": [],
        "kv_mem": [],
        "alloc": [],
        "reserved": [],
        "frag": [],
    }

    step = 0
    try:
        if name == "prealloc":
            kv = kv_cls(BATCH, HEADS, HEAD_DIM, MAX_LEN, DEVICE, DTYPE)
        elif name == "paged":
            kv = kv_cls(BATCH, HEADS, HEAD_DIM, BLK_SIZE, DEVICE, DTYPE)
        else:
            kv = kv_cls(BATCH, HEADS, HEAD_DIM)
        for step in range(1, MAX_STEPS + 1):
            k = torch.randn(BATCH, HEADS, 1, HEAD_DIM, device=DEVICE)
            v = torch.randn(BATCH, HEADS, 1, HEAD_DIM, device=DEVICE)

            kv.append(k, v)
            seq_len = len(kv)

            if step % 20 == 0:
                fragment()
            if step % 10 == 0:
                temp_alloc(seq_len)

            if step % LOG_INTERVAL == 0:
                alloc, reserved, frag = get_mem()
                kv_mem = kv.kv_mem_mb()

                results["steps"].append(step)
                results["kv_mem"].append(kv_mem)
                results["alloc"].append(alloc)
                results["reserved"].append(reserved)
                results["frag"].append(frag)

                print(f"{name} step {step}")

    except RuntimeError:
        print(f"{name} stopped at step {step}")

    return results


# PLOT
def plot(all_results):
    os.makedirs("plots", exist_ok=True)

    # Fragmentation
    plt.figure()
    for name, res in all_results.items():
        plt.plot(res["steps"], res["frag"], label=name)
    plt.xlabel("Steps")
    plt.ylabel("Fragmentation (MB)")
    plt.legend()
    plt.title("Fragmentation vs Steps")
    plt.savefig("plots/frag.png")

    # Efficiency
    plt.figure()
    for name, res in all_results.items():
        eff = [kv / r if r > 0 else 0 for kv, r in zip(res["kv_mem"], res["reserved"])]
        plt.plot(res["steps"], eff, label=name)
    plt.xlabel("Steps")
    plt.ylabel("Efficiency (kv_mem / reserved)")
    plt.legend()
    plt.title("Memory Efficiency")
    plt.savefig("plots/efficiency.png")

    print("Saved plots in /plots")


if __name__ == "__main__":
    all_results = {}

    for name, cls in KV_IMPLS.items():
        all_results[name] = run(name, cls)

    plot(all_results)
