import torch
import time
import random

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

N_HEADS = 32
HEAD_DIM = 1024
MAX_STEPS = 100000  # tokens to simulate
PRINT_EVERY = 5000


# Utils
def print_mem(step, tag=""):
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag} step {step}] alloc={alloc:.0f}MB reserved={reserved:.0f}MB")


def new_kv_block(seq_len=1):
    return torch.zeros(1, N_HEADS, seq_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)


def prefill_memory():
    blocks = []
    try:
        blocks.append(torch.randn(2048, 2048, device=DEVICE, dtype=DTYPE))
    except RuntimeError:
        pass
    return blocks


def inject_fragmentation_noise():
    junk = []
    for _ in range(10):
        size = random.randint(1024, 1024)
        junk.append(torch.randn(size, size, device=DEVICE))

    # randomly free some
    for i in range(len(junk)):
        if random.random() < 0.5:
            junk[i] = None


# Strategy 1: Naive (torch.cat)
def run_naive():
    print("\n=== NAIVE (torch.cat growth) ===")
    torch.cuda.empty_cache()
    prefill_memory()
    kv = None

    for step in range(1, MAX_STEPS + 1):
        if step % 50 == 0:
            inject_fragmentation_noise()
        try:
            new = new_kv_block(1)

            if kv is None:
                kv = new
            else:
                kv = torch.cat([kv, new], dim=2)

        except RuntimeError as e:
            print(f"OOM at step {step}")
            print(str(e))
            return step

        if step % PRINT_EVERY == 0:
            print_mem(step, "naive")

    print("Completed without OOM")
    return MAX_STEPS


# Strategy 2: Chunked doubling
def run_chunked():
    print("\n=== CHUNKED (doubling buffer) ===")
    torch.cuda.empty_cache()
    prefill_memory()

    capacity = 1
    kv = new_kv_block(capacity)
    length = 0

    for step in range(1, MAX_STEPS + 1):
        if step % 50 == 0:
            inject_fragmentation_noise()
        try:
            if length >= capacity:
                # grow buffer
                new_capacity = capacity * 2
                new_kv = new_kv_block(new_capacity)

                new_kv[:, :, :capacity, :] = kv
                kv = new_kv
                capacity = new_capacity

            # write new token
            kv[:, :, length : length + 1, :] = new_kv_block(1)
            length += 1

        except RuntimeError as e:
            print(f"OOM at step {step}")
            print(str(e))
            return step

        if step % PRINT_EVERY == 0:
            print_mem(step, "chunked")

    print("Completed without OOM")
    return MAX_STEPS


# Strategy 3: Preallocated
def run_prealloc():
    print("\n=== PREALLOCATED ===")
    torch.cuda.empty_cache()
    prefill_memory()

    try:
        kv = new_kv_block(MAX_STEPS)
    except RuntimeError as e:
        print("Failed to preallocate upfront")
        print(str(e))
        return 0

    for step in range(1, MAX_STEPS + 1):
        if step % 50 == 0:
            inject_fragmentation_noise()
        try:
            kv[:, :, step - 1 : step, :] = new_kv_block(1)

        except RuntimeError as e:
            print(f"OOM at step {step}")
            print(str(e))
            return step

        if step % PRINT_EVERY == 0:
            print_mem(step, "prealloc")

    print("Completed without OOM")
    return MAX_STEPS


# Strategy 4: Paged KV (list of blocks)
def run_paged(page_size=128):
    print(f"\n=== PAGED (page_size={page_size}) ===")
    torch.cuda.empty_cache()
    prefill_memory()

    pages = []
    current_page = None
    offset = 0

    for step in range(1, MAX_STEPS + 1):
        if step % 50 == 0:
            inject_fragmentation_noise()
        try:
            if current_page is None or offset >= page_size:
                current_page = new_kv_block(page_size)
                pages.append(current_page)
                offset = 0

            current_page[:, :, offset : offset + 1, :] = new_kv_block(1)
            offset += 1

        except RuntimeError as e:
            print(f"OOM at step {step}")
            print(str(e))
            return step

        if step % PRINT_EVERY == 0:
            print_mem(step, "paged")

    print("Completed without OOM")
    return MAX_STEPS


def main():
    if DEVICE == "cpu":
        print("Run this on GPU for meaningful results.")
        return

    print(f"Running on {DEVICE}")

    results = {}

    results["naive"] = run_naive()
    time.sleep(2)

    results["chunked"] = run_chunked()
    time.sleep(2)

    results["prealloc"] = run_prealloc()
    time.sleep(2)

    results["paged"] = run_paged(page_size=128)

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k:10s} → survived {v} steps")


if __name__ == "__main__":
    main()
