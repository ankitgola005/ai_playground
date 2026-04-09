import copy
import json
import os
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt

from ai_playground.utils import build_data_pipeline, build_model, get_strategy, set_seed
from ai_playground.utils.config import get_config
from ai_playground.trainer import Trainer


# ------------------------
# Save / Load
# ------------------------
def save_progress(path, data):
    dir_name = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = tmp.name

    os.replace(tmp_path, path)


def load_progress(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ------------------------
# Configs
# ------------------------
def get_dense_config(base_config, run_id):
    config = copy.deepcopy(base_config)
    config.model.model_kwargs["num_experts"] = 0
    config.trainer.run_name = f"dense_run_{run_id}"
    return config


def get_moe_config(base_config, run_id):
    config = copy.deepcopy(base_config)
    config.model.model_kwargs["num_experts"] = 4
    config.trainer.run_name = f"moe_run_{run_id}"
    return config


# ------------------------
# Single Run
# ------------------------
def run_once(config, run_id):
    tokenizer, train_loader, val_loader = build_data_pipeline(
        config.data, config.trainer.batch_size, config.trainer.seed + run_id
    )

    model = build_model(config.model)(
        config.model, tokenizer.vocab_size, config.data.block_size, True
    )

    trainer = Trainer(
        config,
        strategy=get_strategy(config.distributed),
        logger_metrics=["moe"],
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, train_loader, val_loader)

    max_mem = torch.cuda.max_memory_allocated() / (1024**2)

    history = trainer.history
    train_loss = [x["train_loss"] for x in history if "train_loss" in x]
    val_loss = [x["val_loss"] for x in history if "val_loss" in x]
    tps = [x["tps"] for x in history if "tps" in x]

    return {
        "run_id": run_id,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "tokens_per_sec": tps,
        "max_mem": max_mem,
        "final_val_loss": val_loss[-1] if len(val_loss) > 0 else None,
        "history": history,
    }


# ------------------------
# Experiment Runner
# ------------------------
def run_experiment(base_config, num_runs=2, save_path="progress.json"):
    assert num_runs % 2 == 0, "num_runs should be even"

    # ---- load state ----
    state = load_progress(save_path)
    if state is None:
        state = {
            "dense_results": [],
            "moe_results": [],
            "completed": [],
        }

    dense_results = state["dense_results"]
    moe_results = state["moe_results"]
    completed = set(tuple(x) for x in state["completed"])

    # ---- deterministic order ----
    half = num_runs // 2
    orders = ["dense_first"] * half + ["moe_first"] * half
    rng = np.random.RandomState(base_config.trainer.seed)
    rng.shuffle(orders)

    for i, order in enumerate(orders):
        print(f"\n=== RUN {i} ({order}) ===")

        dense_config = get_dense_config(base_config, i)
        moe_config = get_moe_config(base_config, i)

        def run_and_save(run_type, config):
            if (i, run_type) in completed:
                print(f"Skipping {run_type} run {i}")
                return

            result = run_once(config, i)

            if run_type == "dense":
                dense_results.append(result)
            else:
                moe_results.append(result)

            completed.add((i, run_type))

            save_progress(
                save_path,
                {
                    "dense_results": dense_results,
                    "moe_results": moe_results,
                    "completed": list(completed),
                },
            )

        if order == "dense_first":
            run_and_save("dense", dense_config)
            run_and_save("moe", moe_config)
        else:
            run_and_save("moe", moe_config)
            run_and_save("dense", dense_config)

    return dense_results, moe_results


# ------------------------
# Utils
# ------------------------
def moving_avg(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def aggregate(results):
    def safe_list(key):
        return [r[key] for r in results if r.get(key) is not None]

    final_losses = safe_list("final_val_loss")

    tps_per_run = []
    for r in results:
        if len(r["tokens_per_sec"]) > 0:
            tps = r["tokens_per_sec"]
            stable = tps[len(tps) // 3 :]
            tps_per_run.append(np.mean(stable))

    mem = safe_list("max_mem")

    def safe_mean(x):
        return float(np.mean(x)) if len(x) > 0 else None

    def safe_std(x):
        return float(np.std(x)) if len(x) > 0 else None

    return {
        "loss_mean": safe_mean(final_losses),
        "loss_std": safe_std(final_losses),
        "tps_mean": safe_mean(tps_per_run),
        "tps_std": safe_std(tps_per_run),
        "mem_mean": safe_mean(mem),
    }


# ------------------------
# Plotting
# ------------------------
def plot_loss(dense_results, moe_results):
    plt.figure(figsize=(8, 5))

    def avg_curve(results, key):
        curves = [r[key] for r in results if len(r[key]) > 0]
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        return np.mean(curves, axis=0)

    # train loss
    dense_train = avg_curve(dense_results, "train_loss")
    moe_train = avg_curve(moe_results, "train_loss")

    # val loss
    dense_val = avg_curve(dense_results, "val_loss")
    moe_val = avg_curve(moe_results, "val_loss")

    train_x = np.arange(len(dense_train)) * 10
    val_x = np.arange(len(dense_val)) * 100

    # --- Train plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_x, dense_train, label="Dense", linewidth=2)
    plt.plot(train_x, moe_train, label="MoE", linewidth=2, linestyle="--")

    plt.title("Train Loss (Mean across runs)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_loss_curve.png")
    plt.close()

    # --- Val plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(val_x, dense_val, label="Dense", linewidth=2)
    plt.plot(val_x, moe_val, label="MoE", linewidth=2, linestyle="--")

    plt.title("Validation Loss (Mean across runs)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_loss_curve.png")
    plt.close()


def plot_bar(dense_stats, moe_stats):
    labels = ["Val Loss", "Tokens/sec", "Memory (MB)"]

    # Dense baseline = 0%
    dense_vals = [0.0, 0.0, 0.0]

    # Relative % change
    moe_vals = [
        (moe_stats["loss_mean"] - dense_stats["loss_mean"]) / dense_stats["loss_mean"],
        (moe_stats["tps_mean"] - dense_stats["tps_mean"]) / dense_stats["tps_mean"],
        (moe_stats["mem_mean"] - dense_stats["mem_mean"]) / dense_stats["mem_mean"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, dense_vals, width, label="Dense")
    plt.bar(x + width / 2, moe_vals, width, label="MoE")

    plt.xticks(x, labels)

    # show as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))

    plt.axhline(0, linewidth=1)  # baseline reference

    plt.legend()
    plt.title("Relative Comparison (% change vs Dense)")

    plt.savefig("final_metrics.png")
    plt.close()


# ------------------------
# Main
# ------------------------
def main():
    base_config = get_config("minigpt_config.yaml")
    set_seed(base_config.trainer.seed)

    # ---- Warmup ----
    warmup_config = copy.deepcopy(base_config)
    warmup_config.trainer.max_steps = 100
    warmup_config.trainer.val_interval = 0
    warmup_config.trainer.log_interval = 0

    run_experiment(warmup_config, num_runs=2, save_path="warmup_progress.json")

    # ---- Main Experiment ----
    dense_results, moe_results = run_experiment(
        base_config, num_runs=2, save_path="experiment_progress.json"
    )

    dense_stats = aggregate(dense_results)
    moe_stats = aggregate(moe_results)

    print("\n=== FINAL RESULTS ===")
    print("Dense:", dense_stats)
    print("MoE:", moe_stats)

    with open("results.json", "w") as f:
        json.dump(
            {
                "config": base_config.model.model_kwargs,
                "dense": dense_results,
                "moe": moe_results,
                "dense_stats": dense_stats,
                "moe_stats": moe_stats,
            },
            f,
            indent=2,
        )

    plot_loss(dense_results, moe_results)
    plot_bar(dense_stats, moe_stats)


if __name__ == "__main__":
    main()
