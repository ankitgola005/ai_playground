
# Experiments Overview

This folder contains all experiment results and scripts.  
Each experiment is self-contained in its own folder with its own README, scripts, assets, and config.

---

| Experiment | Description | Config | Link |
|------------|-------------|--------|------|
| 01 — Scaling Laws | Study how model width, depth, and other parameters affect training and validation loss. Key observation: wider and deeper models reduce validation loss but with diminishing returns and higher compute cost. | [`config.yaml`](../configs/base_config.yaml) | [Read more](01_scaling_laws/README.md) |

---

> New experiments should follow the same structure: each in its own folder, with an internal README summarizing results, links to plots/scripts, and reference to the base config file.