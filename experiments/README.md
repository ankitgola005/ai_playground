
# Experiments Overview

This folder contains all experiment results and scripts.  
Each experiment is self-contained in its own folder with its own README, scripts, assets, and config.

---

|S.No |Experiment | Description |Key Observations| Config | Link |
|------|----------|-------------|---|-----|------|
| 01|Scaling | Study how model width (n_embed), and depth (hidden_dim), affect validation loss. | wider and deeper models reduce validation loss but with diminishing returns and higher compute cost. | [`gpt_config.yaml`](../configs/gpt_config.yaml) | [Read more](scaling/README.md) |

---

> New experiments should follow the same structure: each in its own folder, with an internal README summarizing results, links to plots/scripts, and reference to the base config file.