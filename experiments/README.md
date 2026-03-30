# Experiments Overview

This folder contains all experiment results and scripts.  
Each experiment is self-contained in its own folder with its own README, scripts, assets, and config.

---

| S.No | Experiment              | Description                                                                           | Key Observations                                                                                                                                                                                                  | Link                                           |
| ---- | ----------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| 01   | Scaling                 | Study how model width (n_embed), and depth (hidden_dim), affect validation loss.      | Wider and deeper models reduce validation loss but with diminishing returns and higher compute cost.                                                                                                              | [Read more](scaling/README.md)                 |
| 02   | Attention Visualization | Visualize attention maps across transformer layers and heads for a sample sequence.   | Early layers show strong diagonal/local attention while deeper layers distribute attention across broader context.                                                                                                | [Read more](attention_visualization/README.md) |
| 03   | Context Length Scaling  | Study how increasing transformer context length (block_size) affects validation loss. | Larger context lengths generally improve performance up to a point but increase memory and compute requirements.                                                                                                  | [Read more](context_length_scaling/README.md)  |
| 04   | LR Schedulers           | Compare different learning rate schedulers during training.                           | Schedulers with warmup and decay provide more stable training and faster convergence than constant learning rates.                                                                                                | [Read more](lr_schedulers/README.md)           |
| 05   | KV Cache                | Effect of KV cache on memory and compute.                                             | KV cache reduces compute by caching KV values during inference. Performance gains depend on the compute - memory trade off, where performance is more apparent for larger models where compute is the bottleneck. | [Read more](kv_cache/README.md)                |
| 06   | KV Cache strategies     | Study how different KV Caching strategies help use GPU memory optimally               | Paged caching strategy seems to work best, especially in case of fragmented memory allocations.                                                                                                                   | [Read more](kv_cache_strategies/README.md)     |

---

> New experiments should follow the same structure: each in its own folder, with an internal README summarizing results, links to plots/scripts, and reference to the base config file.
