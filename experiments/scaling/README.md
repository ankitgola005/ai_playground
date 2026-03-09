# Experiment 1: Effect of parameter scaling on validation loss

##  Experimental Setup

| Component   | Details |
|------------|---------|
| Dataset    | `shakespeare.txt` from `ai_playground/data/datasets/text_datasets/` |
| Model      | MiniGPT-style transformer |
| Config     | [gpt_config.yaml](../../configs/gpt_config.yaml) |


> **Objective:** Study the effect of model width (hidden dimension / number of heads) on validation loss.  

---

## Steps to reproduce the results

From the experiment folder:

```bash
python -u scaling.py --law [ "width" | "depth" | "depth_width" ]
```
---

**Parameters overridden for the experiment:**

- depth: `model.model_kwargs.n_layer`  
- width: `model.model_kwargs.n_embed`, and `model.model_kwargs.hidden_dim`  

---

## 1.1 Width vs Validation Loss

In this experiment, we varied the model width by sweeping `n_embed` while keeping other hyperparameters constant.

### Observations

- The validation loss decreases as the embedding dimension (`n_embed`) increases.  
- The trend roughly follows a **reciprocal curve**.  
- Increasing width improves performance, but memory cost rises faster than performance gains beyond a point.
- **Diminishing returns**:
  - Small widths see a rapid decrease in loss.
  - Large widths still improve, but improvements are smaller, indicating sufficient model capacity for given dataset.
 

### Plot

**Validation Loss vs Width**

![Validation Loss Plot](./.assets/width_vs_val_loss.png)

- The plot traces a rectangular hyperbola, illustrating diminishing returns with larger embedding dimensions.

## 1.2 Depth vs Validation Loss

In this experiment, we varied the model depth by changing the number of layers while keeping other hyperparameters constant.

### Observations

- The validation loss **decreases with increasing depth**.
- Increasing depth improves model performance, but **gains are more gradual** compared to increasing width.  
- Depth scaling alone is less impactful than width scaling for this dataset and model size.  
- **Diminishing returns** are still present:
  - Early layers provide noticeable improvement, as seen by steeper slope. 
  - Adding more layers after a point yields smaller decreases in validation loss.


### Plot

**Validation Loss vs Depth**

![Validation Loss vs Depth](./.assets/depth_vs_val_loss.png)

- The plot shows a **gentler downward trend**, reflecting the less pronounced behavior compared to width.