# AI Playground

Experiments with transformers, training systems, and distributed training.

---

## Repository Structure

- `configs/` -- experiment configs
- `models/` -- model architectures
- `distributed/` -- distributed strategies
- `utils/` -- utility functions
- `trainer.py` -- training loop
- `train.py` -- running example

---

## Training Notes

### Instability & NaNs

1. **High learning rates** may cause training instability (NaNs or loss spikes), especially early in training.

2. **Large gradient norms** can destabilize training.

3. **Gradient norms are often large at the start**, but variance should decrease as training progresses.

4. If you encounter **NaNs**, try:
   - reducing the learning rate
   - enabling gradient clipping

5. **Loss patterns can indicate failure modes**
   - Zig-zag loss → NaN → usually **exploding gradients**
   - NaN first → smooth loss → usually **FP16 overflow**

6. **Sudden scaler drops** indicate overflow detection in mixed precision.

---

### Mixed Precision Notes

1. FP16 training may overflow in:
   - softmax
   - attention scores
   - large matmuls

2. If overflow occurs frequently:
   - reduce LR
   - reduce loss scale
   - switch to **BF16 if available**

3. **Dynamic loss scaling** prevents gradient overflow but may slow training when constantly adjusting.

4. BF16 overflows less often but check for hardware support.

---

### Debug Signals (Check in this order)

1. `loss`
2. `grad_norm`
3. `scaler`
4. `step_time`
5. `learning_rate`

---

### Common Failure Signals

| Symptom                   | Likely Cause                   |
| ------------------------- | ------------------------------ |
| Grad norm explodes        | LR too high                    |
| Loss becomes NaN suddenly | softmax overflow               |
| Scaler constantly drops   | FP16 overflow                  |
| Training very slow        | dataloader or GPU sync issue   |
| Grad norm = 0             | broken graph / detached tensor |

---

### Performance Checks

1. **GPU utilization should be high** (>90% ideally)
2. **Step time should be stable**
3. Large variance in step time → dataloader or CPU bottleneck
4. Small batch sizes often cause unstable gradients

---

### Reproducibility

For reproducible runs:

- set random seeds
- log config + hyperparameters
- log commit hash
- log environment info
