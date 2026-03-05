# ai_playground

# Notes:
# 1.    A high LR may cause training instability (NaNs / loss spikes) especially in the early stages.
# 2.    A high gradient may cause training instability (NaNs / loss spikes) especially in the early stages.
# 3.    Gradient norms can be large at start of training, but variance should reduce as training progress.   
# 4.    If you see NaNs, try reducing the LR and/or enabling gradient clipping.
# 5.    A loss zig zig followed by NaN is usually exploding gradients (usually followed by softmax overflow), while a NaN followed by smooth loss is usually FP16 overflow.
# 6.    Scalar drop is sign of overflow detection
# 7.    General checks for debug, in decreasing order of priority: loss, grad_norms, scaler, step_time