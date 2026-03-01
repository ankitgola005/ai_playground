from config import Config


class MiniGPTConfig(Config):
    # Model
    block_size = 128  # context length, larger than tiny test
    n_layer = 2  # fewer layers for mini GPT
    n_head = 4  # attention heads
    n_embed = 64  # embedding dimension
    hidden_dim = 256  # FFN hidden size
    dropout = 0.0  # TBD

    # Training
    batch_size = 16
    max_steps = 1000
    val_interval = 50
    lr = 3e-4  # standard small GPT LR
    warmup_steps = 50
    weight_decay = 0.01
    grad_clip = 1.0
    save_interval = 100
