class Config:
    # Reproducibility
    seed = 42

    # Data
    split = 0.9
    num_workers = 0

    # Model
    block_size = 8  # Maximum context length
    n_layer = 4
    n_head = 4
    n_embed = 4
    dropout = 0.1

    # Training
    batch_size = 32  # How many independent sequences will be processed
    max_steps = 500
    val_interval = 10
    lr = 1e-2
    warmup_steps = 100
    weight_decay = 0.1
    grad_clip = 1.0
    save_interval = 0  # checkpoint every 100 steps

    # Precision
    use_fp16 = False
