class Config:
    # Model
    vocab_size = 32000  # placeholder. Patcherd later from dictionary
    block_size = 8  # Maximum context length
    n_layer = 4
    n_head = 4
    n_embed = 4
    dropout = 0.1

    # Training
    batch_size = 32  # How many independent sequences will be processed
    max_steps = 1000
    lr = 1e-2
    warmup_steps = 100
    weight_decay = 0.1
    grad_clip = 1.0

    # Precision
    use_fp16 = False
