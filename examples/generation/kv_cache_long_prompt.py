import torch
from ai_playground.inference.generator import Generator
from ai_playground.utils import get_config, build_data_pipeline, set_seed, build_model

cfg = get_config("minigpt_config.yaml")
set_seed(cfg.trainer.seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer, _, _ = build_data_pipeline(cfg.data, batch_size=cfg.trainer.batch_size)
model = build_model(cfg.model)(cfg.model, tokenizer.vocab_size, cfg.data.block_size)
model = model.to(device)
model.eval()
generator = Generator(model, tokenizer, device=device)

prompts = [
    "",  # empty
    "To be",  # short
    "O Romeo, Romeo! wherefore art thou Romeo?",  # medium
    "This is a much longer prompt that tests the paged KV cache and generation over multiple blocks.",  # long
]

MAX_TOKENS = 20
for use_cache in [False, True]:
    print(f"\n=== Generating with use_cache={use_cache} ===\n")
    outputs, stats = generator.generate(
        prompts, max_tokens=MAX_TOKENS, use_cache=use_cache
    )

    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        print(f"Generated tokens: {len(output) - len(prompt)}\n")

print("\n=== Determinism Check ===\n")
outputs1, _ = generator.generate(
    prompts, max_tokens=MAX_TOKENS, use_cache=True, temperature=0.0
)
outputs2, _ = generator.generate(
    prompts, max_tokens=MAX_TOKENS, use_cache=True, temperature=0.0
)

for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
    match = "PASS" if o1 == o2 else "FAIL"
    print(f"Prompt {i} deterministic match: {match}")
