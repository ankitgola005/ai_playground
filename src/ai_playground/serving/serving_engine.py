from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import json
import torch

from ai_playground.inference.generator import Generator
from ai_playground.utils import get_config, build_data_pipeline, set_seed, build_model

# Load config & model
cfg = get_config("minigpt_config.yaml")
set_seed(cfg.trainer.seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer, _, _ = build_data_pipeline(cfg.data, batch_size=cfg.trainer.batch_size)
model = build_model(cfg.model)(cfg.model, tokenizer.vocab_size, cfg.data.block_size).to(
    device
)
model.eval()
generator = Generator(cfg.trainer, model, tokenizer, device=device)

# FastAPI app
app = FastAPI(title="Streaming GPT Generator")


class GenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 100
    use_cache: bool = True
    temperature: float = 1.0
    topk: int | None = None


@app.post("/generate_stream")
def generate_stream(req: GenerateRequest):
    def event_stream():
        # Generate for each prompt sequentially
        for prompt_id, prompt in enumerate(req.prompts):
            output_tokens = []

            # Use Generator to generate tokens one by one
            last_tokens = torch.tensor(
                [generator.tokenizer.encode(prompt)[-generator.model.block_size :]],
                device=generator.device,
            )
            finished = torch.zeros(1, dtype=torch.bool, device=generator.device)

            # Prefill if use_cache
            past_key_values = (
                generator.model.init_kv_cache(1, device=generator.device)
                if req.use_cache
                else None
            )

            for _ in range(req.max_tokens):
                logits, _, past_key_values = generator.model(
                    last_tokens,
                    past_key_values=past_key_values,
                    use_cache=req.use_cache,
                )

                next_token = generator.sample(logits, req.temperature, req.topk)
                token_id = next_token[0].item()
                output_tokens.append(token_id)

                # Send SSE for this token
                yield f"data: {json.dumps({'prompt_id': prompt_id, 'token': generator.tokenizer.decode([token_id])})}\n\n"

                # Update last_tokens and finished
                last_tokens = next_token
                if token_id == generator.tokenizer.eos_token_id:
                    finished[0] = True
                    break

            # Signal end of prompt
            yield f"data: {json.dumps({'prompt_id': prompt_id, 'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
