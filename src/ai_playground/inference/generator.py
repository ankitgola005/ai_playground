import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ai_playground.utils import load_checkpoint

if TYPE_CHECKING:
    from typing import List, Tuple, Dict
    from ai_playground.configs import TrainerConfig


class Generator:
    def __init__(
        self, config: TrainerConfig, model: nn.Module, tokenizer, device: torch.device
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        step = self._maybe_load_checkpoints()
        if step < 0:
            print("No checkpoint found. Generation will be garbage")
        else:
            print(f"Checkpoints found for step: {step}")

        self.time_dict = {
            "ctx_len": 0.0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
        }

    def _maybe_load_checkpoints(self):
        return load_checkpoint(
            trainer_config=self.config,
            model=self.model,
            device=self.device,
            optimizer=None,
            scheduler=None,
            scaler=None,
        )

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        use_cache: bool = True,
        temperature: float = 1.0,
        topk: int | None = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Generate text continuations for a batch of prompts.

        Args:
            prompts (List[str]): Input prompt strings.
            max_tokens (int): Number of tokens to generate.
            use_cache (bool): Whether to use KV caching.
            temperature (float):
            topk (Optional[int]): Sample top k max probs in generation

        Returns:
            Tuple:
                - List[str]: Generated text outputs (decoded)
                - Dict[str, float]: Timing stats (ctx_len, prefill_time, decode_time)
        """
        self.time_dict_reset()

        token_list: List = []

        for p in prompts:
            tokens = self.tokenizer.encode(p)

            # handle empty prompt
            if len(tokens) == 0:
                tokens = [self.tokenizer.eos_token_id]

            # truncate
            tokens = tokens[-self.model.block_size :]

            token_list.append(tokens)

        B = len(token_list)

        self.time_dict["ctx_len"] = max(len(t) for t in token_list)

        model = self.model
        model.eval()

        with torch.inference_mode():
            # Prefill (per sample)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            past_key_values: List | None = None

            if use_cache:
                past_key_values = model.init_kv_cache(B, device=self.device)
                assert past_key_values is not None

                # For paged KV cache, just pass full batch at once
                batch_tokens = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(t, device=self.device) for t in token_list],
                    batch_first=True,
                    padding_value=self.tokenizer.eos_token_id,
                )
                model(batch_tokens, past_key_values=past_key_values, use_cache=True)
                # For Contig KV Cache
                # for b in range(B):
                #     tokens = torch.tensor(token_list[b], device=self.device).unsqueeze(
                #         0
                #     )

                #     model(
                #         tokens,
                #         past_key_values=[
                #             pkv.slice_batch(b, b + 1) for pkv in past_key_values
                #         ],
                #         use_cache=True,
                #     )

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.time_dict["prefill_time"] = time.perf_counter() - start

            # Initialize decode state
            last_tokens = torch.tensor(
                [t[-1] for t in token_list],
                device=self.device,
            ).unsqueeze(1)

            output_tokens = [list(t) for t in token_list]
            finished = torch.zeros(B, dtype=torch.bool, device=self.device)

            # Decode
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(max_tokens):

                logits, _, past_key_values = model(
                    last_tokens,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

                next_token = self.sample(logits, temperature, topk)

                last_tokens = next_token

                for i in range(B):
                    if not finished[i]:
                        token_id = next_token[i].item()
                        output_tokens[i].append(token_id)

                        if token_id == self.tokenizer.eos_token_id:
                            finished[i] = True

                if finished.all():
                    break

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.time_dict["decode_time"] = time.perf_counter() - start

        texts = [self.tokenizer.decode(toks) for toks in output_tokens]
        return texts, self.time_dict

    def sample(self, logits, temperature=1.0, top_k=None):
        logits = logits[:, -1, :]  # (B, V)

        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            min_val = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val, torch.full_like(logits, -float("inf")), logits
            )

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def time_dict_reset(self) -> None:
        """Reset timing statistics."""
        self.time_dict = {
            "ctx_len": 0.0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
        }
