import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from typing import List, Tuple, Dict, Optional


class Generator:
    """
    Autoregressive text generator for GPT-style models.

    Handles:
    - Prompt tokenization
    - KV cache prefill (optional)
    - Token-by-token decoding
    - Basic timing (prefill vs decode)

    This class is intentionally self-contained and does NOT depend on Trainer.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
    ) -> None:
        """
        Args:
            model (nn.Module): Language model (expects GPT-style forward).
            tokenizer: Tokenizer with encode/decode methods.
            device (torch.device): Device for inference.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.time_dict: Dict[str, float] = {
            "ctx_len": 0.0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
        }

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        use_cache: bool = True,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Generate text continuations for a batch of prompts.

        Args:
            prompts (List[str]): Input prompt strings.
            max_tokens (int): Number of tokens to generate.
            use_cache (bool): Whether to use KV caching.

        Returns:
            Tuple:
                - List[str]: Generated text outputs (decoded)
                - Dict[str, float]: Timing stats (ctx_len, prefill_time, decode_time)
        """
        self.time_dict_reset()

        token_list: List[List[int]] = [self.tokenizer.encode(p) for p in prompts]
        self.time_dict["ctx_len"] = len(token_list[0])

        model = self.model
        model.eval()

        with torch.inference_mode():
            B: int = len(prompts)
            max_len: int = max(len(t) for t in token_list)

            # Build padded batch
            batch_context = torch.zeros(
                B, max_len, dtype=torch.long, device=self.device
            )

            for i, tokens in enumerate(token_list):
                batch_context[i, : len(tokens)] = torch.tensor(
                    tokens, device=self.device
                )

            past_key_values: Optional[List] = None

            # Prefill
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            if use_cache:
                past_key_values = model.init_kv_cache(B, device=self.device)

                # Fill cache with full context
                model(
                    batch_context,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            output = batch_context

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.time_dict["prefill_time"] = time.perf_counter() - start_time

            # Decode
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(max_tokens):
                logits, _, past_key_values = model(
                    output[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

                next_token = self.sample(logits)

                # Safety check
                assert next_token.dim() == 2, f"Expected (B,1), got {next_token.shape}"

                output = torch.cat([output, next_token], dim=1)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.time_dict["decode_time"] = time.perf_counter() - start_time

        texts = [self.tokenizer.decode(o.tolist()) for o in output]
        return texts, self.time_dict

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Greedy sampling from logits.

        Args:
            logits (Tensor): (B, T, C)

        Returns:
            Tensor: Next tokens (B, 1)
        """
        return torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)

    def time_dict_reset(self) -> None:
        """Reset timing statistics."""
        self.time_dict = {
            "ctx_len": 0.0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
        }
