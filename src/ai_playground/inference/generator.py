import torch
import time


class Generator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.time_dict = {
            "ctx_len": 0.0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
        }

    def generate(
        self,
        prompts,
        max_tokens=500,
        use_cache=True,
    ):
        self.time_dict_reset()
        token_list = [self.tokenizer.encode(p) for p in prompts]
        self.time_dict["ctx_len"] = len(token_list[0])

        model = self.model
        model.eval()

        with torch.inference_mode():

            token_list = [self.tokenizer.encode(p) for p in prompts]
            max_len = max(len(t) for t in token_list)

            B = len(prompts)

            batch_context = torch.zeros(
                B, max_len, dtype=torch.long, device=self.device
            )

            for i, tokens in enumerate(token_list):
                batch_context[i, : len(tokens)] = torch.tensor(
                    tokens, device=self.device
                )

            past_key_values = None

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            if use_cache:
                past_key_values = model.init_kv_cache(
                    B,
                    device=self.device,
                )
                # Prefill
                model(batch_context, past_key_values=past_key_values, use_cache=True)
            output = batch_context

            torch.cuda.synchronize()
            prefill_time = time.perf_counter() - start_time

            torch.cuda.synchronize()
            self.time_dict["prefill_time"] = prefill_time
            start_time = time.perf_counter()
            for _ in range(max_tokens):
                # decode
                logits, _, past_key_values = model(
                    output[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

                next_token = self.sample(logits)
                output = torch.cat([output, next_token], dim=1)
            torch.cuda.synchronize()
            decode_time = time.perf_counter() - start_time
            self.time_dict["decode_time"] = decode_time
        return [self.tokenizer.decode(o.tolist()) for o in output]

    def sample(self, logits):
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    def time_dict_reset(self):
        self.time_dict = {
            "ctx_len": 0.0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
        }
