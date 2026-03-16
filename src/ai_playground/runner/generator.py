import torch


class Generator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompts,
        max_tokens=500,
        use_cache=True,
        max_cache_len=256,
    ):

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

            if use_cache:
                past_key_values = model.init_kv_cache(
                    B,
                    max_cache_len,
                    device=self.device,
                )

                model(batch_context, past_key_values=past_key_values, use_cache=True)

            output = batch_context

            for _ in range(max_tokens):

                logits, _, past_key_values = model(
                    output[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

                next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                output = torch.cat([output, next_token], dim=1)

        return [self.tokenizer.decode(o.tolist()) for o in output]
