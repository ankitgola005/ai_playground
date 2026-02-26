import torch


class Trainer:
    def __init__(self, device="cpu", optimizer=None, max_steps=1):
        self.max_steps = max_steps
        self.device = device
        self.optimizer = optimizer

    def configure_optimizer(self, model):
        self.optimizer = (
            self.optimizer
            if self.optimizer is not None
            else torch.optim.AdamW(model.parameters(), lr=0.01)
        )

    def fit(self, model, data):
        for iter in range(self.max_steps):
            xb, yb = data.get_batch()

            logits, loss = model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def generate(self, model, tokenizer, context=None, num_tokens=500):
        context = (
            context if context is not None else torch.zeros((1, 1), dtype=torch.long, device=self.device)
        )
        return tokenizer.decode(
            model.generate(context, max_new_tokens=num_tokens)[0].tolist()
        )
