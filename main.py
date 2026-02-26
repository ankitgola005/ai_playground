from data import Data
from config import Config
from bigram import BiGram
from trainer import Trainer

cuda = False  # torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
config = Config()
data = Data(config)
config.vocab_size = data.vocab_size

model = BiGram(config=config).to(device)
trainer = Trainer(config.max_steps)
trainer.configure_optimizer(model)
trainer.fit(model, data)
generated = trainer.generate(model, data.tokenizer)
print(f"{generated=}")
