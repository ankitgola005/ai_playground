import torch

from ai_playground.models.bigram import BiGram
from ai_playground.utils import set_seed


def make_model(vocab_size=20):
    return BiGram(vocab_size=vocab_size)


def test_forward_no_targets():
    model = make_model()
    idx = torch.randint(0, 20, (2, 5))  # (B=2, T=5)
    logits, loss = model(idx)

    assert logits.shape == (2, 5, 20)
    assert loss is None


def test_forward_with_targets():
    model = make_model()
    idx = torch.randint(0, 20, (2, 5))
    targets = torch.randint(0, 20, (2, 5))
    logits, loss = model(idx, targets)

    assert logits.shape == (2, 5, 20)
    assert loss is not None
    assert loss.item() >= 0  # CE loss always >= 0


def test_loss_reduction():
    model = make_model(vocab_size=10)
    idx = torch.randint(0, 10, (1, 3))
    targets = torch.randint(0, 10, (1, 3))
    logits, loss = model(idx, targets)

    # manual CE check
    logits_flat = logits.view(-1, 10)
    targets_flat = targets.view(-1)

    loss_manual = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    assert torch.allclose(loss, loss_manual)


def test_generate_shape():
    model = make_model(vocab_size=15)
    idx = torch.randint(0, 15, (2, 4))
    out = model.generate(idx, max_new_tokens=3)

    assert out.shape == (2, 7)


def test_generate_appends_tokens():
    model = make_model(vocab_size=15)
    idx = torch.randint(0, 15, (1, 5))
    out = model.generate(idx, max_new_tokens=2)

    assert out.shape[1] == 7


def test_generate_deterministic_with_seed():
    out = []
    for _ in range(2):
        set_seed(42)
        model = make_model(vocab_size=10)
        idx = torch.randint(0, 10, (1, 3))
        _out = model.generate(idx, max_new_tokens=5)
        out.append(_out)

    assert torch.equal(out[0], out[1])


def test_generated_tokens_in_vocab():
    model = make_model(vocab_size=10)
    idx = torch.randint(0, 10, (1, 3))
    out = model.generate(idx, max_new_tokens=5)

    assert torch.all(out >= 0)
    assert torch.all(out < 10)


def test_generate_zero_tokens():
    model = make_model(vocab_size=10)
    idx = torch.randint(0, 10, (2, 4))
    out = model.generate(idx, max_new_tokens=0)

    assert torch.equal(out, idx)
