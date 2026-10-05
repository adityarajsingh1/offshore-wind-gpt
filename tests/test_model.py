import torch

from owgpt.config import ModelConfig
from owgpt.model import GPT

CFG = ModelConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_forward_shapes_and_loss():
    model = GPT(CFG)
    x = torch.randint(0, 100, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 100)
    # an untrained model should be close to random guessing: ln(100) ~ 4.6
    assert 3.5 < loss.item() < 5.5


def test_causal_no_peeking_at_the_future():
    model = GPT(CFG).eval()
    x = torch.randint(0, 100, (1, 10))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 100  # change only the last token
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)


def test_loss_mask_ignores_masked_tokens():
    model = GPT(CFG)
    x = torch.randint(0, 100, (1, 8))
    mask = torch.zeros(1, 8, dtype=torch.long)
    mask[0, 4:] = 1
    _, masked = model(x, x, loss_mask=mask)
    y2 = x.clone()
    y2[0, :4] = 0  # changing masked-out targets shouldn't change the loss
    _, masked2 = model(x, y2, loss_mask=mask)
    assert torch.isclose(masked, masked2)


def test_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    model = GPT(CFG)
    opt = model.configure_optimizer(1e-2, 0.0)
    x = torch.randint(0, 100, (2, 16))
    for _ in range(60):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


def test_generate_stops_at_stop_token():
    model = GPT(CFG).eval()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 20, stop_ids=set(range(100)))
    assert out.shape[1] == 2
