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


def test_kv_cache_gives_the_same_logits():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    x = torch.randint(0, 100, (1, 12))
    full, _ = model(x)
    # feed the first 5 tokens at once, then the rest one by one through the cache
    logits, cache = model.forward_cached(x[:, :5])
    steps = [logits[:, -1]]
    for t in range(5, 12):
        logits, cache = model.forward_cached(x[:, t:t + 1], cache)
        steps.append(logits[:, -1])
    assert torch.allclose(torch.stack(steps, 1), full[:, 4:], atol=1e-4)


def test_generate_same_with_and_without_cache():
    model = GPT(CFG).eval()
    start = torch.randint(0, 100, (1, 3))
    # past block_size too, so the fallback path gets used as well
    torch.manual_seed(1)
    a = model.generate(start, 40, temperature=1e-6, top_k=1, use_cache=True)
    torch.manual_seed(1)
    b = model.generate(start, 40, temperature=1e-6, top_k=1, use_cache=False)
    assert torch.equal(a, b)


def test_both_position_schemes_work():
    from dataclasses import replace
    for rope in (True, False):
        model = GPT(replace(CFG, rope=rope)).eval()
        x = torch.randint(0, 100, (1, 10))
        full, _ = model(x)
        logits, cache = model.forward_cached(x[:, :4])
        for t in range(4, 10):
            logits, cache = model.forward_cached(x[:, t:t + 1], cache)
        assert torch.allclose(logits[:, -1], full[:, -1], atol=1e-4)


def test_rope_scores_depend_on_distance_only():
    from owgpt.model import apply_rope, rope_tables
    cos, sin = rope_tables(8, 32)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
    def score(i, j):
        qi = apply_rope(q, cos[i:i + 1], sin[i:i + 1])
        kj = apply_rope(k, cos[j:j + 1], sin[j:j + 1])
        return (qi * kj).sum()
    # same gap of 3, different absolute positions
    assert torch.isclose(score(5, 2), score(20, 17), atol=1e-5)


def test_old_checkpoints_without_rope_still_load(tmp_path):
    from dataclasses import asdict, replace
    from owgpt.train_utils import load_model
    old = GPT(replace(CFG, rope=False))
    cfg = asdict(old.cfg)
    del cfg["rope"]  # what an old checkpoint looks like
    torch.save({"model": old.state_dict(), "model_config": cfg, "step": 0}, tmp_path / "old.pt")
    model, _ = load_model(tmp_path / "old.pt")
    assert model.cfg.rope is False
