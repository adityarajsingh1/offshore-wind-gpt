"""
The GPT itself: a decoder-only transformer, same basic recipe as GPT-2.

  tokens -> token embedding + position embedding
         -> N x [ LayerNorm -> causal self-attention -> LayerNorm -> MLP ]
         -> LayerNorm -> linear layer -> next-token probabilities

Written to be read, so no fancy tricks beyond what actually matters.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def rope_tables(head_dim, max_len, base=10000.0):
    """
    Precompute the rotation angles for rotary embeddings (RoPE).

    RoPE encodes position by rotating pairs of dimensions in q and k by an
    angle that grows with the position. The dot product q.k then only
    depends on how far apart two tokens are, which is exactly what attention
    cares about, and there are no extra parameters to learn.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (max_len, head_dim / 2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x is (B, heads, T, head_dim). rotate each (even, odd) pair of dims
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, past=None, rope=None):
        """
        past is the (k, v) cache from earlier tokens, only used when generating.
        rope is (cos, sin) for the positions of the tokens in x, if RoPE is on.
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_size)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            # rotate before caching, so cached keys already carry their positions
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if past is not None:
            # the new token(s) attend to everything we've already seen
            assert T == 1, "with a cache, feed one token at a time"
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        # is_causal=True means each position can only look at itself and earlier positions.
        # a single new token is the last position anyway, so it needs no mask
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=past is None, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, past=None, rope=None):
        # pre-norm + residual connections, trains much more stably than post-norm
        attn_out, present = self.attn(self.ln1(x), past, rope)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.rope:
            cos, sin = rope_tables(cfg.n_embd // cfg.n_head, cfg.block_size)
            # buffers move with .to(device) but aren't saved, they're cheap to rebuild
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # share weights between the input embedding and the output layer (saves params, helps a bit)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: scale down the residual projections by depth
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # position embeddings aren't really "model" params, GPT-2 paper excludes them too
        n = sum(p.numel() for p in self.parameters())
        return n if self.cfg.rope else n - self.pos_emb.weight.numel()

    def _embed(self, idx, start=0):
        """Token embeddings (+ learned positions if not using RoPE), and the rope angles."""
        T = idx.size(1)
        x = self.tok_emb(idx)
        if self.cfg.rope:
            return x, (self.rope_cos[start:start + T], self.rope_sin[start:start + T])
        pos = torch.arange(start, start + T, device=idx.device)
        return x + self.pos_emb(pos), None

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} is longer than block_size {self.cfg.block_size}"
        x, rope = self._embed(idx)
        x = self.drop(x)
        for block in self.blocks:
            x, _ = block(x, rope=rope)
        logits = self.head(self.ln_f(x))

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if loss_mask is not None:
            # for fine-tuning: only learn from the answer tokens, not the question
            mask = loss_mask.reshape(-1).float()
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        return logits, loss

    @torch.no_grad()
    def forward_cached(self, idx, cache=None):
        """
        Like forward(), but keeps the attention keys/values of earlier tokens
        so each new token only costs one position of work instead of
        re-running the whole sequence. Returns (logits, new_cache).
        """
        start = 0 if cache is None else cache[0][0].size(2)
        x, rope = self._embed(idx, start)
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, None if cache is None else cache[i], rope)
            new_cache.append(present)
        return self.head(self.ln_f(x)), new_cache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, stop_ids=None, use_cache=True):
        cache = None
        for _ in range(max_new_tokens):
            # the cache only works while everything fits in the context window (the
            # position tables stop at block_size). after that, crop and recompute
            if use_cache and idx.size(1) <= self.cfg.block_size:
                if cache is None:
                    logits, cache = self.forward_cached(idx)
                else:
                    logits, cache = self.forward_cached(idx[:, -1:], cache)
            else:
                cache = None
                logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_ids and next_id.item() in stop_ids:
                break
        return idx

    def configure_optimizer(self, lr, weight_decay):
        # weight decay on the big matrices only, not on biases / layernorm / embeddings-as-vectors
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
