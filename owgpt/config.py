"""
All the knobs in one place.

The defaults are sized so the model trains on a laptop CPU in reasonable
time. If you have a GPU, bump n_layer / n_embd / block_size up and use
the "medium" preset.
"""

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    block_size: int = 256   # how many tokens the model can see at once
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    rope: bool = True       # rotary position embeddings instead of a learned position table


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 5000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 1337


PRESETS = {
    # quick sanity check, trains in a couple of minutes on cpu
    "tiny": ModelConfig(block_size=128, n_layer=2, n_head=2, n_embd=128),
    # the default
    "small": ModelConfig(),
    # needs a gpu really
    "medium": ModelConfig(block_size=512, n_layer=12, n_head=12, n_embd=768),
}


def to_dict(cfg):
    return asdict(cfg)
