"""Bits shared by pretraining and fine-tuning."""

import math

import torch

from .config import ModelConfig
from .model import GPT


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # apple silicon
        return "mps"
    return "cpu"


def lr_at(step, cfg):
    """Linear warmup, then cosine decay down to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, model, optimizer, step, val_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "model_config": vars(model.cfg),
        "step": step,
        "val_loss": val_loss,
    }, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = GPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


class LossLog:
    """Appends eval results to a small csv so runs can be plotted later."""

    def __init__(self, path, fresh=True):
        self.path = path
        if fresh or not path.exists():
            path.write_text("step,train_loss,val_loss\n", encoding="utf-8")

    def add(self, step, train_loss, val_loss):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{step},{train_loss:.4f},{val_loss:.4f}\n")
