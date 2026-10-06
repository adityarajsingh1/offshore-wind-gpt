#!/usr/bin/env python3
"""
Step 1 of 2: pretraining. The model learns to predict the next token
on the offshore wind corpus, which is how it picks up the vocabulary,
facts and writing style of the domain.

    python train.py                  # default "small" model
    python train.py --preset tiny    # quick test run
    python train.py --resume         # carry on from the last checkpoint
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from owgpt.config import PRESETS, TrainConfig
from owgpt.data import get_batch, load_tokens
from owgpt.model import GPT
from owgpt.tokenizer import Tokenizer
from owgpt.train_utils import LossLog, load_model, lr_at, pick_device, save_checkpoint

PROC = Path("data/processed")
CKPT_DIR = Path("checkpoints")


@torch.no_grad()
def estimate_loss(model, splits, cfg, block_size, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, cfg.batch_size, block_size, device))[1].item()
                  for _ in range(cfg.eval_batches)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS.keys(), default="small")
    parser.add_argument("--steps", type=int, help="override max_steps")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.steps:
        cfg = replace(cfg, max_steps=args.steps)
    if args.batch_size:
        cfg = replace(cfg, batch_size=args.batch_size)
    torch.manual_seed(cfg.seed)
    device = pick_device()

    tok = Tokenizer.load(PROC / "tokenizer.json")
    train_data = load_tokens(PROC / "train.bin")
    val_data = load_tokens(PROC / "val.bin")
    print(f"train {len(train_data):,} tokens | val {len(val_data):,} tokens | device {device}")

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / "pretrain.pt"
    start_step = 0
    if args.resume and ckpt_path.exists():
        model, ckpt = load_model(ckpt_path, device)
        start_step = ckpt["step"] + 1
        print(f"resuming from step {start_step}")
    else:
        model_cfg = replace(PRESETS[args.preset], vocab_size=tok.vocab_size)
        model = GPT(model_cfg).to(device)
        ckpt = None
    print(f"model has {model.num_params() / 1e6:.1f}M parameters")

    optimizer = model.configure_optimizer(cfg.lr, cfg.weight_decay)
    if ckpt and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])

    block_size = model.cfg.block_size
    log = LossLog(CKPT_DIR / "pretrain_log.csv", fresh=not args.resume)
    best_val = float("inf")
    t0 = time.time()

    for step in range(start_step, cfg.max_steps):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, cfg)

        if step % cfg.eval_every == 0 or step == cfg.max_steps - 1:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, cfg, block_size, device)
            print(f"step {step}: train {losses['train']:.3f}, val {losses['val']:.3f}")
            log.add(step, losses["train"], losses["val"])
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(ckpt_path, model, optimizer, step, best_val)

        x, y = get_batch(train_data, cfg.batch_size, block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % 50 == 0:
            print(f"  step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")

    print(f"done. best val loss {best_val:.3f}, saved to {ckpt_path}")


if __name__ == "__main__":
    main()
