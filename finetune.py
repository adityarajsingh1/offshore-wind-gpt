#!/usr/bin/env python3
"""
Step 2 of 2: supervised fine-tuning on question/answer pairs.

Starts from the pretrained checkpoint and teaches the model to answer
questions in a chat format.

    python finetune.py
    python finetune.py --data data/qa/my_extra_qa.jsonl --steps 1500
"""

import argparse
import random
from pathlib import Path

import torch

from owgpt.sft import SFTBatches, load_pairs
from owgpt.tokenizer import Tokenizer
from owgpt.train_utils import load_model, pick_device, save_checkpoint

PROC = Path("data/processed")


@torch.no_grad()
def eval_loss(model, data, batches, device):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y, m = data.get_batch(16, device)
        losses.append(model(x, y, loss_mask=m)[1].item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=sorted(str(p) for p in Path("data/qa").glob("*.jsonl")))
    parser.add_argument("--base", default="checkpoints/pretrain.pt")
    parser.add_argument("--out", default="checkpoints/sft.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)  # lower than pretraining so we don't wreck what it learned
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = pick_device()
    tok = Tokenizer.load(PROC / "tokenizer.json")
    model, _ = load_model(args.base, device)

    pairs = [p for path in args.data for p in load_pairs(path)]
    random.Random(0).shuffle(pairs)
    n_val = max(1, len(pairs) // 10)
    pad = tok.special["<|endoftext|>"]
    train = SFTBatches(tok, pairs[n_val:], model.cfg.block_size, pad)
    val = SFTBatches(tok, pairs[:n_val], model.cfg.block_size, pad)
    print(f"{len(train)} train pairs, {len(val)} val pairs, device {device}")

    optimizer = model.configure_optimizer(args.lr, weight_decay=0.0)
    best = float("inf")
    for step in range(args.steps):
        x, y, m = train.get_batch(args.batch_size, device)
        _, loss = model(x, y, loss_mask=m)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == args.steps - 1:
            v = eval_loss(model, val, 5, device)
            print(f"step {step}: train {loss.item():.3f}, val {v:.3f}")
            if v < best:
                best = v
                save_checkpoint(args.out, model, None, step, v)

    print(f"done. best val loss {best:.3f}, saved to {args.out}")


if __name__ == "__main__":
    main()
