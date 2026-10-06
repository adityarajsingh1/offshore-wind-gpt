#!/usr/bin/env python3
"""
How good is the model, actually? This scores a checkpoint on a held-out
set of offshore wind questions (data/eval/eval_qa.jsonl) that are never
used for training.

Two numbers:
  - answer perplexity: how surprised the model is by the reference answer.
    lower is better, and it's smooth, so it shows progress even when the
    generated text is still bad.
  - keyword recall: the model answers each question (greedy-ish), and we
    check how many of the expected keywords show up in its answer. crude,
    but it tracks "did it say the right thing" without needing another LLM.

    python evaluate.py
    python evaluate.py --ckpt checkpoints/pretrain.pt --show
"""

import argparse
import json
import math

import torch

from chat import answer
from owgpt.sft import encode_pair
from owgpt.tokenizer import Tokenizer
from owgpt.train_utils import load_model, pick_device


def load_eval(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def answer_perplexity(model, tok, rows, device):
    total_loss, total_tokens = 0.0, 0
    for row in rows:
        ids, mask = encode_pair(tok, row["question"], row["answer"], model.cfg.block_size)
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        m = torch.tensor([mask[1:]], device=device)
        _, loss = model(x, y, loss_mask=m)
        n = int(m.sum())
        total_loss += loss.item() * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


def keyword_recall(text, keywords):
    text = text.lower()
    hits = [k for k in keywords if k.lower() in text]
    return len(hits) / len(keywords), hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/sft.pt")
    parser.add_argument("--data", default="data/eval/eval_qa.jsonl")
    parser.add_argument("--show", action="store_true", help="print every question and answer")
    args = parser.parse_args()

    device = pick_device()
    torch.manual_seed(0)  # so generated answers are repeatable between runs
    tok = Tokenizer.load("data/processed/tokenizer.json")
    model, ckpt = load_model(args.ckpt, device)
    model.eval()
    rows = load_eval(args.data)

    ppl = answer_perplexity(model, tok, rows, device)

    recalls = []
    for row in rows:
        reply = answer(model, tok, row["question"], device, max_tokens=80, temperature=0.1)
        score, hits = keyword_recall(reply, row["keywords"])
        recalls.append(score)
        if args.show:
            print(f"Q: {row['question']}\nA: {reply}\n   keywords {hits} / {row['keywords']} -> {score:.0%}\n")

    print(f"checkpoint:      {args.ckpt} (step {ckpt.get('step')})")
    print(f"questions:       {len(rows)}")
    print(f"answer ppl:      {ppl:.1f}")
    print(f"keyword recall:  {sum(recalls) / len(recalls):.0%}")


if __name__ == "__main__":
    main()
