#!/usr/bin/env python3
"""
See what the pretrained model writes on its own. Good for checking
whether pretraining is going anywhere.

    python sample.py "Floating wind turbines"
"""

import argparse

import torch

from owgpt.tokenizer import Tokenizer
from owgpt.train_utils import load_model, pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Offshore wind")
    parser.add_argument("--ckpt", default="checkpoints/pretrain.pt")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("-n", type=int, default=2, help="how many samples")
    args = parser.parse_args()

    device = pick_device()
    tok = Tokenizer.load("data/processed/tokenizer.json")
    model, _ = load_model(args.ckpt, device)
    model.eval()

    for i in range(args.n):
        idx = torch.tensor([tok.encode(args.prompt)], device=device)
        out = model.generate(idx, args.tokens, temperature=args.temperature)
        print(f"--- sample {i + 1} ---")
        print(tok.decode(out[0].tolist()))
        print()


if __name__ == "__main__":
    main()
