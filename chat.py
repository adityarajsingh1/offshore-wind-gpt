#!/usr/bin/env python3
"""
Ask the fine-tuned model questions about offshore wind.

    python chat.py
    python chat.py --ckpt checkpoints/sft.pt --temperature 0.5
"""

import argparse

import torch

from owgpt.sft import END, format_prompt
from owgpt.tokenizer import Tokenizer
from owgpt.train_utils import load_model, pick_device


def answer(model, tok, question, device, max_tokens=200, temperature=0.6):
    ids = tok.encode(format_prompt(question))
    idx = torch.tensor([ids], device=device)
    stop = {tok.special[END], tok.special["<|user|>"]}
    out = model.generate(idx, max_tokens, temperature=temperature, top_k=40, stop_ids=stop)
    reply = tok.decode(out[0, len(ids):].tolist())
    return reply.replace(END, "").replace("<|user|>", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/sft.pt")
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    device = pick_device()
    tok = Tokenizer.load("data/processed/tokenizer.json")
    model, _ = load_model(args.ckpt, device)
    model.eval()

    print("offshore wind GPT 🌊 ask me anything about offshore wind (ctrl+c to quit)\n")
    while True:
        try:
            q = input("you > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye!")
            break
        if q:
            print(f"gpt > {answer(model, tok, q, device, temperature=args.temperature)}\n")


if __name__ == "__main__":
    main()
