#!/usr/bin/env python3
"""
Train the BPE tokenizer on the training split.

    python scripts/train_tokenizer.py --vocab-size 4096
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owgpt.tokenizer import Tokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--input", default=str(ROOT / "data" / "processed" / "train.txt"))
    parser.add_argument("--out", default=str(ROOT / "data" / "processed" / "tokenizer.json"))
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    print(f"training on {len(text):,} chars...")
    start = time.time()
    tok = Tokenizer()
    tok.train(text, args.vocab_size, verbose=True)
    tok.save(args.out)
    print(f"done in {time.time() - start:.0f}s, saved to {args.out}")

    sample = "The monopile foundation supports the nacelle and the turbine blades."
    ids = tok.encode(sample)
    print(f"\nexample: {len(sample)} chars -> {len(ids)} tokens")
    print([tok.decode([i]) for i in ids])


if __name__ == "__main__":
    main()
