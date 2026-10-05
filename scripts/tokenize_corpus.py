#!/usr/bin/env python3
"""
Turn data/processed/{train,val}.txt into token id files for training.

    python scripts/tokenize_corpus.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owgpt.data import tokenize_file  # noqa: E402
from owgpt.tokenizer import Tokenizer  # noqa: E402

PROC = ROOT / "data" / "processed"


def main():
    tok = Tokenizer.load(PROC / "tokenizer.json")
    for split in ("train", "val"):
        n = tokenize_file(tok, PROC / f"{split}.txt", PROC / f"{split}.bin")
        print(f"{split}: {n:,} tokens")


if __name__ == "__main__":
    main()
