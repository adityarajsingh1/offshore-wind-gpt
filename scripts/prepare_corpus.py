#!/usr/bin/env python3
"""
Combine everything in data/ into one cleaned corpus file and split it
into train / validation.

Sources (all optional, use whatever you have):
  data/raw/wikipedia/*.txt   from collect_wikipedia.py
  data/raw/extra/*.txt|.md   anything else you want to add (reports, notes...)
  data/notes/*.md            the hand-written primers that ship with the repo

    python scripts/prepare_corpus.py
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owgpt.clean import clean_text, clean_wikipedia  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "processed"
DOC_SEP = "\n<|endoftext|>\n"
VAL_FRACTION = 0.1


def gather():
    docs = []
    for f in sorted((DATA / "raw" / "wikipedia").glob("*.txt")):
        docs.append((f.name, clean_wikipedia(f.read_text(encoding="utf-8"))))
    for pattern in ("*.txt", "*.md"):
        for f in sorted((DATA / "raw" / "extra").glob(pattern)):
            docs.append((f.name, clean_text(f.read_text(encoding="utf-8"))))
    for f in sorted((DATA / "notes").glob("*.md")):
        docs.append((f.name, clean_text(f.read_text(encoding="utf-8"))))
    return [(name, text) for name, text in docs if len(text) > 200]


def main():
    docs = gather()
    if not docs:
        sys.exit("no documents found, run scripts/collect_wikipedia.py first")

    # split by document (not by character) so val text is genuinely unseen
    random.Random(42).shuffle(docs)
    n_val = max(1, int(len(docs) * VAL_FRACTION))
    val, train = docs[:n_val], docs[n_val:]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        text = DOC_SEP.join(t for _, t in split)
        (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"{name}: {len(split)} docs, {len(text):,} chars")


if __name__ == "__main__":
    main()
