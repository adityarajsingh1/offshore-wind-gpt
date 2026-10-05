#!/usr/bin/env bash
# Runs the whole thing from scratch: data -> tokenizer -> pretrain -> fine-tune.
# Usage: ./run_pipeline.sh [preset]   (tiny | small | medium, default small)
set -euo pipefail

PRESET="${1:-small}"

echo "== 1/6 downloading articles =="
python scripts/collect_wikipedia.py

echo "== 2/6 cleaning and splitting =="
python scripts/prepare_corpus.py

echo "== 3/6 training the tokenizer =="
python scripts/train_tokenizer.py --vocab-size 4096

echo "== 4/6 tokenizing =="
python scripts/tokenize_corpus.py

echo "== 5/6 pretraining ($PRESET) =="
python train.py --preset "$PRESET"

echo "== 6/6 fine-tuning on Q&A =="
python finetune.py

echo
echo "all done! try it with: python chat.py"
