#!/usr/bin/env python3
"""
Plot the training curves written by train.py and finetune.py.

    python plot_loss.py                # saves checkpoints/loss.png

If val loss starts going up while train loss keeps dropping, the model is
overfitting: more data or fewer steps.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # works without a display, e.g. on a server
import matplotlib.pyplot as plt  # noqa: E402

LOGS = {"pretraining": Path("checkpoints/pretrain_log.csv"), "fine-tuning": Path("checkpoints/sft_log.csv")}


def read_log(path):
    steps, train, val = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return steps, train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="checkpoints/loss.png")
    args = parser.parse_args()

    found = {name: p for name, p in LOGS.items() if p.exists()}
    if not found:
        raise SystemExit("no loss logs yet, train something first")

    fig, axes = plt.subplots(1, len(found), figsize=(6 * len(found), 4), squeeze=False)
    for ax, (name, path) in zip(axes[0], found.items()):
        steps, train, val = read_log(path)
        ax.plot(steps, train, label="train")
        ax.plot(steps, val, label="val")
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
