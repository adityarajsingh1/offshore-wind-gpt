"""
Data loading for pretraining.

The whole corpus is tokenized once and saved as a flat array of uint16
token ids (fine while vocab < 65536). Training then just grabs random
windows of block_size + 1 tokens: the first block_size are the input,
and the same window shifted by one is the target.
"""

from pathlib import Path

import numpy as np
import torch


def tokenize_file(tokenizer, txt_path, bin_path):
    text = Path(txt_path).read_text(encoding="utf-8")
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    ids.tofile(bin_path)
    return len(ids)


def load_tokens(bin_path):
    # memmap so we don't load a huge file into ram
    return np.memmap(bin_path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size, device="cpu"):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
