"""
Supervised fine-tuning (SFT) data.

After pretraining the model can write offshore wind text, but it doesn't
know it's supposed to *answer questions*. SFT fixes that: we show it lots
of (question, answer) pairs in a chat format and train it to produce the
answer. The loss is only computed on the answer tokens, so it learns
"how to respond" rather than memorising the questions.

Format of one example:
    <|user|>What is a monopile?<|assistant|>A monopile is ...<|endoftext|>
"""

import json
import random

import torch

USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|endoftext|>"


def format_prompt(question):
    return f"{USER}{question.strip()}{ASSISTANT}"


def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("question") or not row.get("answer"):
                raise ValueError(f"{path}:{line_no} needs a question and an answer")
            pairs.append((row["question"], row["answer"]))
    return pairs


def encode_pair(tok, question, answer, block_size):
    """Token ids plus a mask that is 1 only where the model should learn (the answer)."""
    prompt_ids = tok.encode(format_prompt(question))
    answer_ids = tok.encode(answer.strip() + END)
    ids = (prompt_ids + answer_ids)[: block_size + 1]
    mask = ([0] * len(prompt_ids) + [1] * len(answer_ids))[: block_size + 1]
    return ids, mask


class SFTBatches:
    def __init__(self, tok, pairs, block_size, pad_id):
        self.examples = [encode_pair(tok, q, a, block_size) for q, a in pairs]
        self.pad_id = pad_id

    def __len__(self):
        return len(self.examples)

    def get_batch(self, batch_size, device="cpu"):
        batch = random.sample(self.examples, min(batch_size, len(self.examples)))
        T = max(len(ids) for ids, _ in batch) - 1
        x = torch.full((len(batch), T), self.pad_id, dtype=torch.long)
        y = torch.full((len(batch), T), self.pad_id, dtype=torch.long)
        m = torch.zeros((len(batch), T), dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids) - 1
            x[i, :n] = torch.tensor(ids[:-1])
            y[i, :n] = torch.tensor(ids[1:])
            m[i, :n] = torch.tensor(mask[1:])  # mask lines up with the targets
        return x.to(device), y.to(device), m.to(device)
