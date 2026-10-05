"""
A byte-level BPE tokenizer, written from scratch.

How BPE works in one paragraph: start with the 256 possible bytes as the
vocabulary. Find the pair of tokens that appears next to each other most
often in the training text, give that pair a new id, and replace it
everywhere. Repeat until the vocab is the size you want. Domain words like
"monopile" or "nacelle" end up as one or two tokens instead of six, which
is a big part of why training our own tokenizer is worth it.

Speed trick: we first split the text into words and count them, then run
the merges on the unique words (weighted by count) instead of the full text.
"""

import json
import re
from collections import Counter

# split into words / numbers / punctuation, keeping the leading space with the word (like GPT-2)
SPLIT = re.compile(r""" ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+""")

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>"]


def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self):
        self.merges = {}  # (a, b) -> new id, in the order they were learned
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special = {}  # "<|endoftext|>" -> id
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special)

    def train(self, text, vocab_size, verbose=False):
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        assert n_merges > 0, "vocab_size too small"

        # never learn merges across special tokens
        for tok in SPECIAL_TOKENS:
            text = text.replace(tok, " ")
        word_counts = Counter(SPLIT.findall(text))
        words = {w: list(w.encode("utf-8")) for w in word_counts}

        for step in range(n_merges):
            pairs = Counter()
            for w, ids in words.items():
                c = word_counts[w]
                for a, b in zip(ids, ids[1:]):
                    pairs[(a, b)] += c
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = 256 + step
            for w, ids in words.items():
                if len(ids) > 1:
                    words[w] = merge(ids, best, new_id)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            if verbose and step % 250 == 0:
                print(f"  merge {step}/{n_merges}: {self.vocab[new_id]!r} ({pairs[best]} times)")

        base = len(self.vocab)
        self.special = {tok: base + i for i, tok in enumerate(SPECIAL_TOKENS)}
        self._cache = {}

    def _encode_word(self, word):
        if word in self._cache:
            return self._cache[word]
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            # apply the earliest-learned merge available, same order as training
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text):
        if not self.special:
            return self._encode_plain(text)
        # pull special tokens out first so they become single ids
        pattern = "(" + "|".join(re.escape(t) for t in self.special) + ")"
        ids = []
        for part in re.split(pattern, text):
            if part in self.special:
                ids.append(self.special[part])
            elif part:
                ids.extend(self._encode_plain(part))
        return ids

    def _encode_plain(self, text):
        ids = []
        for word in SPLIT.findall(text):
            ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids):
        inverse = {v: k for k, v in self.special.items()}
        out = []
        for i in ids:
            out.append(inverse[i].encode("utf-8") if i in inverse else self.vocab[i])
        return b"".join(out).decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
            "special": self.special,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        for a, b, i in data["merges"]:
            tok.merges[(a, b)] = i
            tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
        tok.special = data["special"]
        return tok
