from __future__ import annotations

from collections import Counter

import torch


class SimpleTokenizer:
    def __init__(self, texts: list[str], max_len: int = 24) -> None:
        self.max_len = max_len
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._split(text))
        vocab = ["<pad>", "<unk>"] + sorted(counter.keys())
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = vocab

    @staticmethod
    def _split(text: str) -> list[str]:
        return text.lower().replace("-", " ").replace("_", " ").split()

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> torch.Tensor:
        tokens = self._split(text)[: self.max_len]
        ids = [self.stoi.get(token, 1) for token in tokens]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)
