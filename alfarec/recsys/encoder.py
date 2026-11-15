"""Local, revision-pinned sentence encoding; no remote code or runtime downloads."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class SentenceEncoder:
    def __init__(self, path: str | Path, *, device: str = "cpu", e5: bool = False):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "cpu":
            torch.set_num_threads(4)
        self.model = SentenceTransformer(
            str(path), device=device, local_files_only=True, trust_remote_code=False
        )
        self.e5 = e5
        self.prompted = e5 or "passage" in self.model.prompts
        self.max_tokens = min(self.model.max_seq_length, 512)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        prefix = "query: " if self.prompted else ""
        return self.model.encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=16,
            prompt="",
        ).astype(np.float32)

    def documents(self, texts: Sequence[str]) -> np.ndarray:
        """Token chunks, with equal-weight averaging of their unit vectors."""
        tokenizer = self.model.tokenizer
        chunks, owners = [], []
        prefix = "passage: " if self.prompted else ""
        for owner, text in enumerate(texts):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            width = self.max_tokens - 16
            for start in range(0, max(len(tokens), 1), width):
                chunks.append(prefix + tokenizer.decode(tokens[start : start + width]))
                owners.append(owner)
        encoded = self.model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=16,
            prompt="",
        )
        vectors = np.zeros((len(texts), encoded.shape[1]), dtype=np.float32)
        for owner, vector in zip(owners, encoded, strict=True):
            vectors[owner] += vector
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or (norms == 0).any():
            raise ValueError("Invalid document embeddings")
        return vectors / norms
