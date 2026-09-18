"""Deterministic feature-hashing embeddings (CP 3.1: "embedded as vector math,
readable semantically and lower data footprint").

No model download, no API key, reproducible across runs and platforms
(md5-salted hash buckets, NOT Python's salted per-process hash()).
Cosine similarity on L2-normalized vectors == dot product.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Sequence

DIM = 512

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _bucket(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % DIM


def _features(tokens: List[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for i, tok in enumerate(tokens):
        feats[f"1:{tok}"] = feats.get(f"1:{tok}", 0.0) + 1.0
        if i + 1 < len(tokens):
            key = f"2:{tok}_{tokens[i + 1]}"
            feats[key] = feats.get(key, 0.0) + 0.6
        if i + 2 < len(tokens):
            key = f"3:{tok}_{tokens[i + 1]}_{tokens[i + 2]}"
            feats[key] = feats.get(key, 0.0) + 0.3
    return feats


def embed_sparse(text: str, dim: int = DIM) -> Dict[int, float]:
    """Sparse {bucket: weight}. Sublinear term weighting, L2-normalized."""
    tokens = tokenize(text)
    if not tokens:
        return {}
    raw = [0.0] * dim
    for tok, base in _features(tokens).items():
        raw[_bucket(tok)] += 1.0 + math.log(base)
    norm = math.sqrt(sum(v * v for v in raw))
    if norm <= 0:
        return {}
    return {i: v / norm for i, v in enumerate(raw) if v > 1e-9}


def embed(text: str, dim: int = DIM) -> List[float]:
    vec = [0.0] * dim
    for bucket, weight in embed_sparse(text, dim).items():
        vec[bucket] = weight
    return vec


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        if x and y:
            s += x * y
    return s


def cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    s = 0.0
    for bucket, weight in b.items():
        if bucket in a:
            s += a[bucket] * weight
    return s
