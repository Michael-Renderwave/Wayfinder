"""Vector store + hard search (CP 2.1 top-K similarity, CP 3.1 vector store).

Chunks are stored as sparse hashed vectors with provenance metadata
(source, tier, date, url) so every result is traceable — the paper's core
RAG requirement: "data that has a traceable origin as opposed to arbitrary
reasoning".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .embeddings import embed_sparse, cosine_sparse, DIM
from .splitter import recursive_split

# 'web' = live-web pages fetched this session: unverified by design, so they rank
# between secondary and tertiary (CP 3.1 bias guard: don't trust unvetted sources fully)
TIER_WEIGHT = {"primary": 1.0, "secondary": 0.7, "web": 0.5, "tertiary": 0.35}


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    title: str
    source: str
    tier: str  # primary | secondary | tertiary
    date: str  # ISO date of publication
    url: str
    category: str
    region: str = ""
    vector: Dict[int, float] = field(repr=False, default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["vector"] = {str(k): round(v, 6) for k, v in self.vector.items()}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Chunk":
        d = dict(d)
        d["vector"] = {int(k): v for k, v in d.get("vector", {}).items()}
        d.setdefault("region", "")
        return cls(**d)


@dataclass
class ScoredChunk:
    chunk: Chunk
    relevance: float
    reliability: float
    recency: float
    phrase: float
    score: float
    pruned: bool = False
    prune_reason: str = ""


class VectorStore:
    def __init__(self, chunks: Optional[List[Chunk]] = None):
        self.chunks: List[Chunk] = chunks or []
        self.docs: Dict[str, dict] = {}
        self._by_doc: Dict[str, List[Chunk]] = {}

    # ---- build ---------------------------------------------------------
    def add_document(self, doc_meta: dict, text: str, chunk_size: int = 700, chunk_overlap: int = 150) -> int:
        doc_id = doc_meta["id"]
        self.docs[doc_id] = doc_meta
        buckets = []
        for i, piece in enumerate(recursive_split(text, chunk_size, chunk_overlap)):
            chunk = Chunk(
                id=f"{doc_id}::{i}",
                doc_id=doc_id,
                text=piece,
                title=doc_meta.get("title", doc_id),
                source=doc_meta.get("source", doc_meta.get("title", "unknown")),
                tier=doc_meta.get("tier", "secondary").lower(),
                date=doc_meta.get("date", ""),
                url=doc_meta.get("url", ""),
                category=doc_meta.get("category", "general"),
                region=doc_meta.get("region", ""),
                vector=embed_sparse(f"{doc_meta.get('title', '')} {piece}"),
            )
            self.chunks.append(chunk)
            buckets.append(chunk)
        self._by_doc[doc_id] = buckets
        return len(buckets)

    # ---- persist -------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "dim": DIM,
            "docs": self.docs,
            "chunks": [c.to_json() for c in self.chunks],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        store.docs = payload.get("docs", {})
        store.chunks = [Chunk.from_json(c) for c in payload.get("chunks", [])]
        for c in store.chunks:
            store._by_doc.setdefault(c.doc_id, []).append(c)
        return store

    # ---- search --------------------------------------------------------
    def search(self, query: str, k: int = 8, min_relevance: float = 0.0,
               anchor: str = "") -> List[ScoredChunk]:
        """Top-K similarity candidates. `anchor` (the canonical destination,
        e.g. 'Brazil') guarantees the destination's own documents reach the
        candidate pool even when the country name is only 1 of N query words
        and raw cosine dilutes it out of the top-k*2."""
        qvec = embed_sparse(query)
        if not qvec:
            return []
        scored: List[ScoredChunk] = []
        for c in self.chunks:
            rel = cosine_sparse(qvec, c.vector)
            if rel < min_relevance:
                continue
            scored.append(ScoredChunk(chunk=c, relevance=rel, reliability=0.0, recency=0.0, phrase=0.0, score=0.0))
        scored.sort(key=lambda s: s.relevance, reverse=True)
        pool = scored[: k * 2]  # return extra candidates for the ranking rubric
        if anchor:
            import re as _re
            tokens = {t for t in _re.findall(r"[a-z0-9]+", anchor.lower()) if len(t) > 2}
            if tokens:
                pool_ids = {s.chunk.id for s in pool}
                for s in scored[k * 2:]:
                    hay = {t for t in _re.findall(r"[a-z0-9]+", f"{s.chunk.region} {s.chunk.title}")}
                    if tokens & hay:
                        pool.append(s)
        return pool

    def chunks_of(self, doc_id: str) -> List[Chunk]:
        """All chunks belonging to one document, in insertion order."""
        return self._by_doc.get(doc_id, [])

    def trim_to(self, max_docs: int) -> int:
        """Cap the store at max_docs (most recently added win); drop older docs
        and their chunks. Returns how many documents were dropped."""
        if len(self.docs) <= max_docs:
            return 0
        drop = list(self.docs)[: -max_docs]
        for d in drop:
            del self.docs[d]
            self._by_doc.pop(d, None)
        keep = set(self.docs)
        self.chunks = [c for c in self.chunks if c.doc_id in keep]
        return len(drop)

    def stats(self) -> dict:
        return {
            "documents": len(self.docs),
            "chunks": len(self.chunks),
            "dimensions": DIM,
            "doc_list": [
                {
                    "id": d["id"],
                    "title": d.get("title"),
                    "tier": d.get("tier"),
                    "date": d.get("date"),
                    "source": d.get("source"),
                    "url": d.get("url"),
                    "category": d.get("category"),
                    "chunks": len(self._by_doc.get(d["id"], [])),
                }
                for d in self.docs.values()
            ],
        }
