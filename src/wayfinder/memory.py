"""Memory (CP 2.1):
  - Semantic memory:  short-term conversation context, re-packaged into each
    new planning step ("it resends that information, and packages it into the
    current prompt to reestablish a correlation").
  - Episodic memory:  long-term log of tool calls, executions and database
    lookups ("Tool calling, executions, and database lookups can all be stored
    inside the agent's long-term episodic memory"), retrieved again with a
    cache ("retrieved again with a cache").
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class EpisodicRecord:
    ts: str
    query: str
    destination: str
    actions: List[dict] = field(default_factory=list)
    top_sources: List[str] = field(default_factory=list)
    summary: str = ""
    cache_key: str = ""


class Memory:
    def __init__(self, path: str, semantic_window: int = 6):
        self.path = path
        self.semantic_window = semantic_window
        self.episodic: List[EpisodicRecord] = []
        self.semantic: List[dict] = []  # {"role": "user"|"agent", "text": str}
        self._load()

    # ---- persistence ----------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.episodic = [EpisodicRecord(**r) for r in payload.get("episodic", [])]
            self.semantic = payload.get("semantic", [])
        except (json.JSONDecodeError, TypeError):
            self.episodic, self.semantic = [], []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "episodic": [asdict(r) for r in self.episodic],
            "semantic": self.semantic[-self.semantic_window * 2 :],
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

    # ---- semantic -------------------------------------------------------
    def remember_turn(self, user_text: str, agent_text: str) -> None:
        self.semantic.append({"role": "user", "text": user_text})
        self.semantic.append({"role": "agent", "text": agent_text})
        self.semantic = self.semantic[-self.semantic_window * 2 :]

    def context(self) -> List[dict]:
        return list(self.semantic)

    # ---- episodic -------------------------------------------------------
    def record(self, rec: EpisodicRecord) -> None:
        self.episodic.append(rec)
        self.episodic = self.episodic[-50:]
        self.save()

    def recall(self, query: str, destination: str) -> Optional[EpisodicRecord]:
        """Cache lookup (CP 2.1): if we already ran a search for the same
        destination+intent, the cached lookups can be reused."""
        q_tokens = {w for w in query.lower().replace("-", " ").split()}
        best, best_overlap = None, 0.0
        for rec in self.episodic:
            if rec.destination and rec.destination != destination:
                continue
            r_tokens = set(rec.query.lower().replace("-", " ").split())
            if not r_tokens:
                continue
            overlap = len(q_tokens & r_tokens) / max(1, min(len(q_tokens), len(r_tokens)))
            if overlap > best_overlap and overlap >= 0.6:
                best, best_overlap = rec, overlap
        return best

    def list(self) -> List[dict]:
        return [asdict(r) for r in reversed(self.episodic[-20:])]
