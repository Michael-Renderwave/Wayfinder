"""Request log — append-only audit trail of every search/query request (CP 2.1 + logging).

`data/request_log.jsonl` is the SINGLE source of truth for request history:
  - written by the server on every /api/query (agent + direct) and /api/search call
  - read by the /api/requests endpoint (UI memory tab + programmatic access)
  - read by the agent at run start (history-aware memory events: "you asked about
    this before — carrying that context forward")

Append-only JSONL: survives restarts, is never rewritten, trivially greppable,
and grows cheaply (one small line per request). Thread-safe appends.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List

_LOCK = threading.Lock()


def _path(data_dir: str) -> str:
    return os.path.join(data_dir, "request_log.jsonl")


def log_request(data_dir: str, entry: Dict[str, Any]) -> None:
    """Append one request record (never raises into the request path)."""
    try:
        os.makedirs(data_dir, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _LOCK:
            with open(_path(data_dir), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        pass  # logging must never break a search


def recent_requests(data_dir: str, n: int = 200) -> List[Dict[str, Any]]:
    """Last n request records, NEWEST FIRST. Tolerates partial/corrupt lines."""
    path = _path(data_dir)
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    n = max(1, int(n))
    return list(reversed(out[-n:]))


def count_requests(data_dir: str) -> int:
    """Total number of logged requests (accurate even beyond any read window)."""
    path = _path(data_dir)
    if not os.path.exists(path):
        return 0
    n = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    except OSError:
        return 0
    return n
