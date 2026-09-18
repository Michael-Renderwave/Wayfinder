"""Wayfinder server — FastAPI, zero required dependencies beyond fastapi/uvicorn.

Run:  python app.py   ->   http://localhost:8000
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wayfinder.agent import Agent
from wayfinder.llm import LLM
from wayfinder.planner import HeuristicPlanner, parse_goal
from wayfinder.world import effective_home
from wayfinder.reqlog import count_requests, log_request, recent_requests
from wayfinder.ranking import rank
from wayfinder.tools import advanced_search_tool, search_flights, search_hotels
from wayfinder.vectorstore import VectorStore

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
DATA = os.path.join(ROOT, "data")
INDEX = os.path.join(DATA, "index.json")
KB_DIR = os.path.join(DATA, "kb")

app = FastAPI(title="Wayfinder", version="1.0.0")


@app.middleware("http")
async def no_cache_frontend(request: Request, call_next):
    """Never let a browser/webview serve a STALE app.js/style.css/index.html —
    an old cached bundle silently drops new answer blocks (quick report, photos,
    trip bottom line) because the fields exist in the event but the old JS has
    no renderer for them. The page itself revalidates on every load."""
    response = await call_next(request)
    if request.url.path in ("/", "/index.html", "/app.js", "/style.css"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


def region_hint_for(query: str) -> str:
    """Derive a region hint from the raw query for direct (non-agent) searches,
    so the same bias guard applies: a 'kennywood discounts' search must not
    surface Cayman chunks (CP 3.1: prevent off-topic data poisoning results)."""
    dest = parse_goal(query).get("destination") or ""
    if not dest:
        return ""
    return " ".join([dest, HeuristicPlanner.DEST_CONTEXT.get(dest, "")])


def home_for(query: str) -> str:
    """Destination home for the ranking HARD bias guard (CP 3.1). '' for
    curated demos / unknown destinations so those stay on the soft path."""
    return effective_home(parse_goal(query).get("destination") or "")


# ----------------------------------------------------------------------
# ingestion (CP 3.1: chunk -> embed -> vector store)
# ----------------------------------------------------------------------
def parse_front_matter(text: str):
    meta = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = parts[2]
    return meta, body


def build_store() -> VectorStore:
    store = VectorStore()
    if not os.path.isdir(KB_DIR):
        return store
    for fname in sorted(os.listdir(KB_DIR)):
        if not fname.endswith((".md", ".txt")):
            continue
        path = os.path.join(KB_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        meta, body = parse_front_matter(raw)
        meta = {
            "id": meta.get("id", fname),
            "title": meta.get("title", fname),
            "source": meta.get("source", meta.get("title", fname)),
            "tier": meta.get("tier", "secondary"),
            "date": meta.get("date", ""),
            "url": meta.get("url", ""),
            "category": meta.get("category", "general"),
            "region": meta.get("region", ""),
        }
        store.add_document(meta, body.strip())
    return store


def get_agent() -> Agent:
    if not os.path.exists(INDEX):
        store = build_store()
        store.save(INDEX)
    else:
        store = VectorStore.load(INDEX)
    return Agent(store, DATA, LLM())


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------
class QueryBody(BaseModel):
    query: str
    mode: str = "agent"  # agent | direct
    k: int = 8
    # Home base (top-of-page setting): {"country": "United States", "region": "Nevada"}
    # -> flights anchor on the user's own gateway (Las Vegas/LAS, Turkey -> IST),
    # never on random US hubs. Null/absent = the standard no-origin behavior.
    home: Optional[Dict[str, str]] = None


class SearchBody(BaseModel):
    query: str
    k: int = 8


def _search_log(query: str, results, pruned) -> None:
    """Log a raw search (direct /api/query or /api/search). Never raises."""
    try:
        log_request(DATA, {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": "search",
            "query": query,
            "kept": len(results),
            "pruned": len(pruned),
            "status": "ok",
        })
    except Exception:
        pass


class ResearchBody(BaseModel):
    """Source-aware public discovery request (results are leads, not facts)."""
    query: str
    k_per_source: int = 3


@app.get("/api/airports")
def airports_meta():
    """Home-base dropdown data: countries + their regions from data/airports.json
    (47 countries, key-free Wikipedia build — see scripts/fetch_airports.py).
    The UI sends {country, region?} back as QueryBody.home."""
    from wayfinder import airports
    return {
        "countries": [
            {"name": name, "regions": airports.regions(name)}
            for name in airports.country_list()
        ],
    }


@app.get("/api/health")
def health():
    agent = get_agent()
    web_stats = {k: v for k, v in agent.web_store.stats().items() if k != "doc_list"}
    return {
        "ok": True,
        "mode": agent.mode,
        "llm": agent.llm.describe(),
        "kb": agent.store.stats(),
        "web": web_stats,  # live-web crawl store (2nd vector index, CP 3.1 provenance)
    }


@app.post("/api/query")
async def query(body: QueryBody, request: Request):
    """SSE stream of ReAct events: memory, thought, action, observation, prune, answer, done.

    CP 6.1 L6 (human-intervention plan): no HITL is required — the guardrails keep the
    agent in check autonomously — but the user keeps control: if the client disconnects
    (the Stop button aborts the fetch), the stream is cancelled and marked as such."""

    async def event_stream():
        if body.mode == "direct":
            store = get_agent().store
            today = date.today()
            raw = store.search(body.query, k=body.k)
            ranked = rank(raw, today, query=body.query, region_hint=region_hint_for(body.query),
                         home_country=home_for(body.query))
            survivors = [s for s in ranked if not s.pruned]
            payload = {
                "results": [
                    {
                        "id": s.chunk.id, "title": s.chunk.title, "text": s.chunk.text,
                        "source": s.chunk.source, "tier": s.chunk.tier, "date": s.chunk.date,
                        "url": s.chunk.url,
                        "scores": {"relevance": round(s.relevance, 4), "reliability": round(s.reliability, 3),
                                   "recency": round(s.recency, 3), "phrase": round(s.phrase, 3),
                                   "total": round(s.score, 4)},
                    }
                    for s in survivors
                ],
                "pruned": [{"id": s.chunk.id, "title": s.chunk.title, "score": round(s.score, 4),
                            "reason": s.prune_reason} for s in ranked if s.pruned],
            }
            _search_log(body.query, payload["results"], payload["pruned"])
            yield f"data: {json.dumps({'type': 'search_results', **payload})}\n\n"
            yield "data: " + json.dumps({"type": "done", "mode": "direct"}) + "\n\n"
            return
        agent = get_agent()
        # Request log (CP 2.1 + logging): data/request_log.jsonl is the single source
        # of truth. We capture run stats as they stream by and write one record in
        # `finally` — even on cancellation or a mid-stream error, logging never blocks
        # or breaks the stream (log_request swallows I/O errors).
        log_entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": "agent",
            "query": body.query,
            "status": "ok",
        }
        captured = {}
        t0 = time.time()
        try:
            for ev in agent.run(body.query, home=body.home):
                if await request.is_disconnected():
                    log_entry["status"] = "cancelled"
                    yield ("data: " + json.dumps({
                        "type": "cancelled", "ts": time.strftime("%H:%M:%S"),
                        "note": "stream cancelled by the user (CP 6.1 L6 — no HITL, but you keep control)",
                    }, ensure_ascii=False) + "\n\n")
                    break
                etype = ev.get("type") if isinstance(ev, dict) else None
                if etype == "answer":
                    captured["confidence_pct"] = ev.get("confidence_pct")
                    captured["confidence_label"] = ev.get("confidence_label")
                    d = (ev.get("goal") or {}).get("destination") or ""
                    if d:
                        captured["destination"] = d
                elif etype == "done":
                    captured["steps"] = ev.get("steps")
                    captured["searches"] = ev.get("searches")
                    captured["pruned"] = ev.get("pruned")
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            log_entry["elapsed_ms"] = int((time.time() - t0) * 1000)
            log_entry.update(captured)
            try:
                log_request(DATA, log_entry)
            except Exception:
                pass

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/search")
def direct_search(body: SearchBody):
    """Raw hard search, no agent loop."""
    store = get_agent().store
    today = date.today()
    raw = store.search(body.query, k=body.k)
    ranked = rank(raw, today, query=body.query, region_hint=region_hint_for(body.query),
                 home_country=home_for(body.query))
    results = [
        {
            "id": s.chunk.id, "title": s.chunk.title, "text": s.chunk.text,
            "source": s.chunk.source, "tier": s.chunk.tier, "date": s.chunk.date, "url": s.chunk.url,
            "scores": {"relevance": round(s.relevance, 4), "reliability": round(s.reliability, 3),
                       "recency": round(s.recency, 3), "phrase": round(s.phrase, 3),
                       "total": round(s.score, 4)},
        }
        for s in ranked if not s.pruned
    ]
    pruned = [{"id": s.chunk.id, "title": s.chunk.title, "score": round(s.score, 4),
               "reason": s.prune_reason} for s in ranked if s.pruned]
    _search_log(body.query, results, pruned)
    return {"results": results, "pruned": pruned}


@app.get("/api/requests")
def requests_log(n: int = 200):
    """Request log (CP 2.1 + logging) — the append-only audit trail, newest first.

    `n` is clamped to 1..500; `total` is the full log length (may exceed the
    returned window). This is the server-side history the agent reads at run
    start and the Memory tab renders — one source of truth, not a copy.
    """
    try:
        n = max(1, min(int(n), 500))
    except (TypeError, ValueError):
        n = 200
    return {"total": count_requests(DATA), "requests": recent_requests(DATA, n)}


@app.post("/api/research")
def research(body: ResearchBody):
    """Structured cross-source research leads for API clients and advanced UIs.

    The same federated discovery the agent uses in its loop (advanced_search tool):
    public indexed pages from Reddit, Instagram, Tripadvisor, Expedia, Wikivoyage,
    and Wikipedia — leads, not facts (Reddit/Instagram are never fetched).
    """
    result, observation = advanced_search_tool(body.query, max(1, min(body.k_per_source, 5)))
    return {"observation": observation, **result}


@app.get("/api/kb")
def kb():
    return get_agent().store.stats()


@app.post("/api/kb/reindex")
def reindex():
    store = build_store()
    store.save(INDEX)
    return {"reindexed": True, **store.stats()}


@app.get("/api/memory")
def memory():
    agent = get_agent()
    return {"episodic": agent.memory.list(),
            "semantic": agent.memory.context()[-6:],
            "bookings": agent.bookings}


@app.get("/api/flights")
def flights():
    agent = get_agent()
    return {"flights": agent.flights_db}


@app.get("/img")
def img_proxy(url: str):
    """Same-origin proxy for the Commons photo thumbnails (renderPhotos).
    Some in-app webviews block third-party image loads, so the browser fetches
    from localhost and the server fetches the bytes instead — through the
    SSRF-guarded ``http_get`` (public http(s) only, size-capped, 429 retry).
    The cited/linked URL remains the original Wikimedia Commons page, so
    provenance is unchanged (CP 3.1)."""
    from wayfinder.web import WebError, http_get
    try:
        data = http_get(url, max_bytes=1_500_000)
    except WebError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    ctype = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return Response(content=data, media_type=ctype)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


# catch-all: serve the vanilla-JS frontend (index.html, app.js, style.css)
# from the site root; explicit routes above still take precedence.
app.mount("/", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    # pre-build index on first run
    get_agent()
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),  # 0.0.0.0 inside Docker
        port=int(os.environ.get("PORT", "8000")),
        log_level="warning",
    )
