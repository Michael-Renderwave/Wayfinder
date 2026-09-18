"""CP 6.1 — Safety Guardrails and Human Intervention Plan (Section B).

The failure mode this module fends off: an agent that confidently serves the WRONG
information — "Chipotle Mexican Grill in Quito" for "where can I go to eat in Rio de
Janeiro?". A wrong answer that sounds right poisons the user's trust and, for travel,
their decisions. The plan is a mostly-autonomous system — **no HITL, no escalation
queue** — protected by multiple cheap guardrail layers, balanced so they keep the
agent in check **without slowing it to a halt** (no extra network round-trips, no new
dependencies, everything local and deterministic):

  L1  Filtered input     — incomplete queries are clarified BEFORE any tool work;
                           the agent never guesses a missing destination
  L2  Scoped permissions — each planner "job" may only call a fixed tool set over a
                           fixed data scope (local KB + flights/hotels; the web crawl
                           store when the web is enabled — on by default, WAYFINDER_WEB=0 disables it); out-of-scope calls are
                           denied with a logged reason
  L3  Validation + rate limits — every tool call is argument-validated and budgeted
                           per run; rejected calls stop after a circuit breaker
  L4  Output scoring     — every answer gets a deterministic confidence score + band
                           (the pass/fail tolerance), shown to the user for
                           reassurance; unverified web content can never reach High
  L5  Human-readable log — a `guardrail` summary event closes every run
  L6  User cancellation  — no human-in-the-loop is required, but the user can always
                           Stop a running search (enforced in app.py + the frontend)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# L2 — scoped permissions: restricting access to specific databases and tool calls
# ---------------------------------------------------------------------------
# Each planner "job" occupies a bounded role (CP 6.1: "agents are bound by a set of
# rules for each 'job' they can occupy at any time"). `compose_answer` is the loop
# terminator, not a tool — implicitly permitted to every job.
WEB_TOOLS = frozenset({"web_search", "web_fetch", "advanced_search", "local_places", "travel_safety",
                       "luxury"})  # local_places = OSM over the internet; travel_safety = official advisories over the internet; luxury = OSM + Wikipedia API over the internet

PERMITTED_TOOLS: Dict[str, frozenset] = {
    "heuristic": frozenset({
        "search_kb", "search_flights", "search_hotels", "book_ticket", "lookup",
        "flight_info", "best_time",
    }),
    "llm": frozenset({
        "search_kb", "search_flights", "search_hotels", "book_ticket", "lookup",
        "flight_info", "best_time",
    }),
}
# The web-research tools are in scope for EVERY job only when the deployment opted in
# (on by default; WAYFINDER_WEB=0 switches to offline mode): both planners emit the same live-web path (CP 2.1), so the
# scope follows the deployment flag, not the planner mode. With the web off, the
# web tools are out of scope for every job and are denied here — RunBudget denies
# them too as a second layer (CP 6.1 L3).

# Data scope per job (documented in the L5 log so the trace shows WHAT the job may
# touch — the tools above are the enforcement, this is the human-readable record).
DATA_SCOPES: Dict[str, List[str]] = {
    "heuristic": [
        "local KB (data/index.json)", "flights.json", "hotels.json", "bookings.json",
        "key-free Wikipedia geocode + IATA lookup (flight_info)",
        "flightconnections.com route pages (live schedule data for flight_info, key-free)",
        "seasons dataset (data/seasons.json — key-free climate normals + holidays for best_time)",
        "U.S. State Department travel advisory (travel_safety — key-free: live official page via "
        "search + official data snapshot, data/safety_cache.json)",
    ],
    "llm": [
        "local KB (data/index.json)", "flights.json", "hotels.json", "bookings.json",
        "key-free Wikipedia geocode + IATA lookup (flight_info)",
        "flightconnections.com route pages (live schedule data for flight_info, key-free)",
        "seasons dataset (data/seasons.json — key-free climate normals + holidays for best_time)",
        "U.S. State Department travel advisory (travel_safety — key-free: live official page via "
        "search + official data snapshot, data/safety_cache.json)",
    ],
}


def permitted_tools(mode: str, web_enabled: bool = False) -> frozenset:
    tools = PERMITTED_TOOLS.get(mode, PERMITTED_TOOLS["heuristic"])
    if web_enabled:
        tools = tools | WEB_TOOLS
    return tools


def data_scope(mode: str, web_enabled: bool = False) -> List[str]:
    scope = list(DATA_SCOPES.get(mode, DATA_SCOPES["heuristic"]))
    if web_enabled and not any("web crawl" in s for s in scope):
        scope.append("web crawl store (data/web_index.json) — on by default (WAYFINDER_WEB=0 disables)")
    return scope


def check_permission(mode: str, action: str, web_enabled: bool = False) -> Tuple[bool, str]:
    """L2 gate: may this planner job call this tool at all?"""
    if action == "compose_answer":
        return True, "loop terminator (implicit)"
    if action in permitted_tools(mode, web_enabled):
        return True, "in scope"
    extra = " (web tools need WAYFINDER_WEB=1 — it is off in this deployment)" if action in WEB_TOOLS and not web_enabled else ""
    return False, (
        f"'{action}' is not available in this mode — "
        f"allowed: {', '.join(sorted(permitted_tools(mode, web_enabled)))}{extra}"
    )


# ---------------------------------------------------------------------------
# L3 — validation + rate limits
# ---------------------------------------------------------------------------
def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _short_str(v: Any, maxlen: int) -> bool:
    return isinstance(v, str) and 0 < len(v.strip()) <= maxlen


def validate_action(action: str, args: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """L3 gate: validate a tool call's arguments BEFORE dispatch.

    Returns (ok, reason, normalized_args). Ranges are clamped rather than rejected
    where harmless (k=999 -> k=12) so a sloppy call still degrades to a valid one;
    wrong *types* are rejected outright (never guess intent, CP 6.1 L1 spirit).
    """
    a = dict(args) if isinstance(args, dict) else {}
    if action == "search_kb":
        if not _short_str(a.get("query"), 200):
            return False, "search_kb: 'query' must be a non-empty string (<=200 chars)", a
        k = a.get("k", 8)
        if not _is_int(k):
            return False, "search_kb: 'k' must be an integer", a
        a["k"] = max(1, min(k, 12))
    elif action == "search_flights":
        for key in ("origin", "destination"):
            if key in a and not (a[key] is None or (isinstance(a[key], str) and len(a[key]) <= 30)):
                return False, f"search_flights: '{key}' must be a short airport code / string", a
        if a.get("max_price") is not None and not _is_int(a["max_price"]):
            return False, "search_flights: 'max_price' must be an integer or null", a
    elif action == "flight_info":
        for key in ("origin", "destination"):
            if key in a and not (a[key] is None or (isinstance(a[key], str) and len(a[key]) <= 80)):
                return False, f"flight_info: '{key}' must be a short string (<=80 chars)", a
    elif action == "search_hotels":
        if "destination" in a and a["destination"] is not None \
                and not (isinstance(a["destination"], str) and len(a["destination"]) <= 80):
            return False, "search_hotels: 'destination' must be a string (<=80 chars)", a
        if a.get("max_price") is not None and not _is_int(a["max_price"]):
            return False, "search_hotels: 'max_price' must be an integer or null", a
    elif action == "best_time":
        if "destination" in a and a["destination"] is not None \
                and not (isinstance(a["destination"], str) and len(a["destination"]) <= 80):
            return False, "best_time: 'destination' must be a string (<=80 chars)", a
        if a.get("month") is not None:
            if not _is_int(a["month"]) or not (1 <= a["month"] <= 12):
                return False, "best_time: 'month' must be an integer 1-12 (or null)", a
    elif action == "book_ticket":
        if "flight" in a and not isinstance(a["flight"], dict):
            return False, "book_ticket: 'flight' must be an object (a concrete flight)", a
    elif action == "lookup":
        if not _short_str(a.get("entity"), 120):
            return False, "lookup: 'entity' must be a non-empty string (<=120 chars)", a
    elif action == "web_search":
        if not _short_str(a.get("query"), 150):
            return False, "web_search: 'query' must be a non-empty string (<=150 chars)", a
        k = a.get("k", 5)
        if not _is_int(k):
            return False, "web_search: 'k' must be an integer", a
        a["k"] = max(1, min(k, 10))
    elif action == "web_fetch":
        url = a.get("url")
        if not (isinstance(url, str) and re.match(r"^https?://", url, re.I) and len(url) <= 1000):
            return False, "web_fetch: 'url' must be an http(s) URL (<=1000 chars)", a
    elif action == "advanced_search":
        if not _short_str(a.get("query"), 150):
            return False, "advanced_search: 'query' must be a non-empty string (<=150 chars)", a
        k = a.get("k_per_source", 3)
        if not _is_int(k):
            return False, "advanced_search: 'k_per_source' must be an integer", a
        a["k_per_source"] = max(1, min(k, 5))  # clamp like /api/research (min 1, max 5)
    elif action == "local_places":
        if not _short_str(a.get("place"), 120):
            return False, "local_places: 'place' must be a non-empty string (<=120 chars)", a
        f = a.get("focus", "dining")
        # Keep in sync with osm.FOCUS_FILTERS (wayfinder/osm.py) and the LLM tool
        # schema in planner.py — the model is told all eight foci are valid, so
        # denying any of them is a silent feature-loss (the safety pass —
        # hospitals/pharmacies + police/fire — was being dropped exactly this way).
        if not (isinstance(f, str) and f in {"dining", "attractions", "hotels", "health",
                                             "luxury", "authorities", "safety", "all"}):
            return False, ("local_places: 'focus' must be one of "
                           "dining|attractions|hotels|health|luxury|authorities|safety|all"), a
        k = a.get("k", 12)
        if not _is_int(k):
            return False, "local_places: 'k' must be an integer", a
        a["k"] = max(2, min(k, 24))  # bounded pulls: polite to the key-free OSM services
    elif action == "travel_safety":
        if not _short_str(a.get("place"), 120):
            return False, "travel_safety: 'place' must be a non-empty string (<=120 chars)", a
    elif action == "luxury":
        d = a.get("destination", "")
        if not (isinstance(d, str) and len(d) <= 120):
            return False, "luxury: 'destination' must be a string (<=120 chars; empty = global best-places view)", a
    else:
        return False, f"unknown tool '{action}' — not available", a
    return True, "ok", a


# Per-run tool-call budget (CP 6.1 L3: "the tool calls are validated and rate-limited
# to stop overwhelming hits that may occur during search"). The planner already
# self-limits; this is the HARD second layer, so neither planner mode can exceed it.
# Values mirror the documented caps (MAX_PASSES depth guard, WEB_MAX_PASSES web depth).
TOOL_BUDGET: Dict[str, int] = {
    "search_kb": 4,        # depth guard (CP 4.1: depth 3-4)
    "search_flights": 1,
    "search_hotels": 1,
    "book_ticket": 1,      # PENDING per request (CP 1.1)
    "lookup": 2,
    "flight_info": 1,      # one pricing/duration pass per run (estimate or schedule)
    "best_time": 1,        # one seasons-dataset pass per run (deterministic local lookup)
    "web_search": 2,       # pass 1 = the place, pass 2 = facts/history (CP 2.1 depth)
    "web_fetch": 3,        # 2 pages + 1 fallback (CP 1.1)
    "advanced_search": 1,  # one federated discovery pass per run (CP 1.1 breadth cap)
    "local_places": 1,     # one OSM radius pass per run (key-free; 7-day disk cache)
    "travel_safety": 1,    # one advisory lookup per run (live page + 7-day snapshot cache)
    "luxury": 1,           # one luxury pass per run (OSM 4-5★/experiences + Wikipedia three-star; 30-day cache)
}

# Circuit breaker (CP 6.1 L3): this many REJECTED calls in a row halts the run and
# composes from what we have — a broken planner must not spin until MAX_STEPS.
MAX_DENIED = 3


class RunBudget:
    """Per-run rate-limit ledger (L3) + denial accounting (L5 log)."""

    def __init__(self, web_enabled: bool):
        self.web_enabled = web_enabled
        self.used: Dict[str, int] = {}
        self.denied_total = 0
        self.denied_streak = 0
        self.invalid_total = 0

    def allow(self, action: str) -> Tuple[bool, str]:
        if action == "compose_answer":
            return True, "ok"
        cap = TOOL_BUDGET.get(action)
        if cap is None:
            return False, f"'{action}' has no tool-call budget (not a permitted tool)"
        if not self.web_enabled and action in WEB_TOOLS:
            return False, "web research is disabled in this deployment (offline-first mode)"
        if self.used.get(action, 0) >= cap:
            return False, f"rate limit: '{action}' already used {self.used[action]}/{cap} this run"
        return True, f"budget ok ({self.used.get(action, 0)}/{cap} used)"

    def consume(self, action: str) -> None:
        if action in TOOL_BUDGET:
            self.used[action] = self.used.get(action, 0) + 1
        self.denied_streak = 0

    def record_denied(self, reason: str) -> None:
        self.denied_total += 1
        self.denied_streak += 1
        if "validation" in reason or "no validation schema" in reason or "must be" in reason:
            self.invalid_total += 1

    @property
    def halted(self) -> bool:
        return self.denied_streak >= MAX_DENIED

    def summary(self) -> str:
        parts = []
        for tool, cap in TOOL_BUDGET.items():
            if tool in WEB_TOOLS and not self.web_enabled:
                continue
            parts.append(f"{tool} {self.used.get(tool, 0)}/{cap}")
        parts.append(f"denied {self.denied_total}")
        if self.invalid_total:
            parts.append(f"invalid {self.invalid_total}")
        return " · ".join(parts)


def check_call(mode: str, action: str, args: Any, budget: RunBudget) -> Tuple[bool, str, Dict[str, Any]]:
    """One-stop L2+L3 gate used by the agent dispatch: permission -> validation -> budget.

    Returns (ok, reason, normalized_args)."""
    if action == "compose_answer":
        return True, "ok", args
    ok, reason = check_permission(mode, action, budget.web_enabled)
    if not ok:
        return False, reason, args
    ok, reason, norm = validate_action(action, args)
    if not ok:
        return False, reason + " (input check)", norm
    ok, reason = budget.allow(action)
    if not ok:
        return False, reason + " (fair-use limit)", norm
    return True, "ok", norm


# ---------------------------------------------------------------------------
# L1 — filtered input: clarify queries with incomplete information before acting
# ---------------------------------------------------------------------------
# Bare acknowledgements are NOT incomplete input — they continue the previous
# turn (e.g. 'confirm' after a PENDING booking), so they pass the filter and the
# existing conversation flow handles them.
_ACK_WORDS = {"confirm", "confirmed", "yes", "y", "yeah", "yep", "ok", "okay", "sure", "go", "book it"}

_TRIP_INTENT = re.compile(
    r"\b(trip|vacation|itinerary|getaway|flight|flights|fly|ticket|tickets|book|booking"
    r"|hotel|hotels|stay|stays|lodge|inn|accommodation|discount|resort)\b",
    re.I,
)


def check_input(query: str, goal: Dict[str, Any], prev_dest: str = "") -> Optional[Dict[str, str]]:
    """L1 — the first line of defense (CP 6.1): monitor filtered input.

    Returns {'reason', 'question'} when the query is incomplete and must be
    clarified BEFORE any tool work, else None. Cheap local checks only — no
    network, no LLM — so clarification adds zero latency to the search itself.
    """
    text = (query or "").strip()
    words = re.findall(r"[a-z0-9$]+", text.lower())
    if not words:
        return {
            "reason": "empty query — nothing to act on",
            "question": (
                "What would you like me to research? Tell me a destination — and dates or a "
                "budget if you have them — e.g. 'plan a 5 day trip to the Cayman Islands from Miami'."
            ),
        }
    if len(words) == 1 and words[0] in _ACK_WORDS:
        return None  # continues the previous turn — the conversation flow handles it
    has_dest = bool(goal.get("destination")) or bool(prev_dest)
    if len(words) < 2 or len(text) < 8:
        return {
            "reason": "query too short to act on safely (1 word / <8 chars) — clarifying instead of guessing",
            "question": (
                "I want to be sure I research the right thing — which destination are we talking about? "
                "I have guides for 197 countries, plus deep dives on the Cayman Islands / George Town and "
                "Kennywood / Pittsburgh — or name any place and I'll do my best with what I have on hand."
            ),
        }
    # LUXURY focus: a destination-less query is a VALID, complete answer — the honest
    # global best-places view ('where are the best places for a luxury trip?'). The
    # other focuses all need a place, so they still clarify below (CP 1.1: show the
    # honest global view rather than force a destination the user never named).
    if "luxury" in (goal.get("focus") or []):
        return None
    if not has_dest:
        if _TRIP_INTENT.search(text):
            reason = "trip/stay/booking intent without a destination — clarifying instead of guessing"
        else:
            reason = "no destination in the query or our conversation yet — clarifying instead of guessing"
        return {
            "reason": reason,
            "question": (
                "Which destination would you like me to research? I have guides for 197 countries, "
                "plus deep dives on the Cayman Islands / George Town and Kennywood / Pittsburgh — "
                "or name any place and I'll do my best with what I have on hand."
            ),
        }
    return None


# ---------------------------------------------------------------------------
# L4 — output scoring: confidence + pass/fail tolerance bands
# ---------------------------------------------------------------------------
# "Outputs are scored for the most confident output" (CP 6.1) — and the user gets
# the certainty level for reassurance ("how much certainty can information be
# valid?"). Bands are the pass/fail tolerance; the thresholds are deliberately
# simple local math (no extra LLM calls), and unverified web content can NEVER
# lift an answer into the High band (CP 3.1 bias guard: "bias and inaccurate data
# sources can poison the final results").
CONF_HIGH = 0.70      # >= 70%  -> High
CONF_MED = 0.40       # >= 40%  -> Medium; below -> Low
WEB_CONFIDENCE_CAP = 0.50  # unverified 'web' tier ceiling (CP 3.1)
WEB_CONFIDENCE_FLOOR = 0.30  # a summarized web pass is *some* grounding, not nothing


def score_confidence(*, kb_top: float, kb_hits: int, flights: int, hotels: int,
                     web_pages: int, places: int, dest_covered: bool,
                     flight_estimate: bool = False,
                     flight_live: bool = False,
                     seasons: bool = False,
                     osm_places: int = 0,
                     safety: bool = False,
                     luxury: bool = False,
                     clarified: bool = False) -> Dict[str, Any]:
    """Deterministic confidence score (0..1) + band + human-readable evidence summary."""
    if clarified:
        return {
            "confidence": 0.10, "confidence_pct": 10, "confidence_label": "Low",
            "confidence_detail": "clarification requested — nothing was retrieved or invented "
                                 "(input check)",
        }
    if kb_hits:
        conf = 0.35 + 0.55 * max(0.0, min(1.0, kb_top))
        if dest_covered:
            conf += 0.05  # the KB explicitly covers this destination (region guard passed)
    else:
        conf = 0.10  # nothing grounded — whatever follows is structured sample data or web
    if flights:
        conf += 0.08   # structured flight schedule (mock OAG-style)
    if hotels:
        conf += 0.08   # structured hotel schedule
    if flight_estimate:
        conf += 0.05   # labeled distance-based estimate — some grounding, never High (CP 1.1)
    if flight_live:
        conf += 0.06   # real route data (flightconnections.com) — unverified tier, still not High (CP 3.1)
    if seasons:
        conf += 0.08   # seasonal dataset — real 2023-2025 climate normals + public holidays (key-free)
    if safety:
        conf += 0.08   # official travel advisory (U.S. State Department — verified source, key-free)
    if luxury:
        # Key-free Wikipedia Michelin 3-star + OSM brand data. Real, but a single
        # community-maintained source — a firm Medium, never High (CP 1.1 honesty).
        conf = max(conf, 0.50)
    if web_pages or places or osm_places:
        conf = max(conf, WEB_CONFIDENCE_FLOOR)
        conf = min(conf, WEB_CONFIDENCE_CAP)  # unverified tier can't reach High (CP 3.1)
    conf = max(0.05, min(conf, 0.98))
    label = "High" if conf >= CONF_HIGH else ("Medium" if conf >= CONF_MED else "Low")
    bits: List[str] = []
    if kb_hits:
        bits.append(f"{kb_hits} KB source{'s' if kb_hits != 1 else ''} (top {kb_top:.2f}"
                    + (", covered destination" if dest_covered else "") + ")")
    if flights:
        bits.append(f"{flights} flight{'s' if flights != 1 else ''} (sample schedule)")
    if hotels:
        bits.append(f"{hotels} stay{'s' if hotels != 1 else ''} (sample schedule)")
    if flight_estimate:
        bits.append("flight pricing estimate (calibrated distance model — not live)")
    if flight_live:
        bits.append("flight route data (flightconnections.com — real schedules, double-check before booking)")
    if seasons:
        bits.append("seasonal dataset (real 2023-2025 climate normals + holidays — key-free APIs)")
    if safety:
        bits.append("official travel advisory (U.S. State Department — verified source, key-free)")
    if luxury:
        bits.append("luxury dataset (Wikipedia Michelin 3-star list + OpenStreetMap brand data — key-free, single source)")
    if web_pages:
        bits.append(f"{web_pages} web page{'s' if web_pages != 1 else ''} (community-sourced — double-check before booking)")
    if places:
        bits.append(f"{places} research lead{'s' if places != 1 else ''} (community-sourced)")
    if osm_places:
        bits.append(f"{osm_places} local place{'s' if osm_places != 1 else ''} "
                    "(community-mapped — double-check before visiting)")
    if not bits:
        bits.append("no grounded evidence — honest fallback, no invented facts")
    return {
        "confidence": round(conf, 3),
        "confidence_pct": int(round(conf * 100)),
        "confidence_label": label,
        "confidence_detail": " · ".join(bits),
    }


# ---------------------------------------------------------------------------
# L5 — human-readable guardrail log (summary event shape)
# ---------------------------------------------------------------------------
def summary_checks(mode: str, input_filter: str, budget: RunBudget,
                   answer: Dict[str, Any]) -> Dict[str, str]:
    """The human-readable log that closes every run (CP 6.1 L5)."""
    return {
        "L1 input filter": input_filter,
        "L2 permissions": (
            f"job='{mode}' · tools={', '.join(sorted(permitted_tools(mode, budget.web_enabled)))} · "
            f"scope={'; '.join(data_scope(mode, budget.web_enabled))}"
        ),
        "L3 validation + rate limits": budget.summary(),
        "L4 output score": (
            f"{answer.get('confidence_label', '?')} {answer.get('confidence_pct', '?')}% — "
            f"{answer.get('confidence_detail', '')}"
        ),
    }
