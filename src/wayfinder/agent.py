"""The agent (CP 2.1 ReAct + CP 1.1 guardrails + CP 4.1 beam limits).

`Agent.run(goal_text)` yields ordered events the server streams to the UI:
  memory  - episodic/semantic memory activity (what the agent is 'thinking' about before acting)
  thought - the ReAct reasoning step (insight into what the agent is searching for)
  action  - exact tool + arguments
  observation - what came back (counts, best hit, scores)
  prune   - beam-pruned candidates with reasons (CP 4.1)
  answer  - final composed answer with citations + follow-ups + confidence (CP 6.1 L4)
  guardrail - CP 6.1 check events (L1 clarify / L2-L3 denials) + the L5 summary
  done    - terminal stats (+ confidence, CP 6.1 L4)
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any, Dict, Iterator, List, Optional

from . import airports  # home-base dataset API (data/airports.json, offline)
from .llm import LLM
from .memory import EpisodicRecord, Memory
from .planner import (
    EXPLICIT_CAPITAL_RE,
    HeuristicPlanner,
    KNOWN_DESTINATIONS,
    LLMPlanner,
    REF_CAPITAL_RE,
    WEAK_KB_TOP,
    parse_goal,
)
from .world import (resolve_capital, effective_home, region_home, home_country,
                    country_facts, _norm_place)  # offline capital resolution + home guard + area brief

_DEST_CONTEXT = HeuristicPlanner.DEST_CONTEXT
from . import guardrails  # CP 6.1 — safety guardrails & human intervention plan
from .reqlog import recent_requests
from .tools import (advanced_search_tool, best_time, book_ticket, flight_info, local_places_tool,
                    lookup, luxury_experiences_tool, search_flights, search_hotels, search_kb,
                    travel_safety_tool,
                    web_fetch_tool, web_search_tool)
from .vectorstore import VectorStore
from .web import WEB_ENABLED, wikipedia_intro

MAX_STEPS = 8          # CP 1.1: "guard rails in place to stop useless information from disrupting"
MAX_PASSES = 4         # CP 4.1: depth 3-4

_WEB_JUNK = re.compile(
    r"^\s*(coordinates:|several terms redirect|this article is about|.*?redirects? here"
    r"|for other uses|place in \w+( metropolitan)? area|populated place in"
    r"|from wikivoyage|updated |see also|references|external links"
    r"|this section needs|needs more citations|this article needs"
    r"|learn how and when to remove|it is requested that editors)",
    re.I,
)


def _clean_web_line(s: str) -> str:
    """Normalize an extracted web line: kill bullet/no-break-space artifacts
    ('• • Mayor…') and strip Wikipedia reference markers ('[1][2]', '[update]',
    '[citation needed]') so bullets read as prose, not markup."""
    s = s.strip(" -*•\t\u00a0")
    s = re.sub(r"(\[\d+\])+|\[update\]|\[citation needed\]|\[a\]|\[note ?[^\]]*\]|\[fn ?[^\]]*\]",
               " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+([,.;:)\]])", r"\1", s)  # '…January 2026 , and…' -> '…January 2026, and…'
    return s.strip(" ,;\t")


_WEB_FOOTNOTE = re.compile(r"^\d+\s+[A-Z]")  # '1 French Land Register data…' = a footnote


def _web_bullets(text: str, n: int = 2) -> List[str]:
    """Pick the first n substantive lines of fetched web text (skip infobox/
    hatnote boilerplate like 'Coordinates:' or 'Several terms redirect here…').
    Complete sentences are preferred over same-length image captions."""
    prose: List[str] = []
    other: List[str] = []
    for ln in text.splitlines():
        s = _clean_web_line(ln)
        # >=100 chars skips image captions / infobox fragments — real prose is longer
        if len(s) < 100 or _WEB_JUNK.match(s) or _WEB_FOOTNOTE.match(s):
            continue
        (prose if s.endswith(".") else other).append(s)
    out = (prose or other)[:n]
    if not out:
        body = re.sub(r"\s+", " ", text.strip())
        if body:
            out = [body[:280]]
    return out


_WEB_FACT_NUM = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9]|[0-9][0-9,]{2,})\b")


def _web_facts(pages: List[dict], n: int = 4) -> List[str]:
    """'At a glance' bullets: fact-like lines (40-220 chars, containing a year or
    a number) from the fetched pages — deduped, hatnote/infobox/maintenance junk
    skipped. Long prose lines are split into sentences so their facts are found.
    Deterministic extraction (no LLM in compose) over tier-'web' text (CP 3.1)."""
    seen = set()
    out: List[str] = []

    def consider(s: str) -> None:
        s = _clean_web_line(s)
        if not (40 <= len(s) <= 220) or _WEB_JUNK.match(s) or _WEB_FOOTNOTE.match(s):
            return
        if not _WEB_FACT_NUM.search(s):
            return
        # Short lines that don't end sentence-like are usually image captions
        # ('…a c. 1805 portrait by James Eights') — real facts usually do.
        if len(s) < 100 and not s.endswith((".", ":", ")", "]")):
            return
        key = s.lower()[:80]
        if key in seen:
            return
        seen.add(key)
        out.append(s)

    for p in pages:
        for ln in (p.get("text") or "").splitlines():
            ln = _clean_web_line(ln)
            if len(ln) <= 220:
                consider(ln)
            elif len(ln) <= 700:
                for sent in re.split(r"(?<=[.!?])\s+", ln):
                    consider(sent)
            if len(out) >= n:
                return out
    return out


# ---- Quick report (Do / Go / Stay): 'what most people recommend' rollup --------
# Deterministic (no LLM in compose, CP 3.1): a line only qualifies if it carries a
# recommendation signal AND a bucket keyword — so the top summary never invents a
# recommendation. Order of evidence = order retrieved (highest-signal first).
_REC_SIGNAL = re.compile(
    r"(recommend|recommendation|must[- ]?see|must[- ]?do|don'?t miss|not to be missed|"
    r"top |best |worth|iconic|famous|popular|favourite|favorite|hidden gem|"
    r"can'?t miss|you should|should visit|try |notable|signature|can'?t go wrong)", re.I)

_DO_KW = re.compile(
    r"(snorkel|dive|beach|ride|museum|tour|eat|dining|food|cuisine|sight|attraction|"
    r"visit|walk|cruise|shop|market|park|quarry|swim|stingray|raft|cave|zoo|aquarium|"
    r"golf|safari|festival|concert|gallery|theatre|theater|climb|hike|fish fry|food)", re.I)

_GO_KW = re.compile(
    r"(beach|district|town|city|quarry|bay|square|promenade|street|island|islands|"
    r"neighborhood|neighbourhood|area|capital|waterfall|harbor|harbour|garden|lighthouse|"
    r"plaza|walkway|cove|reef|headland|old town|downtown)", re.I)

_STAY_KW = re.compile(
    r"(stay|hotel|resort|accommodation|sleep|where to stay|bed and breakfast|bnb|"
    r"guesthouse|villa|apartment|suite|inn|hostel|room|lodging)", re.I)

# Photo-topic mapping (CP 3.1 — the photos should mirror what the user ASKED for:
# a 'best beaches in Hawaii' question leads with beach photos, not a random sight
# plus food). (scene, UI group label, trigger words in the raw question). Order =
# priority: an upscale question that also names hotels stays 'Luxury', not 'Where
# to stay'; 'stays' beats 'dining' when both are named, and so on.
_PHOTO_SCENES = [
    ("luxury", "Luxury",
     ("luxury", "luxurious", "upscale", "exclusive", "michelin", "yacht", "penthouse")),
    ("stays", "Where to stay",
     ("hotel", "hotels", "stay", "stays", "accommodation", "resort", "resorts")),
    ("beaches", "Beaches & coast",
     ("beach", "beaches", "shore", "coast", "coastline", "surfing", "snorkeling")),
    ("museums", "Museums & culture",
     ("museum", "museums", "gallery", "galleries", "exhibition", "exhibitions")),
    ("nature", "Outdoors & scenery",
     ("nature", "waterfall", "waterfalls", "hiking", "jungle", "forest",
      "mountain", "mountains", "lake", "lakes", "canyon", "canyons", "wildlife",
      "desert", "deserts", "dune", "dunes", "snow", "skiing")),
    ("nightlife", "Nightlife", ("nightlife", "clubs", "cocktails")),
    ("shopping", "Shopping & markets",
     ("shop", "shops", "shopping", "gifts", "souvenir", "souvenirs",
      "market", "markets", "bazaar")),
    ("dining", "Dining & food",
     ("dining", "restaurant", "restaurants", "food", "cuisine", "eat", "coffee",
      "cafe", "wine", "winery", "vineyard", "vineyards")),
    ("festivals", "Festivals & events",
     ("festival", "festivals", "carnival", "parade", "parades", "celebration",
      "celebrations", "lantern", "lanterns")),
    ("wildlife", "Wildlife & animals",
     ("wildlife", "animals", "animal", "safari", "zoo", "zoos", "aquarium",
      "whale", "whales", "dolphin", "dolphins", "elephant", "elephants",
      "lion", "lions", "monkey", "monkeys", "penguin", "penguins", "shark", "sharks")),
    ("history", "History & heritage",
     ("history", "historical", "historic", "ancient", "old town", "ruins",
      "monument", "monuments", "heritage")),
    ("art", "Art & galleries",
     ("art", "painting", "paintings", "sculpture", "sculptures", "mural", "murals")),
    ("night views", "Night views & skyline",
     ("night views", "skyline", "illuminated", "city lights")),
]


def _photo_scenes_for_goal(goal: Dict[str, Any]) -> Dict[str, Any]:
    """Photo groups mirroring the question's topic: {"scenes": [(scene, label), ...],
    "explicit": bool}. Up to 2 scenes. explicit=True when a topic word was found in
    the question itself — the topic should then DOMINATE the photo block (4 of 6
    images, e.g. 'best beaches in Greece'). The dining fallback for general trip
    questions keeps the older landmark-led balance (dining 2, sights 4). A
    'best time' run is about the destination as a whole, so it gets scenery."""
    raw = (goal.get("raw") or "").lower()
    if (goal.get("focus") or []) == ["seasons"]:
        return {"scenes": [("nature", "Outdoors & scenery")], "explicit": True}
    found: List[tuple] = []
    for scene, label, words in _PHOTO_SCENES:
        if any(re.search(rf"\b{re.escape(w)}\b", raw) for w in words):
            found.append((scene, label))
        if len(found) >= 2:
            break
    if found:
        return {"scenes": found, "explicit": True}
    if "luxury" in (goal.get("focus") or []):
        return {"scenes": [("luxury", "Luxury")], "explicit": False}
    return {"scenes": [("dining", "Dining & food")], "explicit": False}

# Real 'things to do & see' OpenStreetMap categories — a POSITIVE set, because OSM
# 'sights' also carries parking, fuel and generic buildings, which must NOT read as
# activities (a parking lot is not a recommendation).
_DO_CATS = {
    "attraction", "museum", "viewpoint", "theme_park", "miniature_park", "wildlife_park",
    "zoo", "aquarium", "garden", "nature_reserve", "monastery", "castle", "fort",
    "waterfall", "artwork", "place_of_worship", "theatre", "cinema", "stadium",
    "swimming_pool", "beach", "lighthouse",
}  # NB: generic 'park' / 'spa' / 'fountain' are deliberately excluded — in OSM these
    # are often parking lots, ornamental features or spa services, not 'things to do'.


def _rec_lines(texts: List[str], kw: "re.Pattern", n: int = 3) -> List[str]:
    """Up to n concise recommendation lines that both carry a recommendation signal
    AND mention a bucket keyword (e.g. an activity for Do). Cleans citation markers
    and whitespace, keeps 25-160 chars, dedupes case-insensitively. Deterministic:
    the order of `texts` (retrieval order) is the order candidates are considered."""
    seen: set = set()
    out: List[str] = []
    for text in texts:
        if not text:
            continue
        frags: List[str] = []
        for ln in str(text).splitlines():
            ln = _clean_web_line(ln)
            if len(ln) <= 180:
                frags.append(ln)
            else:
                frags.extend(re.split(r"(?<=[.!?])\s+", ln))
        for f in frags:
            f = re.sub(r"(\[\d+\])+|\[update\]|\[citation needed\]", " ", f)
            f = re.sub(r"\s+", " ", f).strip(" .;,")
            if not (25 <= len(f) <= 160):
                continue
            if not (_REC_SIGNAL.search(f) and kw.search(f)):
                continue
            key = f.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
            if len(out) >= n:
                return out
    return out


# ============================================================================
# Area brief — "📖 {destination} in a minute": the most important facts, shown
# ABOVE the Quick report. Grounded ONLY (CP 1.1 / CP 3.1): the gist comes from
# the KB guide's Overview, the "can't-skip history" from the KB guide's Context
# (or a key-free Wikipedia intro for known countries when the KB has none), and
# the at-a-glance row from the world dataset + this run's seasons/safety. Any
# fact with no grounded source is OMITTED (never invented / misattributed).
# ============================================================================
_KB_SLUG_MAP_CACHE: Dict[str, Dict[str, str]] = {}
_KB_BRIEF_CACHE: Dict[str, dict] = {}


def _kb_slug_map(data_dir: str) -> Dict[str, str]:
    """Map a KB country-doc slug ('congo republic') -> its file path, for the
    cNNN_*.md guide docs only. Demo docs (01_cayman…, 04_kennywood…) start with
    a digit and are excluded — they have no country history, so they fall back
    to a retrieval gist (honest)."""
    if data_dir in _KB_SLUG_MAP_CACHE:
        return _KB_SLUG_MAP_CACHE[data_dir]
    m: Dict[str, str] = {}
    kb_dir = os.path.join(data_dir, "kb")
    if os.path.isdir(kb_dir):
        try:
            for fn in os.listdir(kb_dir):
                # c100_congo_republic.md  ->  slug "congo republic"
                if fn.startswith("c") and len(fn) >= 9 and fn[1:4].isdigit() and fn.endswith(".md"):
                    m[fn[5:-3].replace("_", " ")] = os.path.join(kb_dir, fn)
        except OSError:
            pass
    _KB_SLUG_MAP_CACHE[data_dir] = m
    return m


def _kb_country_doc(data_dir: str, dest: str) -> str | None:
    """The KB guide doc for a destination (if one exists), matched on the
    accent-stripped canonical slug. Tries effective_home / home_country / dest in
    turn so a bare 'congo' or a city both resolve to the country guide."""
    pm = _kb_slug_map(data_dir)
    if not pm:
        return None
    for raw in (effective_home(dest), home_country(dest), dest):
        n = _norm_place(raw or "")
        if n and n in pm:
            return pm[n]
    return None


def _md_section(md: str, header: str) -> str:
    """The body of a `## <header>` section, flattened to one line ('' if absent)."""
    m = re.search(r"^\s*##\s+" + re.escape(header) + r"\s*$(.*?)(?=^\s*##\s|\Z)",
                  md, re.M | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def _sentences(text: str, n: int) -> str:
    """The first up-to-`n` sentences of a block of text (whitespace-collapsed)."""
    s = re.sub(r"\s+", " ", text or "").strip()
    if not s:
        return ""
    return " ".join([p for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()][:n]).strip()


def _kb_doc_brief(path: str) -> dict:
    """Light, cached parse of a KB guide doc: frontmatter title/source/url plus the
    Overview (gist), Context (history), and the 'Top places' line. Never raises."""
    if path in _KB_BRIEF_CACHE:
        return _KB_BRIEF_CACHE[path]
    out = {"title": "", "source": "", "url": "", "overview": "", "context": "", "top_places": ""}
    try:
        with open(path, encoding="utf-8") as f:
            md = f.read()
    except OSError:
        _KB_BRIEF_CACHE[path] = out
        return out
    for field in ("title", "source", "url"):
        m = re.search(r"^" + field + r":\s*(.+)$", md, re.M)
        if m:
            out[field] = m.group(1).strip()
    out["overview"] = _md_section(md, "Overview")
    out["context"] = _md_section(md, "Context") or _md_section(md, "History")
    top = _md_section(md, "Where to go")
    mm = re.search(r"Top places:\s*(.+)", top)
    if mm:
        out["top_places"] = mm.group(1).strip()
    _KB_BRIEF_CACHE[path] = out
    return out


_HISTORY_SIGNAL = re.compile(
    r"(histor|founded?|independen|ancient|empire|colon|revolu|coup|dynasty|medieval"
    r"|world war|wwi|wwii|centur|trade route|silk road|invasion|conquer|treaty"
    r"|constitution|declar|sovereign|republic|monarch|kingdom|pre-colum|viking"
    r"|roman|ottoman|united|merg|divid)", re.I)


def _pick_history_sentence(intro: str) -> str:
    """From a Wikipedia intro, the most history-like sentence (else sentence 2).
    Keeps the 'can't-skip history' line grounded even when the KB has no Context."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (intro or "").strip()) if s.strip()]
    if not sents:
        return ""
    for s in sents:
        if _HISTORY_SIGNAL.search(s):
            return s
    return sents[1] if len(sents) >= 2 else sents[0]


def _area_brief_md(ab: dict) -> str:
    """Markdown form of the area brief — CLI/SSE parity for the richer UI card."""
    lines = [
        f"## 📖 {ab['destination']} in a minute",
        "_The short version — the facts worth knowing before you go._",
        "",
    ]
    if ab.get("gist"):
        lines += [f"**What it is.** {ab['gist']}", ""]
    if ab.get("history"):
        h = ab["history"] + (f"  _({ab['history_source']})_" if ab.get("history_source") else "")
        lines += [f"**The history you can't skip.** {h}", ""]
    facts = ab.get("facts") or []
    if facts:
        lines.append("**At a glance:** " + " · ".join(f"{f['label']}: **{f['value']}**" for f in facts))
        lines.append("")
    if ab.get("provenance"):
        lines.append(f"_{ab['provenance']}_")
    return "\n".join(lines)


class Agent:
    def __init__(self, store: VectorStore, data_dir: str, llm: LLM):
        self.store = store
        self.data_dir = data_dir
        self.llm = llm
        self.memory = Memory(os.path.join(data_dir, "memory.json"))
        self.planner = LLMPlanner(llm) if llm.enabled else HeuristicPlanner(llm)
        self.flights_db = self._load("flights.json", [])
        self.hotels_db = self._load("hotels.json", [])
        self.bookings: List[dict] = self._load("bookings.json", [])
        # Web crawl store: second vector index over live-fetched pages (CP 3.1 provenance).
        # Grows only when the agent fetches the web; cached per-URL (CP 2.1).
        self.web_index_path = os.path.join(data_dir, "web_index.json")
        if os.path.exists(self.web_index_path):
            self.web_store = VectorStore.load(self.web_index_path)
        else:
            self.web_store = VectorStore()
        self.mode = self.planner.name

    def _load(self, name: str, default):
        path = os.path.join(self.data_dir, name)
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, name: str, payload) -> None:
        with open(os.path.join(self.data_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

    # ------------------------------------------------------------------
    def run(self, goal_text: str, home: Optional[Dict[str, str]] = None) -> Iterator[Dict[str, Any]]:
        started = time.time()
        today = date.today()
        goal = parse_goal(goal_text)

        # ---- Home base (the top-of-page setting): flights anchor on the user's
        #      OWN gateway — Nevada -> Las Vegas (LAS), Turkey -> Istanbul (IST) —
        #      not on random US hubs. An explicit "from X" in the query always
        #      wins. Offline + dataset-backed (data/airports.json, 47 countries);
        #      an unusable home degrades to the standard no-origin behavior
        #      (CP 1.1: never guess an airport).
        if home:
            home_desc = airports.describe_home(home)
            if home_desc:
                if not goal.get("origin"):
                    goal["origin"] = home_desc["city"] or home_desc["iata"]
                goal["home"] = home_desc
                _h_loc = home_desc["region"] + ", " if home_desc["region"] else ""
                yield self._ev("memory", kind="home_base",
                               note=(f"Home base: {_h_loc}{home_desc['country']} — anchoring flights on "
                                     f"**{home_desc['airport'] or home_desc['city']} ({home_desc['iata']})**, "
                                     "your main gateway (disclosed curated hub ranking — not a fact). "
                                     "An explicit 'from …' in a query always overrides this."))
            else:
                yield self._ev("memory", kind="home_base",
                               note=("Home base not usable: that country/region isn't in the airport dataset "
                                     "(47 countries covered) or has no resolvable gateway — continuing with "
                                     "the standard no-origin behavior. I won't guess an airport."))

        # ---- semantic memory: context from the prior turn (CP 2.1) ----
        prev_dest = ""
        for turn in reversed(self.memory.semantic):
            if turn.get("role") != "user":
                continue
            prev = parse_goal(turn.get("text", ""))
            if prev["destination"]:
                prev_dest = prev["destination"]
                break

        # Reference follow-ups (CP 2.1: use the conversation context):
        # 'the capital' after a New York conversation means the capital OF New York —
        # a NEW place to research, not the previous place and not the word 'capital'
        # as a concept. We resolve it with a targeted search ('capital of {context}');
        # the search results are the disambiguator and compose() offers alternates.
        resolve_query = resolve_ref = ""
        ask_capital = False
        capital_resolved: Dict[str, str] = {}
        capital_ref = REF_CAPITAL_RE.search(goal_text)
        if capital_ref:
            explicit = EXPLICIT_CAPITAL_RE.search(goal_text)
            base = explicit.group(1).strip() if explicit else prev_dest
            if base:
                hit_cap, hit_kind = resolve_capital(base)
                if hit_cap:
                    # OFFLINE resolution from the 197-country world dataset —
                    # 'the capital' after a California conversation -> Sacramento,
                    # France -> Paris: a NEW place the KB already covers, so no web
                    # research is needed (CP 2.1 context, zero API keys).
                    goal["destination"] = hit_cap
                    capital_resolved = {
                        "from": base, "to": hit_cap, "kind": hit_kind,
                        "ref": (explicit.group(0) if explicit
                                else capital_ref.group(0)).strip(),
                    }
                    yield self._ev("memory", kind="semantic",
                                   note=f"'{capital_resolved['ref']}' — the capital of {base} is "
                                        f"**{hit_cap}** ({hit_kind}); resolved offline from the world "
                                        "dataset, no web research needed.")
                else:
                    resolve_query = f"capital of {base}"[:80]
                    resolve_ref = capital_ref.group(0).strip()
                    goal["destination"] = ""  # 'capital' / 'of X' must not become a destination
                    yield self._ev("memory", kind="semantic",
                                   note=f"'{resolve_ref}' is a reference — resolving it from our conversation "
                                        f"about {base} (from our conversation).")
            else:
                ask_capital = True  # no place in context yet -> compose will ask back
                goal["destination"] = ""
                yield self._ev("memory", kind="semantic",
                               note="'the capital' — no place in our conversation yet, so I'll ask which "
                                    "capital you mean.")
        elif not goal["destination"] and prev_dest:
            # plain semantic inheritance ('how about something cheaper?', 'that place')
            goal["destination"] = prev_dest
            yield self._ev("memory", kind="semantic",
                           note=f"Inherited '{prev_dest}' from the previous turn (from our conversation).")

        # ---- CP 6.1 L1: filtered input — clarify incomplete queries BEFORE any tool work ----
        # The first line of defense: a query missing its destination must not burn KB
        # passes on an empty region hint and then apologize — ask instead (never guess).
        # The 'the capital' paths already ask/resolve on their own, so L1 stays out of
        # their way. Cheap local checks only (guardrails.check_input) — zero extra latency.
        budget = guardrails.RunBudget(WEB_ENABLED)
        clarify = None
        if not ask_capital and not resolve_query:
            clarify = guardrails.check_input(goal_text, goal, prev_dest)
        if clarify:
            answer = self._clarify_answer(goal, clarify)
            yield self._ev("guardrail", layer="L1", check="input filter", result="clarify",
                           detail=clarify["reason"])
            yield self._ev("answer", **answer)
            self.memory.remember_turn(goal_text, clarify["question"][:300])
            self.memory.save()
            yield self._ev("guardrail", layer="summary",
                           checks=guardrails.summary_checks(
                               self.mode, "clarify — " + clarify["reason"], budget, answer))
            yield self._ev("done", steps=0, searches=0, pruned=0,
                           elapsed_ms=int((time.time() - started) * 1000), mode=self.mode,
                           confidence=answer["confidence_pct"],
                           confidence_label=answer["confidence_label"])
            return

        # ---- episodic memory: cache recall (CP 2.1: "retrieved again with a cache")
        recall = self.memory.recall(goal_text, goal.get("destination", ""))
        recall_hit = recall is not None
        if recall_hit:
            yield self._ev("memory", kind="episodic",
                           note=f"Cache hit: a prior query matched this goal ('{recall.query}'). "
                                f"Reusing its lookup path and keeping the best results.")

        # ---- request-log history (CP 2.1 + logging): data/request_log.jsonl is the
        # single source of truth for what this user asked before. A repeat of a
        # destination (or a near-duplicate query) gets an explicit 'I remember'
        # memory event. Read-only, wrapped so a logging hiccup can never break a run.
        # NB: the in-flight run is NOT in the log yet (the server appends it in a
        # finally-block after the stream), so a repeated query correctly sees its
        # own prior runs — exactly what this feature is for.
        try:
            hist = [r for r in recent_requests(self.data_dir, 50)
                    if isinstance(r, dict) and (r.get("query") or "")]
            if hist:
                dest_now = str(goal.get("destination", "") or "").strip()
                prior_same = [r for r in hist if dest_now
                              and str(r.get("destination", "") or "").strip().lower() == dest_now.lower()]
                if prior_same:
                    most_recent = prior_same[0].get("query", "")
                    yield self._ev("memory", kind="history",
                                   note=f"You've asked about {dest_now} {len(prior_same)}\u00d7 before "
                                        f"(most recent: \"{str(most_recent)[:60]}\", logged "
                                        f"{prior_same[0].get('ts', '')}). Carrying that context forward.")
                else:
                    # near-duplicate fallback: token overlap >= 0.5 over content words (len > 3)
                    q_tokens = {t for t in re.findall(r"[a-z]{4,}", goal_text.lower())}
                    similar = None
                    if len(q_tokens) >= 2:
                        for r in hist:
                            r_tokens = {t for t in re.findall(r"[a-z]{4,}", str(r.get("query", "")).lower())}
                            if len(r_tokens) >= 2:
                                overlap = len(q_tokens & r_tokens) / min(len(q_tokens), len(r_tokens))
                                if overlap >= 0.5:
                                    similar = r
                                    break
                    if similar:
                        yield self._ev("memory", kind="history",
                                       note=f"A similar request was logged before — \"{str(similar.get('query', ''))[:60]}\" "
                                            f"({similar.get('ts', '')}). Treating this as a follow-up on the same research thread.")
        except Exception:
            pass  # request-log reads are best-effort; never break the run

        state: Dict[str, Any] = {"steps": [], "kb_results": [], "pruned_total": 0, "recall_hit": recall_hit}
        state["prev_dest"] = prev_dest
        if resolve_query:
            state["resolve_query"] = resolve_query
            state["resolve_ref"] = resolve_ref
            state["resolve_kind"] = "capital"
        if ask_capital:
            state["ask_capital"] = True
        if capital_resolved:
            state["capital_resolved"] = capital_resolved
        events_budget = MAX_STEPS
        region_hint = " ".join(
            [goal.get("destination", ""), _DEST_CONTEXT.get(goal.get("destination", ""), "")]
        )

        def emit(ev_type: str, **payload) -> Dict[str, Any]:
            return self._ev(ev_type, **payload)

        best_flight = None
        for step in range(events_budget):
            if len(state["kb_results"]) >= MAX_PASSES:
                break  # depth guardrail (CP 4.1)
            thought, action, action_input = self.planner.next_step(goal, state)
            yield emit("thought", step=step + 1, text=thought)

            if action == "compose_answer":
                break

            yield emit("action", step=step + 1, tool=action, input=action_input)

            # ---- CP 6.1 L2 + L3 gate: scoped permissions -> validation -> rate budget ----
            # Out-of-scope, invalid, or over-budget calls are DENIED with a logged
            # reason (never guessed, never silently dropped); MAX_DENIED in a row
            # halts the loop instead of letting a broken planner spin (CP 1.1 + CP 6.1).
            gate_ok, gate_reason, norm_args = guardrails.check_call(
                self.mode, action, action_input, budget)
            if not gate_ok:
                budget.record_denied(gate_reason)
                yield self._ev("guardrail", layer="L2/L3", check=f"call gate: {action}",
                               result="denied", detail=gate_reason)
                yield emit("observation", step=step + 1, tool=action,
                           observation=f"denied: {gate_reason} — continuing")
                if budget.halted:
                    yield self._ev("guardrail", layer="L3", check="circuit breaker",
                                   result="halt",
                                   detail=(f"{guardrails.MAX_DENIED} rejected calls in a row — "
                                           "composing from what we have"))
                    break
                continue
            action_input = norm_args

            try:
                if action == "search_kb":
                    result, obs = search_kb(
                        self.store, action_input.get("query", ""), int(action_input.get("k", 8)),
                        today, region_hint=region_hint,
                        anchor=goal.get("destination", ""),
                    )
                elif action == "search_flights":
                    result, obs = search_flights(
                        self.flights_db,
                        action_input.get("origin", goal.get("origin", "")),
                        action_input.get("destination", ""),
                        action_input.get("date", goal.get("date", "")),
                        action_input.get("max_price"),
                    )
                elif action == "search_hotels":
                    result, obs = search_hotels(
                        self.hotels_db,
                        action_input.get("destination", goal.get("destination", "")),
                        action_input.get("max_price"),
                        action_input.get("near", ""),
                    )
                elif action == "flight_info":
                    result, obs = flight_info(
                        self.flights_db,
                        action_input.get("origin") or goal.get("origin") or "",
                        action_input.get("destination") or goal.get("destination") or "",
                        action_input.get("date") or goal.get("date") or "",
                    )
                elif action == "best_time":
                    result, obs = best_time(
                        action_input.get("destination") or goal.get("destination") or "",
                        action_input.get("month"),
                    )
                elif action == "local_places":
                    result, obs = local_places_tool(
                        action_input.get("place") or goal.get("destination") or "",
                        action_input.get("focus", "dining"),
                        int(action_input.get("k", 12)),
                        self.data_dir,
                        raw_context=goal.get("raw") or "",  # disambiguates bare city names ('George Town' + 'Cayman Islands' in the sentence)
                    )
                elif action == "luxury":
                    result, obs = luxury_experiences_tool(
                        action_input.get("destination") or goal.get("destination") or "",
                        self.data_dir,
                    )
                elif action == "travel_safety":
                    result, obs = travel_safety_tool(
                        action_input.get("place") or goal.get("destination") or "",
                        self.data_dir,
                    )
                elif action == "book_ticket":
                    flight = action_input.get("flight") or best_flight or {}
                    result, obs = book_ticket(self.bookings, flight, confirmed=False)
                    self._save("bookings.json", self.bookings)
                elif action == "lookup":
                    result, obs = lookup(self.store, action_input.get("entity", goal.get("destination", "")))
                elif action == "web_search":
                    result, obs = web_search_tool(
                        action_input.get("query", goal.get("destination", "") or goal["raw"][:80]),
                        int(action_input.get("k", 5)),
                    )
                elif action == "web_fetch":
                    result, obs = web_fetch_tool(
                        action_input.get("url", ""), self.web_store, self.web_index_path,
                        provider=action_input.get("provider", ""),
                    )
                elif action == "advanced_search":
                    result, obs = advanced_search_tool(
                        action_input.get("query", goal.get("destination", "") or goal["raw"][:80]),
                        int(action_input.get("k_per_source", 3)),
                    )
                else:
                    result, obs = {"unknown": action}, f"unknown tool '{action}' -> skipped"
            except Exception as exc:  # noqa: BLE001 — CP 1.1 guardrail: a tool error must never kill the loop
                result, obs = {"error": str(exc)}, f"{action} -> tool error: {exc} — continuing"
                if action in {"web_search", "advanced_search"}:
                    state["web_done"] = True
                    state["web_failed"] = True
                    state["web_results"] = []
                elif action == "web_fetch":
                    state["web_fetched"] = True
                    state.setdefault("web_tried", []).append(action_input.get("url", ""))

            budget.consume(action)  # CP 6.1 L3: the attempt used a budget slot
            yield emit("observation", step=step + 1, tool=action, observation=obs)

            if action == "search_kb":
                state["kb_results"].append({"query": action_input.get("query", ""), **result})
                state["pruned_total"] += len(result.get("pruned", []))
                for p in result.get("pruned", []):
                    yield emit("prune", tool=action, id=p["id"], title=p["title"], score=p["score"], reason=p["reason"])
                state["kb_done"] = True
            elif action == "search_flights":
                state["flights"] = result.get("flights", [])
                state["flights_done"] = True
                if state["flights"]:
                    best_flight = min(state["flights"], key=lambda f: int(f.get("price_usd", 10**9)))
                    state["best_flight"] = best_flight
            elif action == "flight_info":
                state["flight_info"] = result
                state["flight_info_done"] = True
            elif action == "best_time":
                state["seasons"] = result
                state["seasons_done"] = True
            elif action == "local_places":
                state["local_places"] = result
                state["local_places_done"] = True
            elif action == "luxury":
                state["luxury"] = result
                state["luxury_done"] = True
            elif action == "travel_safety":
                state["safety"] = result
                state["travel_safety_done"] = True
            elif action == "search_hotels":
                state["hotels"] = result.get("hotels", [])
                state["hotels_done"] = True
            elif action == "book_ticket":
                state["booking_asked"] = True
                state["booking"] = result
            elif action == "web_search":
                state["web_done"] = True
                state["web_results"] = result.get("results", [])
                state.setdefault("web_searches", []).append(action_input.get("query", ""))
                if result.get("error"):
                    state["web_failed"] = True
                elif state.get("resolve_query") and not state.get("resolve_results"):
                    # Keep the FIRST (resolution) pass's results before the deeper pass
                    # overwrites web_results — compose() offers them as alternates.
                    state["resolve_results"] = list(result.get("results", []))
            elif action == "web_fetch":
                if result.get("error"):
                    # Fetch failed (blocked/403/JS-heavy): allow one fallback candidate
                    state.setdefault("web_tried", []).append(action_input.get("url", ""))
                else:
                    state["web_fetched"] = True  # >=1 fetch done (deeper cap: WEB_MAX_PASSES)
                    state["web_passes"] = state.get("web_passes", 0) + 1
                    page = result.get("page")
                    if page:
                        state.setdefault("web_pages", []).append(page)
                        # Track the URL we asked for — a freshly fetched page dict has no
                        # 'url' key (only cache hits do), and planner/compose dedupe on it.
                        state.setdefault("web_fetched_urls", []).append(action_input.get("url") or "")
                        # Resolve mode: derive the RESOLVED place from the page title
                        # ('Albany' from 'Albany - Wikipedia') so the deeper pass and the
                        # 'more places' step target the NEW place the reference points at.
                        if state.get("resolve_query") and not state.get("resolved_place"):
                            title = (page.get("title") or "").strip()
                            cleaned = re.sub(r"\s*[-–|]\s*(wikipedia|wikivoyage|wiki).*$",
                                            "", title, flags=re.I).strip()
                            if cleaned:
                                state["resolved_place"] = cleaned[:60]
            elif action == "advanced_search":
                state["web_done"] = True
                state["web_results"] = result.get("results", [])
                state.setdefault("web_searches", []).append(action_input.get("query", ""))
                state["research_leads"] = result.get("results", [])
                state["research_by_source"] = result.get("by_source", {})
                if state.get("resolve_query") and not state.get("resolve_results"):
                    # Keep the first (resolution) pass's FETCHABLE results before the deeper
                    # pass overwrites web_results — compose() offers them as alternates.
                    state["resolve_results"] = [r for r in result.get("results", [])
                                                if r.get("fetchable", True)]
                if not result.get("results"):
                    state["web_failed"] = True

            state["steps"].append({"step": step + 1, "thought": thought, "action": action,
                                   "input": action_input, "observation": obs})

            # repeated-search guardrail (CP 1.1)
            sig = (action, json.dumps(action_input, sort_keys=True))
            if any(s["action"] == action and json.dumps(s["input"], sort_keys=True) == json.dumps(action_input, sort_keys=True)
                   for s in state["steps"][:-1]):
                yield emit("prune", tool=action, reason="duplicate action skipped")
                break

        # ---- compose ----------------------------------------------------
        answer = self.compose(goal, state)
        yield emit("answer", **answer)

        # ---- CP 6.1 L5: human-readable guardrail log closes the run -------
        yield self._ev("guardrail", layer="summary",
                       checks=guardrails.summary_checks(
                           self.mode, "pass — query complete (destination resolved)", budget, answer))

        # ---- commit memory (CP 2.1) --------------------------------------
        self.memory.remember_turn(goal_text, answer.get("intro", "")[:300])
        self.memory.record(EpisodicRecord(
            ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
            query=goal_text,
            destination=goal.get("destination", ""),
            actions=[{"tool": s["action"], "input": s["input"]} for s in state["steps"]],
            top_sources=[s["source"] for s in answer.get("sources", [])[:5]],
            summary=answer.get("intro", "")[:200],
            cache_key=f"{goal.get('destination', 'any')}:{goal.get('focus', [])}",
        ))
        self.memory.save()

        yield self._ev("done", steps=len(state["steps"]),
                       searches=len(state.get("kb_results", [])),
                       pruned=state.get("pruned_total", 0),
                       elapsed_ms=int((time.time() - started) * 1000),
                       mode=self.mode,
                       confidence=answer.get("confidence_pct"),
                       confidence_label=answer.get("confidence_label"))

    # ------------------------------------------------------------------
    def _clarify_answer(self, goal: Dict[str, Any], clarify: Dict[str, str]) -> Dict[str, Any]:
        """CP 6.1 L1: a clarifying question is an answer too — scored (Low: nothing
        retrieved, nothing invented) and logged like any other output."""
        return {
            "intro": clarify["question"],
            "sections": [],
            "sources": [],
            "followups": [
                "Plan a 5 day trip to the Cayman Islands from Miami Sep 12 2026, budget $2500, with flights and where to stay",
                "What discounts can I get for Kennywood in Pittsburgh and where should I stay?",
                "Tell me about California",
            ],
            "goal": {k: goal[k] for k in ("destination", "origin", "date", "budget", "nights", "focus", "home")
                    if k in goal},
            **guardrails.score_confidence(kb_top=0.0, kb_hits=0, flights=0, hotels=0,
                                          web_pages=0, places=0, dest_covered=False,
                                          clarified=True),
        }

    # ------------------------------------------------------------------
    def build_quick_report(self, goal: Dict[str, Any], state: Dict[str, Any],
                           kb_hits: List[dict]) -> Dict[str, Any] | None:
        """Concise 'what most people recommend' rollup for the TOP of the answer:
        Do (things to do & eat) · Go (where to be + when) · Stay (where to sleep).

        Synthesized deterministically from the evidence this run ALREADY retrieved
        (CP 3.1 — no LLM in compose, no invented recommendations):
          - sample hotel inventory   -> Stay (+ the neighborhoods people stay in -> Go)
          - OpenStreetMap places     -> Do (dining + sights, real mapped names)
          - key-free seasons dataset -> Go (best month)
          - KB + live-web text       -> Do/Go/Stay recommendation lines (fallback)
        Returns None when there's no destination or no grounded evidence for any
        bucket — compose then omits the report instead of guessing (honesty model).
        """
        dest = (goal.get("destination") or state.get("resolved_place") or "").strip()
        if not dest or dest == "your trip":
            return None

        from . import osm as _osm  # local import keeps agent import-light when web is off
        hotels = state.get("hotels") or []
        lp = state.get("local_places") or {}
        places = lp.get("places") or []
        dining = [p for p in places if p.get("category") in _osm.DINING_CATEGORIES]
        sights = [p for p in places if p.get("category") not in _osm.DINING_CATEGORIES]
        seasons = state.get("seasons") or {}

        # Evidence text, strongest-signal first (kb_hits already sorted by score).
        texts: List[str] = [r.get("text", "") for r in kb_hits]
        texts += [(p.get("text") or "") for p in (state.get("web_pages") or [])]
        texts += [((r.get("snippet") or "") + " " + (r.get("title") or ""))
                  for r in (state.get("research_leads") or [])]

        do: List[str] = []
        go: List[str] = []
        stay: List[str] = []

        # DO — real attractions (positive category set) + food, then recommendation
        # lines as a top-up. A positive set is essential: OSM 'sights' also includes
        # parking, fuel and generic buildings, which must not read as 'things to do'.
        attractions = [p for p in sights if p.get("category") in _DO_CATS]
        for p in attractions[:2]:
            tail = p.get("address") or ""
            do.append(f"**{p.get('name')}**" + (f" — {tail}" if tail else ""))
        for p in dining[:2]:
            tail = p.get("cuisine") or ""
            do.append(f"eat at **{p.get('name')}**" + (f" ({tail})" if tail else ""))
        if len(do) < 3:
            do = do + _rec_lines(texts, _DO_KW, max(1, 3 - len(do)))
        do = do[:4]

        # GO — best month first, then the neighborhoods people stay in (deduped).
        if seasons.get("best_month"):
            go.append(f"Best month to go: **{seasons['best_month']}**")
        seen_areas: set = set()
        for h in hotels:
            a = (h.get("area") or "").strip()
            ka = a.lower()
            if a and ka not in seen_areas:
                seen_areas.add(ka)
                go.append(f"base yourself in **{a}**")
        if not go:
            go = _rec_lines(texts, _GO_KW, 3)

        # STAY — top hotels by rating (name · area · $/night).
        stay_src = sorted(hotels, key=lambda h: (-(h.get("rating") or 0), h.get("price_usd") or 10**9))
        for h in stay_src[:3]:
            line = f"**{h.get('name')}**"
            bits = []
            if h.get("area"):
                bits.append(h["area"])
            if h.get("price_usd") is not None:
                bits.append(f"${h['price_usd']}/night")
            if bits:
                line += " — " + " · ".join(bits)
            stay.append(line)
        if not stay:
            stay = _rec_lines(texts, _STAY_KW, 3)

        # �쫠 Upscale tier — a HIGHER-PRICED / 4-5★ room so “a more expensive option” is
        # always on the table (CP 3.1 grounded: the top luxury stay when present, else
        # the priciest sample hotel). Omitted when there's nothing real to point at.
        lux = state.get("luxury") or {}
        lux_stays = lux.get("stays") or []
        ups_name, ups_tail = "", ""
        if lux_stays:
            top = max(lux_stays, key=lambda p: (int(p.get("stars") or 0), len(p.get("name") or "")))
            ups_name = (top.get("name") or "").strip()
            star = (top.get("stars") or "").strip()
            ups_tail = f"{star}★" if star else "upscale"
        else:
            priced = [h for h in hotels if h.get("price_usd") is not None]
            if priced:
                top = max(priced, key=lambda h: h["price_usd"])
                ups_name = (top.get("name") or "").strip()
                ups_tail = f"~${top['price_usd']}/night"
        if ups_name and not any(ups_name in (s or "") for s in stay):
            stay.append(f"�쫠 upscale — **{ups_name}**" + (f" ({ups_tail})" if ups_tail else ""))

        if not (do or go or stay):
            return None

        n_sources = sum([bool(hotels), bool(places), bool(seasons.get("covered")),
                         bool(state.get("web_pages")), bool(kb_hits)])
        return {
            "destination": dest,
            "do": do[:4],
            "go": go[:4],
            "stay": stay[:4],
            "provenance": (f"Synthesized live this run from {n_sources} grounded source(s) — "
                           "full detail, citations and confidence are in the sections below."),
        }

    def build_area_brief(self, goal: Dict[str, Any], state: Dict[str, Any],
                         kb_hits: List[dict]) -> Dict[str, Any] | None:
        """'📖 {destination} in a minute' — the most important facts, shown ABOVE the
        Quick report. Grounded only (CP 1.1 / CP 3.1):
          gist   = the KB guide's Overview ('what it is' / its purpose)
          history= the KB guide's Context (the 'can't-skip' facts) when present, else a
                   key-free Wikipedia intro — but ONLY for known countries (a curated
                   demo like Cayman/Kennywood gets the gist, never an invented history)
          glance = capital · region · top places (world dataset) + best month (this
                   run's seasons) + safety (this run's advisory)
        Any fact with no grounded source is OMITTED; returns None when there is
        nothing grounded at all (honesty model — never guess).
        """
        dest = (goal.get("destination") or state.get("resolved_place") or "").strip()
        if not dest or dest == "your trip":
            return None

        facts = country_facts(dest)                        # world.json, or {}
        kb_path = _kb_country_doc(self.data_dir, dest)     # KB guide doc, or None
        brief = _kb_doc_brief(kb_path) if kb_path else {}

        # --- gist ("what it is") — strongest grounded source first ---
        gist = ""
        if brief.get("overview"):
            gist = _sentences(brief["overview"], 1)
        if not gist and facts.get("name"):
            continent = facts.get("continent") or ""
            gist = facts["name"] + (f" is a country in {continent}." if continent else ".")
        if not gist and kb_hits:
            gist = _sentences((kb_hits[0].get("text") or "").strip().split("\n")[0], 1)

        # --- history ("can't skip") — KB Context, else a grounded Wikipedia intro ---
        history = ""
        history_source = ""
        if brief.get("context"):
            history = _sentences(brief["context"], 2)
            history_source = brief.get("source") or "Knowledge base"
        elif facts.get("name") and WEB_ENABLED:
            wiki = wikipedia_intro(facts["name"], 2)
            if wiki:
                history = _pick_history_sentence(wiki)
                history_source = "Wikipedia"

        # --- at-a-glance row — omit any fact with no grounded source ---
        glance: List[dict] = []
        if facts.get("capital"):
            glance.append({"label": "Capital", "value": facts["capital"]})
        if facts.get("continent"):
            glance.append({"label": "Region", "value": facts["continent"]})
        top = brief.get("top_places") or ""
        if not top and facts.get("top_destinations"):
            top = ", ".join(facts["top_destinations"])
        if top:
            parts = [p.strip().rstrip("_").strip() for p in top.split(",") if p.strip()]
            if parts:
                glance.append({"label": "Top places", "value": ", ".join(parts[:5])})
        seasons = state.get("seasons") or {}
        if seasons.get("best_month"):
            glance.append({"label": "Best month", "value": str(seasons["best_month"])})
        safety = state.get("safety") or {}
        if safety.get("found") and (safety.get("rating") or safety.get("level")):
            level = safety.get("level")
            rating = (safety.get("rating") or "").strip()
            glance.append({"label": "Safety", "value": f"Level {level}" + (f" — {rating}" if rating else "")})

        # --- �쫠 luxury / splurge line — the high-end “things to do” (grounded from
        #     the luxury tool: premium experiences + Michelin fine dining + 4-5★ stays).
        #     Omitted when there's no grounded luxury data (honesty model — no guesses). ---
        lux = state.get("luxury") or {}
        lux_picks: List[str] = []
        for p in (lux.get("experiences") or [])[:2]:
            if p.get("name"):
                lux_picks.append(p["name"] + (f" ({p['category']})" if p.get("category") else ""))
        for r in (lux.get("michelin") or {}).get("restaurants", [])[:2]:
            if r.get("name"):
                lux_picks.append(f"Michelin 3★ {r['name']}")
        if len(lux_picks) < 2:
            for p in (lux.get("stays") or [])[:1]:
                if p.get("name"):
                    lux_picks.append(p["name"] + (f" ({p['stars']}★)" if p.get("stars") else ""))
        lux_line = " · ".join(lux_picks)

        if not (gist or history or glance):
            return None

        # --- honest provenance: name only the sources actually used ---
        prov_bits: List[str] = []
        if gist:
            src = (brief.get("source") or "my travel notes") if brief.get("overview") else "retrieval evidence"
            prov_bits.append(f"overview: {src}")
        if history:
            prov_bits.append(f"history: {history_source}")
        if facts:
            prov_bits.append("quick facts: world dataset")
        if seasons.get("best_month"):
            prov_bits.append("best month: climate dataset")
        if safety.get("found"):
            prov_bits.append("safety: official advisory")
        if lux_line:
            prov_bits.append("splurge: OpenStreetMap + Michelin list")
        provenance = "Grounded — " + " · ".join(prov_bits) + "." if prov_bits \
            else "Grounded from this run's evidence."

        ab: Dict[str, Any] = {
            "destination": dest,
            "gist": gist,
            "history": history,
            "history_source": history_source,
            "luxury": lux_line or None,
            "facts": glance,
            "provenance": provenance,
        }
        ab["md"] = _area_brief_md(ab)   # markdown form for CLI/SSE consumers
        return ab

    def fetch_photos(self, goal: Dict[str, Any], state: Dict[str, Any]) -> List[dict]:
        """Topic-relevant photos for the answer (CP 3.1): the groups mirror what the
        user ASKED about — 'best beaches in Hawaii' leads with beach photos,
        'where should I stay in Tokyo' leads with hotel photos, a luxury question
        leads with upscale stays — alongside the OSM-mapped landmarks and the
        destination hero. Each image is a real Commons file with caption +
        artist/license attribution (never invented). Key-free, gated by
        WEB_ENABLED. Returns ordered [{label, images:[{title,url,thumb_url,
        caption,artist,license}]}]; [] when web is off, no destination, or every
        lookup failed — compose then omits the photos block instead of guessing.
        Capped at ~6 images total to keep the answer concise."""
        from . import photos as _photos

        dest = (goal.get("destination") or state.get("resolved_place") or "").strip()
        if not dest or dest == "your trip" or not _photos.WEB_ENABLED:
            return []

        scene_info = _photo_scenes_for_goal(goal)
        scenes = scene_info["scenes"]      # [(scene, label), ...] — max 2
        explicit = scene_info["explicit"]  # topic word in the question itself

        # OSM-mapped attractions first (specific landmarks — e.g. 'Kennywood' when
        # the destination resolved to the city 'Pittsburgh'), the destination
        # itself as the hero fallback, then the topic scene(s) from the question.
        places = (state.get("local_places") or {}).get("places") or []
        names: List[str] = []
        for p in places:
            n = (p.get("name") or "").strip()
            if n and p.get("category") in _DO_CATS and n not in names:
                names.append(n)
        # Order attractions by relevance to the question's topic — 'Waikiki Beach'
        # first for a beach question, the Louvre first for a museum one; ties keep
        # the OSM order (stable sort).
        scene_set = {s for s, _ in scenes}
        topic_words: set = set()
        for scene, _label, words in _PHOTO_SCENES:
            if scene in scene_set:
                topic_words.update(words)
        if topic_words:
            names.sort(key=lambda n: -sum(
                1 for w in topic_words
                if re.search(rf"\b{re.escape(w)}\b", n.lower())))

        jobs: List[tuple] = []
        for n in names[:2]:
            if dest.lower() not in n.lower() and n.lower() != dest.lower():
                jobs.append((n, lambda nm=n: _photos.photos_for_landmark(nm, dest)))
        if not any(n.lower() == dest.lower() for n in names[:2]) and \
                not any(dest.lower() in n.lower() for n, _ in jobs):
            jobs.append((dest, lambda d=dest: _photos.photos_for_landmark(d, d)))
        for scene, label in scenes:
            jobs.append((label, lambda s=scene, d=dest: _photos.photos_for_scene(s, d)))

        CAP = 6
        scene_labels = {label for _s, label in scenes}
        # Per-group quota: an EXPLICIT topic question is dominated by its topic —
        # 'best beaches in Greece' gets 4 beach photos + 2 contextual (a sight and
        # the destination), not 2. Two topics keep 2 each; the general-trip dining
        # fallback keeps the older landmark-led balance (dining 2, sights 4).
        scene_quota = 4 if (len(scenes) == 1 and explicit) else 2
        n_land = max(1, sum(1 for l, _ in jobs if l not in scene_labels))
        per: Dict[str, int] = {}
        left = CAP
        for l, _ in jobs:
            if l in scene_labels:
                per[l] = min(scene_quota, left)
                left -= per[l]
        for l, _ in jobs:
            if l not in scene_labels:
                per[l] = max(1, left // n_land)
        found: Dict[str, List[dict]] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as ex:
            futs = {ex.submit(fn): label for label, fn in jobs}
            for fut in as_completed(futs):
                label = futs[fut]
                room = CAP - sum(len(v) for v in found.values())
                if room <= 0:
                    continue
                try:
                    imgs = fut.result()
                except Exception:  # noqa: BLE001 — any lookup failure: omit, never fabricate
                    continue
                take = min(len(imgs), per[label], room)
                if take:
                    found[label] = imgs[:take]
        # as_completed returns in finish order; restore the jobs' order (landmarks
        # first, dining last) and drop any group that filled no images.
        return [{"label": label, "images": found[label]}
                for label, _ in jobs if found.get(label)]

    def build_flight_details(self, dest: str, flights: List[dict],
                             fi: Dict[str, Any]) -> Dict[str, Any] | None:
        """🛫 Flight details for the Trip bottom line — supplements the cost figure
        with whatever was ACTUALLY retrieved this run (CP 1.1: no invented fields):
          amadeus       -> real offers: price, stops, carriers, dep -> arr times
          live_schedule -> direct/stop availability + real durations + airlines
                           (flightconnections.com, key-free, cited) + fare band
          estimate      -> distance-model duration/stops + fare band (labeled)
          schedule      -> sample-schedule flights (airline, no., times, price)
          none          -> None (the bottom line then just says 'not retrieved')
        """
        src = fi.get("source")
        if src == "amadeus" and fi.get("offers"):
            offers = fi["offers"][:4]
            return {
                "source": "amadeus",
                "label": "Amadeus Self-Service offers — verify before booking",
                "routes": [{
                    "origin": fi.get("origin") or "",
                    "stops": (f"{o.get('stops')} stop(s)" if o.get("stops") is not None else ""),
                    "airlines": (o.get("carriers") or [])[:4],
                    "times": (f"{o.get('departure', '')} → {o.get('arrival', '')}"
                              if (o.get("departure") or o.get("arrival")) else ""),
                    "price_usd": int(o.get("price_usd", 0)) or None,
                } for o in offers],
            }
        if src in ("live_schedule", "estimate") and fi.get("routes"):
            routes: List[dict] = []
            for r in fi["routes"][:4]:
                dur = r.get("duration_h")
                rng = r.get("duration_range") or []
                dur_txt = (f"~{dur}h" + (f"–{rng[1]}h" if rng and rng[1] else "") if dur else "")
                routes.append({
                    "origin": r.get("origin") or "",
                    "origin_code": r.get("origin_code") or "",
                    "direct": bool(r.get("direct")),
                    "stops": r.get("stops") or "",
                    "duration": dur_txt,
                    "airlines": (r.get("airlines") or [])[:4],
                    "via_options": (r.get("via_options") or [])[:3],
                    "economy_low": r.get("economy_low"),
                    "economy_high": r.get("economy_high"),
                })
            if routes:
                return {
                    "source": src,
                    "label": ("Real route data — flightconnections.com (key-free, double-check before booking); "
                              "fares are labeled estimates"
                              if src == "live_schedule"
                              else "Distance-based estimate — NOT live pricing (key-free model)"),
                    "avg_duration_h": fi.get("avg_duration_h"),
                    "routes": routes,
                    "url": fi.get("schedule_url") or "",
                }
            return None
        if flights:
            return {
                "source": "schedule",
                "label": "Sample schedule (sample data) — not live pricing",
                "routes": [{
                    "origin": f.get("origin") or "",
                    "flight_no": f.get("flight_no") or "",
                    "airlines": [f["airline"]] if f.get("airline") else [],
                    "date": f.get("date") or "",
                    "times": (f"{f.get('dep_time', '')} → {f.get('arr_time', '')}"
                              if (f.get("dep_time") or f.get("arr_time")) else ""),
                    "price_usd": int(f.get("price_usd", 0)) or None,
                } for f in flights[:4]],
            }
        return None

    # ------------------------------------------------------------------
    def compose(self, goal: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        sections: List[str] = []
        sources: List[dict] = []
        citations: List[dict] = []

        # Web / lead / local-place home guard (CP 1.1 omit over misattribute):
        # a bare/ambiguous destination ('Congo') can make the live web, the
        # cross-source leads, or the OSM local-place search return the OTHER
        # place's evidence (DR Congo Wikipedia, a Kinshasa restaurant) even though
        # the KB correctly anchors on Congo (Republic). Drop any such item whose
        # title/URL/address names a DIFFERENT known place; undecidable items
        # (region_home=='') and all items for curated demos (home=='') stay on the
        # soft path. Skipped for explicit comparisons ('Congo vs DR Congo'), where
        # both sides are legitimately wanted.
        _whome = effective_home(goal.get("destination") or "")
        _is_compare = bool(re.search(
            r"\b(vs\.?|versus|compare|compared|comparison|against|better than|which is better|side.by.side)\b",
            (goal.get("raw") or "").lower()))
        if _whome and not _is_compare:
            def _home_ok(text: str) -> bool:
                rh = region_home(text)
                return (not rh) or (rh == _whome)
            if state.get("web_pages"):
                state["web_pages"] = [
                    p for p in state["web_pages"]
                    if _home_ok(" ".join(str(p.get(k, "")) for k in ("title", "url")))]
            if state.get("research_leads"):
                state["research_leads"] = [
                    r for r in state["research_leads"]
                    if _home_ok(" ".join(str(r.get(k, "")) for k in ("title", "url")))]
            _lp = (state.get("local_places") or {})
            _ra = _lp.get("resolved_as") or ""
            if _ra and not _home_ok(_ra):
                # OSM geocoded the ambiguous term to the WRONG city ('Congo' -> a
                # Kinshasa guesthouse in DR Congo). Kinshasa and Brazzaville sit
                # across the same river, so the radius query returns a MIX of both and
                # per-place names often carry no city to verify against. Drop the whole
                # set (CP 1.1 honest omit) rather than show off-destination spots, and
                # let the heading fall back to the correct destination.
                _lp["resolved_as"] = ""
                _lp["places"] = []
            elif _lp.get("places"):
                # Correct (or undecidable) geocode: keep, but drop any individual place
                # that names a different known place (defence in depth).
                _lp["places"] = [
                    p for p in _lp["places"]
                    if _home_ok(" ".join(str(p.get(k, "")) for k in ("name", "address", "area", "city")))]

        def cite(source: dict) -> int:
            for i, s in enumerate(sources):
                if s["id"] == source["id"]:
                    return i + 1
            sources.append(dict(source))
            return len(sources)

        kb_hits: List[dict] = []
        for passres in state.get("kb_results", []):
            for r in passres.get("results", []):
                if all(r["id"] != x["id"] for x in kb_hits):
                    kb_hits.append(r)
        kb_hits.sort(key=lambda r: r["scores"]["total"], reverse=True)
        kb_hits = kb_hits[:6]

        web_pages = state.get("web_pages") or []
        kb_top = max((r["scores"]["total"] for r in kb_hits), default=0.0)
        known_dests = {c for _, c in KNOWN_DESTINATIONS}
        dest_covered = bool(goal.get("destination")) and goal["destination"] in known_dests
        # Bias guard (CP 3.1): weak, off-region chunks are NOT evidence for this destination.
        # If local evidence is thin, suppress them — the live-web section carries the
        # grounding (flagged unverified), or we say plainly that nothing was found.
        if not (dest_covered and kb_top >= WEAK_KB_TOP):
            kb_hits = []

        dest = goal.get("destination") or "your trip"
        intro_parts: List[str] = []
        research_leads = state.get("research_leads") or []
        # Luxury GLOBAL view (no destination named): the section below is a real,
        # key-free Wikipedia-sourced answer — say what it IS up front instead of the
        # generic "couldn't find anything" fallback, and invite the user to go deep.
        if (state.get("luxury") or {}).get("by_country", {}).get("countries"):
            intro_parts.append(
                "You asked where the best luxury experiences are — here's an honest, key-free read on "
                "where the world's fine-dining concentrates (Wikipedia's Michelin three-star list — a "
                "factual count of listed restaurants, not a subjective \u201cbest\u201d ranking). "
                "Name any one of those places and I'll go deep on its 4\u20135\u2605 stays, premium "
                "experiences, and three-star tables."
            )
        # Honest about WHY nothing was found: when the web is off (WAYFINDER_WEB=0), the
        # KB is the only source; when it is on (the default), the web is the second
        # source and may simply be unreachable or off-target.
        web_note = (" (live-web research is off in this deployment)"
                    if not WEB_ENABLED else " (the live web was unreachable or off-target)")
        if not (kb_hits or state.get("flights") or state.get("hotels") or web_pages or research_leads
                or (state.get("seasons") or {}).get("covered")
                or (state.get("local_places") or {}).get("places")
                or (state.get("luxury") or {}).get("by_country", {}).get("countries")):
            # Nothing solid from the KB or the live web — plain fallback instead of
            # citing off-region chunks as evidence (CP 3.1 bias guard).
            if state.get("ask_capital"):
                intro_parts.append(
                    "I've searched far and wide, but I just couldn't find what you might have been looking for — "
                    "and I don't have a place in our conversation yet to pin 'the capital' to. "
                    "Which capital do you mean? Tell me the country or state — e.g. \u201cthe capital of France\u201d "
                    "or \u201cthe capital of New York\u201d — and I'll do my best with what I have on hand."
                )
            elif state.get("capital_resolved"):
                # Offline capital resolution succeeded but KB evidence is thin —
                # still state the fact (it comes from the world dataset, not the
                # web) and be plain about the thin notes.
                cr = state["capital_resolved"]
                intro_parts.append(
                    f"You asked for \u201c{cr.get('ref')}\u201d — from our conversation about {cr['from']}, "
                    f"that's **{cr['to']}**, the capital of {cr['from']} ({cr['kind']}). "
                    f"My notes on {cr['to']} itself are thin right now, so treat this as a starting "
                    "point and double-check local details."
                )
            elif state.get("resolve_query"):
                intro_parts.append(
                    "I've searched far and wide, but I just couldn't find what you might have been looking for. "
                    f"I tried to resolve \u201c{state.get('resolve_ref')}\u201d from our conversation about "
                    f"{state.get('prev_dest') or 'your last destination'}, but my local knowledge base doesn't "
                    f"cover it{web_note}. I have guides for 197 countries, plus deep dives on the Cayman "
                    "Islands / George Town and Kennywood / Pittsburgh — or name the place directly and I'll "
                    "look it up."
                )
            elif dest == "your trip":
                intro_parts.append(
                    "I've searched far and wide, but I just couldn't find what you might have been looking for. "
                    "Could you tell me which destination you're thinking about? I have guides for 197 "
                    "countries, plus deep dives on the Cayman Islands / George Town and Kennywood / "
                    "Pittsburgh — those are the places I'm strongest on."
                )
            else:
                intro_parts.append(
                    "I've searched far and wide, but I just couldn't find what you might have been looking for. "
                    f"My local knowledge base doesn't cover {dest}{web_note}. I have guides for 197 "
                    "countries, plus deep dives on the Cayman Islands / George Town and Kennywood / "
                    "Pittsburgh — try one of those, or name a country or city and I'll do my best."
                )
        # Safety honesty: the user asked about safety but no official advisory could be
        # confirmed (unreachable, unresolvable, or web off) — say so plainly instead of
        # inventing a rating (honesty model: never a guess). Only meaningful with a
        # specific destination — a destination-less global view (e.g. 'where are the best
        # luxury places?') has no place to be safe about, so it must not nag about advisories.
        if ("safety" in (goal.get("focus") or [])
                and (goal.get("destination") or state.get("prev_dest"))
                and not (state.get("safety") or {}).get("found")):
            intro_parts.append(
                f"I also searched for an official travel advisory for {dest} and couldn't confirm one — "
                "I won't invent a safety rating. Check travel.state.gov directly before deciding."
            )
        else:
            if state.get("capital_resolved"):
                # OFFLINE capital resolution: state the answer up front (the fact
                # comes from the 197-country world dataset, not the web), then
                # offer what the KB has on the NEW place.
                cr = state["capital_resolved"]
                intro_parts.append(
                    f"You asked for \u201c{cr.get('ref')}\u201d — from our conversation about "
                    f"{cr['from']}, that's **{cr['to']}**, the capital of {cr['from']} "
                    f"({cr['kind']}). "
                    + (f"Here's what I have on {cr['to']} and the wider area: " if kb_hits
                       else f"My notes below cover the wider area — {cr['to']} itself may be thin "
                            "in my dataset, so double-check local details. ")
                )
            elif state.get("resolve_query") and web_pages:
                resolved = state.get("resolved_place") or dest
                n_searches = len(state.get("web_searches", []))
                if state.get("prev_dest"):
                    intro_parts.append(
                        f"You asked for \u201c{state.get('resolve_ref')}\u201d — from our conversation about "
                        f"{state['prev_dest']} I resolved that to **{resolved}** and researched it on the live web: "
                        f"{n_searches} search(es), {len(web_pages)} page(s) fetched, summarized below "
                        "(community-sourced — double-check before booking)."
                    )
                else:
                    intro_parts.append(
                        f"You asked for \u201c{state.get('resolve_ref')}\u201d — I resolved that to **{resolved}** and "
                        f"researched it on the live web: {n_searches} search(es), {len(web_pages)} page(s) fetched, "
                        "summarized below (community-sourced — double-check before booking)."
                    )
            else:
                # The destination-less luxury GLOBAL view already opens with its own
                # sourced intro — the generic 'grounded plan' line is redundant (and a
                # global overview isn't a concrete 'plan'), so skip it there.
                if not (state.get("luxury") or {}).get("by_country", {}).get("countries"):
                    intro_parts.append(f"Here's a grounded plan for {dest}.")
                if (state.get("seasons") or {}).get("covered"):
                    intro_parts.append(
                        "You asked about the best time to go — the seasonal picture below comes from my "
                        "key-free climate dataset (real 2023-2025 climate normals + public holidays), "
                        "not a live forecast."
                    )
                if goal.get("budget"):
                    intro_parts.append(f"Budget ${goal['budget']}, ~{goal.get('nights', 3)} nights.")
                if not kb_hits and not state.get("flights") and not state.get("hotels") \
                        and (web_pages or research_leads):
                    intro_parts.append(
                        f"My local knowledge base doesn't cover this destination, so I researched {dest} on the "
                        f"live web — {len(state.get('web_searches', []))} search(es)"
                        + (f", {len(web_pages)} page(s) fetched and indexed with provenance" if web_pages else "")
                        + (f", {len(research_leads)} cross-source lead(s)" if research_leads else "")
                        + " (community-sourced). Double-check the details before booking."
                    )
        intro = " ".join(intro_parts)

        # best time to visit — the key-free seasons dataset (climate normals + holidays).
        # Shown before hotels: WHEN to go is the trip-planning question this run is about.
        seasons = state.get("seasons") or {}
        if seasons:
            scountry = seasons.get("country") or dest
            if seasons.get("covered"):
                lines = [
                    f"**Ideal months:** {', '.join(seasons.get('ideal', []))} · "
                    f"**Best overall:** **{seasons.get('best_month', '')}**",
                    f"**Shoulder:** {', '.join(seasons.get('shoulder', []))} · "
                    f"**Off-season:** {', '.join(seasons.get('off', []))}",
                ]
                if seasons.get("reference_point"):
                    lines.append(
                        f"Climate reference point: **{seasons['reference_point']}** "
                        f"({seasons.get('climate_years', '2023-2025')} normals)."
                    )
                mv = seasons.get("month_verdict")
                if mv and mv.get("verdict"):
                    lines.append(f"**Your month ({mv.get('name', mv.get('month'))}):** {mv['verdict']}")
                if seasons.get("prose"):
                    prose = seasons["prose"]
                    if len(prose) > 420:
                        prose = prose[:417] + "…"
                    lines.append(f"**Why (summarized from Wikivoyage):** {prose}")
                hol = seasons.get("holidays") or []
                if hol:
                    hol_bits = ", ".join(
                        h.get("name", "") + (f" ({h['date']})" if h.get("date") else "") for h in hol[:8])
                    more = f" +{len(hol) - 8} more" if len(hol) > 8 else ""
                    lines.append(f"**Public holidays:** {hol_bits}{more}.")
                sections.append(
                    f"## 📅 Best time to visit — {scountry}\n" + "\n".join(lines)
                    + "\n_Derived from REAL climate normals (Open-Meteo 2023-2025) + Wikipedia public "
                      "holidays + Wikivoyage climate notes — key-free public sources, no API keys. "
                      "Ideal/shoulder/off are a comfort+dryness estimate over those normals, not a "
                      "forecast — events and local festivals can shift the real sweet spot._"
                )
            else:
                sections.append(
                    f"## 📅 Best time to visit — {scountry or dest}\n"
                    + (seasons.get("note") or "I don't have verified seasonal data for this place yet.")
                    + "\n_I won't invent months — say the word and I'll research the best time on the "
                      "live web instead (community-sourced — double-check before booking)."
                )

        # travel safety — the #1 traveler question (green->yellow->red meter from the
        # U.S. State Department advisory). Renders BEFORE hotels: safety is the info
        # the user needs first (CP 1.1: "at the top of the page, list the info the
        # user needs"). Two-signal, safety-first reconciliation (confirmed / live /
        # snapshot / conflict); if no rating could be confirmed this section is absent
        # and the intro fallback says so plainly — never a guess.
        safety = state.get("safety") or {}
        if safety.get("found"):
            s_country = safety.get("country") or dest
            if safety.get("status") == "conflict":
                s_status = (
                    f"⚠️ **Sources disagree** — the live official source ({safety.get('live_source') or 'official page'}) "
                    f"shows Level {safety.get('live_level')} and the data snapshot shows Level "
                    f"{safety.get('snapshot_level')}; I'm showing the **more cautious** rating. "
                    "Check travel.state.gov directly before any decision."
                )
            else:
                s_status = {
                    "confirmed": "(confirmed: live official source and the data snapshot agree)",
                    "live": "(live official source — the data snapshot was unreachable)",
                    "snapshot": ("(official data snapshot — the live page was unreachable; "
                                 "verify at travel.state.gov)"),
                }.get(safety.get("status"), "")
            sections.append(
                f"## 🛡️ Travel safety — {s_country}\n"
                f"**Safety meter:** {safety.get('meter', '')}  **{safety.get('dot', '')} {safety.get('rating', '')}**\n"
                f"U.S. Department of State advisory: **Level {safety.get('level')} — {safety.get('advisory', '')}**"
                + (f"\n{s_status}" if s_status else "")
                + f"\n_Source: U.S. Department of State travel advisory for {s_country} — official "
                  "government data, key-free. Advisories can change; re-check at "
                  "https://travel.state.gov close to your travel dates._"
            )

        # Safety & emergency services — REAL mapped hospitals/clinics, pharmacies and
        # local authorities (police/fire/ambulance/coast guard) near the destination,
        # from key-free OpenStreetMap. Rendered right after the advisory when the run
        # is a health/safety-services run (planner focus 'health'). Honesty model:
        # only what OSM actually maps — an empty bucket is said plainly, never filled.
        # `ss_rendered` tells the local-places renderer below to skip its inline
        # health block (this section is the richer, dedicated home for it).
        ss_rendered = False
        if "health" in (goal.get("focus") or []):
            from . import osm as _osm
            ss_places = (state.get("local_places") or {}).get("places") or []
            ss_hosp = [p for p in ss_places if p.get("category") in {"hospital", "clinic", "centre", "doctors"}]
            ss_pharm = [p for p in ss_places if p.get("category") == "pharmacy"]
            ss_auth = [p for p in ss_places if p.get("category") in _osm.AUTHORITY_CATEGORIES]

            def _ss_row(p: dict) -> str:
                tail = " · ".join(x for x in (p.get("address"), p.get("operator")) if x)
                return f"• **{p.get('name')}** ({p.get('category')})" + (f" — {tail}" if tail else "")

            ss_blocks: List[str] = []
            if ss_hosp:
                ss_blocks.append("**🏥 Hospitals & clinics:**\n" + "\n".join(_ss_row(p) for p in ss_hosp[:8]))
            if ss_pharm:
                ss_blocks.append("**💊 Pharmacies:**\n" + "\n".join(_ss_row(p) for p in ss_pharm[:8]))
            if ss_auth:
                ss_blocks.append("**🚓 Local authorities — police, fire, ambulance:**\n"
                                 + "\n".join(_ss_row(p) for p in ss_auth[:8]))
            if ss_blocks:
                sections.append(
                    f"## 🚑 Safety & emergency services — {dest}\n"
                    + "\n\n".join(ss_blocks)
                    + "\n_Source: OpenStreetMap community data, key-free (names/addresses as "
                      "mapped — verify hours and phone numbers before relying on them). In a "
                      "life-threatening emergency, call the local emergency number or ask your "
                      "hotel front desk for the fastest route to care._"
                )
                ss_rendered = True
            elif ss_places:
                # OSM returned places but none in the safety categories — say so plainly.
                sections.append(
                    f"## 🚑 Safety & emergency services — {dest}\n"
                    "I checked the mapped community data (OpenStreetMap, key-free) around "
                    f"{dest} but found no hospitals, pharmacies or police/fire stations "
                    "close enough to list — I won't guess at names. Ask your hotel front "
                    "desk for the nearest hospital, pharmacy and police station, or search "
                    "'hospital near me' once you're on the ground."
                )
                ss_rendered = True
            else:
                # No OSM data at all for this run (offline, API down, or no match).
                sections.append(
                    f"## 🚑 Safety & emergency services — {dest}\n"
                    "I couldn't verify nearby hospitals, pharmacies or police/fire stations "
                    f"for {dest} right now — I won't list unverified names. Ask your hotel "
                    "front desk for the nearest hospital, pharmacy and police station."
                )
                ss_rendered = True

        # hotels — TOP of the answer: the info the user needs first. Where to stay
        # near the airport & the main destinations people actually go (CP 1.1).
        # Sample data, clearly labeled — real rates vary; confirm before booking.
        hotels = state.get("hotels") or []
        if hotels:
            def _h_row(h: dict) -> str:
                dist = []
                if h.get("distance_to_airport_mi") is not None:
                    dist.append(f"{h['distance_to_airport_mi']} mi to the airport")
                if h.get("distance_to_excursion_mi") is not None:
                    dist.append(f"{h['distance_to_excursion_mi']} mi to the main spots")
                return (f"• **{h.get('name')}** — ${h.get('price_usd')}/night, "
                        f"{h.get('stars', 3)}★, {h.get('rating', 4.0)} rating"
                        + (f" · {' · '.join(dist)}" if dist else "")
                        + (f" ({h['area']})" if h.get("area") else ""))
            near_airport = [h for h in hotels if (h.get("distance_to_airport_mi") or 99) <= 5]
            near_airport.sort(key=lambda h: (h.get("distance_to_airport_mi") or 99, int(h.get("price_usd", 0))))
            near_ids = set(map(id, near_airport))
            main_spots = [h for h in hotels if id(h) not in near_ids]
            main_spots.sort(key=lambda h: (h.get("distance_to_excursion_mi") or 99, int(h.get("price_usd", 0))))
            blocks: List[str] = []
            shown: List[dict] = []
            if near_airport:
                blocks.append("**Near the airport:**\n" + "\n".join(_h_row(h) for h in near_airport[:3]))
                shown += near_airport[:3]
            if main_spots:
                blocks.append("**City center / main destinations:**\n" + "\n".join(_h_row(h) for h in main_spots[:3]))
                shown += main_spots[:3]
            if not blocks:
                blocks.append("\n".join(_h_row(h) for h in hotels[:4]))
                shown = hotels[:4]
            # 💎 SPLURGE TIER (luxury runs): the HIGHER-priced rooms — priciest first —
            # so "a more expensive option / tier a more expensive room" is always on the
            # table (real sample prices, labeled as sample, CP 3.1). Plus the luxury
            # pass's OSM-mapped 4–5★ stays: real names, no sample price — say so.
            if "luxury" in (goal.get("focus") or []):
                shown_ids = {id(h) for h in shown}
                priced = [h for h in hotels if h.get("price_usd") is not None]
                splurge = sorted(
                    (h for h in priced if id(h) not in shown_ids),
                    key=lambda h: (-int(h.get("price_usd", 0)), -(h.get("stars") or 0)))
                if splurge:
                    blocks.append("**💎 Splurge tier — pricier, top-rated rooms (tier up here):**\n"
                                  + "\n".join(_h_row(h) for h in splurge[:3]))
                else:
                    # The priciest rooms are ALREADY in the picks above — say which one is
                    # the tier-up, and point at the higher room categories (suite/penthouse)
                    # that base-rate sample data doesn't carry.
                    top = sorted(priced, key=lambda h: (-int(h.get("price_usd", 0)),
                                                        -(h.get("stars") or 0)))
                    if top:
                        t = top[0]
                        blocks.append(
                            f"**💎 Tier up:** {t.get('name')} at **${t.get('price_usd')}/night** is the "
                            f"priciest room on this list ({t.get('stars')}★). For the top room tier "
                            "at any of these properties (suite / penthouse / best view), ask the "
                            "hotel directly — sample data carries base rates only.")
                stays_with_addr = [p for p in ((state.get("luxury") or {}).get("stays") or [])
                                   if p.get("address")]
                for p in stays_with_addr[:3]:
                    blocks.append(
                        f"• **{p.get('name')}** — 4–5★ mapped stay · {p['address']} — no sample "
                        "price (community-mapped); ask the hotel for its top room tier (suite / penthouse).")
            sections.append(
                f"## 🏨 Where to stay — {dest}\n" + "\n\n".join(blocks)
                + "\n_Sample data for the demo (never charged) — real rates vary by season and dates; "
                "confirm availability before booking._"
            )

        # local places — REAL restaurant/cafe/bar names + street addresses and tourist
        # attractions near the destination (key-free OpenStreetMap: Nominatim geocode +
        # Overpass radius query). Rendered right after hotels: where to eat / what to see
        # is the next thing people want (CP 1.1). Community-mapped data, never invented —
        # if the lookup failed or found nothing, `places` is empty and this section is
        # absent (compose's honest fallback already covers the no-data case).
        lp = state.get("local_places") or {}
        lp_places = lp.get("places") or []
        lp_splurge = False  # set below when luxury entries fold into this section
        if lp_places:
            from . import osm as _osm  # local import: keeps agent.py import-light when web is off
            dining = [p for p in lp_places if p.get("category") in _osm.DINING_CATEGORIES]
            health = [] if ss_rendered else [p for p in lp_places if p.get("category") in _osm.HEALTH_CATEGORIES]
            authorities = [p for p in lp_places if p.get("category") in _osm.AUTHORITY_CATEGORIES]
            sights = [p for p in lp_places if p.get("category") not in _osm.DINING_CATEGORIES
                      and p.get("category") not in _osm.HEALTH_CATEGORIES
                      and p.get("category") not in _osm.AUTHORITY_CATEGORIES]

            def _lp_row(p: dict) -> str:
                tail = " · ".join(x for x in (p.get("cuisine"), p.get("address")) if x)
                return (f"• **{p.get('name')}** ({p.get('category') or 'place'})"
                        + (f" — {tail}" if tail else ""))

            resolved = lp.get("resolved_as") or dest
            lp_blocks: List[str] = []
            if dining:
                lp_blocks.append("**Where to eat & drink:**\n" + "\n".join(_lp_row(p) for p in dining[:8]))
            if health:
                lp_blocks.append("**Health & emergency — pharmacies, hospitals, clinics:**\n"
                                 + "\n".join(_lp_row(p) for p in health[:8]))
            if authorities and not ss_rendered:
                lp_blocks.append("**Local authorities — police, fire, ambulance:**\n"
                                 + "\n".join(_lp_row(p) for p in authorities[:8]))
            if sights:
                lp_blocks.append("**Things to see & do:**\n" + "\n".join(_lp_row(p) for p in sights[:8]))
            if not lp_blocks:
                lp_blocks.append("\n".join(_lp_row(p) for p in lp_places[:10]))
            # 💎 Luxury entries IN the things-to-do section (luxury runs): the high-cost
            # stuff — premium experiences (golf · spa · marina · wine) + Michelin
            # three-star — folded into 'Things to see & do' so splurge options live
            # where people look for things to do. Same key-free, never-invented
            # sources as the standalone luxury section (CP 3.1).
            lux_state = state.get("luxury") or {}
            lux_exps = lux_state.get("experiences") or []
            mich = lux_state.get("michelin") or {}
            if "luxury" in (goal.get("focus") or []) and (lux_exps or mich.get("count")):
                spl = []
                for p in lux_exps[:6]:
                    star = (p.get("stars") or "").strip()
                    tag = star + "★" if star else (p.get("operator") or "")
                    tail = " · ".join(x for x in (tag, p.get("address")) if x)
                    spl.append(f"• **{p.get('name')}** ({p.get('category') or 'experience'})"
                               + (f" — {tail}" if tail else ""))
                if mich.get("count"):
                    own = set(mich.get("city_priority") or [])
                    for r in (mich.get("restaurants") or [])[:5]:
                        mark = " ★" if r.get("name") in own else ""
                        spl.append(f"• **{r.get('name')}** — Michelin three-star, "
                                   f"{r.get('city') or mich.get('country') or ''}{mark}")
                if spl:
                    lp_blocks.append(
                        "**💎 The expensive ones (splurge things to do — golf · spa · marina · wine · fine dining):**\n"
                        + "\n".join(spl))
                    lp_splurge = True
            heading = (f"## 🏥 Health & emergency — {resolved}"
                       if health and not dining and not sights
                       else f"## 🍽️ Local places — {resolved}")
            section_text = heading + "\n" + "\n\n".join(lp_blocks)
            if lp.get("fact"):
                section_text += f"\n\n_{lp['fact']} (Wikidata)_"
            sections.append(
                section_text
                + "\n_Source: OpenStreetMap community data via key-free Nominatim + Overpass — "
                "names/addresses as mapped; verify hours & prices before visiting._"
            )

        # luxury experiences — 4-5★ stays + premium experiences (OSM) + Michelin
        # three-star fine dining (Wikipedia list, official API). Rendered right after
        # local places: the upscale cut of the same question. Empty buckets are
        # dropped (compose's honest fallback covers the no-data case); an honest-gap
        # note is carried when a bucket failed.
        lux = state.get("luxury") or {}
        lux_stays = lux.get("stays") or []
        lux_exps = lux.get("experiences") or []
        mich = lux.get("michelin") or {}
        byc = lux.get("by_country") or {}
        # On a luxury run the splurge content is already folded into the sections
        # above — tier-up stays live in 'Where to stay', the expensive things to do
        # live in 'Local places'. This standalone section then only carries what was
        # NOT folded in (the destination-less global view, or buckets whose host
        # section didn't render, e.g. OSM found no places).
        if "luxury" in (goal.get("focus") or []):
            if hotels:      # 'Where to stay' rendered -> the 4–5★ stays folded in there
                lux_stays = []
            if lp_splurge:  # 'Local places' rendered the splurge block -> dedupe
                lux_exps = []
                mich = {}
        if lux_stays or lux_exps or mich.get("count") or byc.get("countries"):
            def _l_row(p: dict) -> str:
                star = (p.get("stars") or "").strip()
                op = (p.get("operator") or "").strip()
                tag = star + "★" if star else op
                tail = " · ".join(x for x in (tag, p.get("address")) if x)
                return (f"• **{p.get('name')}** ({p.get('category') or 'place'})"
                        + (f" — {tail}" if tail else ""))

            lux_blocks: List[str] = []
            resolved = lux.get("resolved_as") or lux.get("destination") or dest
            if byc.get("countries"):
                # destination-less: the honest global view (factual concentration, not opinion)
                top = ", ".join(f"{c['country']} — {c['three_star']}" for c in byc["countries"][:8])
                lux_blocks.append(
                    f"**Where the world's three-star dining concentrates** ({byc.get('total', 0)} listed): "
                    f"{top}, …"
                )
            if lux_stays:
                lux_blocks.append("**Upscale stays (4–5★):**\n" + "\n".join(_l_row(p) for p in lux_stays[:6]))
            if lux_exps:
                lux_blocks.append("**Premium experiences (golf · spa · marina · wine):**\n"
                                  + "\n".join(_l_row(p) for p in lux_exps[:6]))
            if mich.get("count"):
                own = set(mich.get("city_priority") or [])
                mrows = []
                for r in mich["restaurants"][:8]:
                    mark = " ★" if r["name"] in own else ""
                    mrows.append(f"• **{r['name']}** — {r['city'] or mich.get('country')}"
                                 + (f" (since {r['since']})" if r.get("since") else "") + mark)
                lux_blocks.append(
                    f"**Fine dining — Michelin three-star, {mich.get('country')}** (★ = the city you named)\n"
                    + "\n".join(mrows))
            if lux.get("notes"):
                lux_blocks.append("_Honest gaps: " + " ".join(lux["notes"]) + "_")
            sections.append(
                f"## 💎 Luxury experiences — {resolved}\n" + "\n\n".join(lux_blocks)
                + "\n_Source: OpenStreetMap (4–5★ stays & experiences, as community-mapped) + Wikipedia "
                "'List of Michelin 3-star restaurants' (official API, key-free). Star ratings and the "
                "three-star list are as recorded by their sources — verify before booking."
            )

        # flights
        flights = state.get("flights") or []
        if flights:
            rows = []
            for f in flights[:4]:
                rows.append(
                    f"• **{f.get('airline')} {f.get('flight_no')}** — {f.get('date')} "
                    f"{f.get('dep_time')}→{f.get('arr_time')} ({f.get('cabin')}), **${f.get('price_usd')}** "
                    f"({f.get('origin')}→{f.get('destination')})"
                )
            sections.append("## ✈️ Flights\n" + "\n".join(rows) +
                            "\n_Source: sample flight schedule. "
                            "Prices are sample data for the demo._")

        # booking status
        if state.get("booking"):
            b = state["booking"]
            f = b.get("flight") or {}
            if f:
                sections.append(
                    f"## 🎫 Booking\nBookTicket() is **PENDING** for {f.get('airline')} {f.get('flight_no')} — "
                    "I only book with your permission per request. "
                    "Reply **confirm** and I'll issue the (sample) ticket."
                )

        # knowledge highlights with citations
        if kb_hits:
            lines = []
            for r in kb_hits[:5]:
                first = r["text"].strip().split("\n")[0]
                if len(first) > 220:
                    first = first[:217] + "…"
                n = cite({"id": r["id"], "title": r["title"], "source": r["source"], "tier": r["tier"],
                          "date": r["date"], "url": r["url"], "text": r["text"],
                          "scores": r["scores"]})
                lines.append(f"• {first}  [{n}]")
            sections.append("## 🧭 Know before you go (retrieval-grounded, CP 3.1)\n" + "\n".join(lines))

        # live web (fetched this run) — tier 'web' = unverified, flagged per CP 3.1 bias guard.
        # Deeper research (CP 2.1): up to WEB_MAX_PASSES pages were fetched (pass 1 = the place,
        # pass 2 = facts/history) — so this section SUMMARIZES what the agent actually saw:
        #   In short        — the lead of the best page
        #   Digging deeper  — the second page
        #   At a glance     — fact-like lines (year/number) across all pages
        if web_pages:
            def _page_cite(p: dict) -> tuple:
                bullets = _web_bullets((p.get("text") or "").strip(), 2)
                n = cite({
                    "id": p.get("id") or p.get("url"),
                    "title": p.get("title") or p.get("url"),
                    "source": p.get("provider") or "web",
                    "tier": "web",
                    "date": p.get("date", ""),
                    "url": p.get("url", ""),
                    "text": (" ".join(bullets[:2]) or (p.get("text") or ""))[:400],
                    "provider": p.get("provider") or "web",
                    "scores": {"relevance": 0.5, "reliability": 0.5, "recency": 1.0, "phrase": 0.0, "total": 0.5},
                })
                return n, bullets

            lines: List[str] = []
            p1 = web_pages[0]
            n1, b1 = _page_cite(p1)
            t1 = p1.get("title") or p1.get("url")
            if b1:
                lines.append(f"**In short** (from \u201c{t1}\u201d):")
                for b in b1:
                    if len(b) > 240:
                        b = b[:237] + "…"
                    lines.append(f"• {b}  [{n1}]")
            else:
                lines.append(f"• Fetched \u201c{t1}\u201d — no quotable lead paragraph  [{n1}]")
            if len(web_pages) > 1:
                p2 = web_pages[1]
                n2, b2 = _page_cite(p2)
                t2 = p2.get("title") or p2.get("url")
                lines.append("")
                lines.append(f"**Digging deeper** (from \u201c{t2}\u201d):")
                if b2:
                    for b in b2:
                        if len(b) > 240:
                            b = b[:237] + "…"
                        lines.append(f"• {b}  [{n2}]")
                else:
                    lines.append(f"• Fetched \u201c{t2}\u201d — no quotable lead paragraph  [{n2}]")
            facts = _web_facts(web_pages, 4)
            if facts:
                lines.append("")
                lines.append("**At a glance:**")
                lines.extend(f"• {f}" for f in facts)
            resolved = (state.get("resolved_place") or "").strip()
            sections.append(
                f"## 🌐 {resolved or dest} — summarized from the live web (community-sourced)\n"
                + "\n".join(lines)
                + "\n_Fetched and indexed this run into the web crawl store with full provenance (url, provider, "
                "fetch date). Community-sourced — double-check before booking._"
            )

            # Reference-resolution alternates (CP 2.1): the first search pass is the
            # disambiguator — offer the other concrete places it surfaced (excluding the
            # already-fetched page, the previous place, and lists/concept articles).
            alts: List[dict] = []
            if state.get("resolve_results"):
                fetched_urls = set(state.get("web_fetched_urls", []))
                prev_norm = (state.get("prev_dest") or "").lower().replace(" ", "").replace(",", "")
                for r in state["resolve_results"]:
                    t = (r.get("title") or "").lower()
                    tn = t.replace(" ", "").replace(",", "")
                    if "disambiguation" in t or t.startswith(("list of", "map of", "category:", "capital")):
                        continue
                    if r.get("url") in fetched_urls:
                        continue
                    if prev_norm and (prev_norm in tn or tn in prev_norm):
                        continue  # the previous place itself — not the NEW answer
                    alts.append(r)
                    if len(alts) >= 2:
                        break
            if alts:
                names = " or ".join(f"**{a.get('title') or a.get('url')}** ([source]({a.get('url')}))" for a in alts)
                sections.append(
                    f"## 🧭 Or did you mean…\n{names}? Say the word and I'll go deeper on any of them "
                    "(community-sourced — double-check before booking)."
                )

        # Cross-source research leads (the 'more places' agent, now federated): structured,
        # source-labelled leads — title, URL, snippet, source type. Kept separate from the
        # fetched evidence: a lead is a pointer to a public page, not a claim extracted
        # from it. Reddit/Instagram are never fetched behind a login wall (fetchable=False).
        lead_items: List[dict] = []  # structured leads for the UI (per-source dropdowns)
        if research_leads:
            source_order = ("wikipedia", "wikivoyage", "tripadvisor", "expedia", "reddit", "instagram")
            chosen: List[dict] = []
            for source_type in source_order:
                hit = next((r for r in research_leads if r.get("source_type") == source_type), None)
                if hit:
                    chosen.append(hit)
            lines = []
            for hit in chosen:
                snippet = re.sub(r"\s+", " ", hit.get("snippet", "")).strip()
                if len(snippet) > 190:
                    snippet = snippet[:187] + "…"
                source_type = hit.get("source_type", "web")
                n = cite({
                    "id": hit.get("url", "") or f"{source_type}:{hit.get('title', '')}",
                    "title": hit.get("title", "Untitled research lead"),
                    "source": hit.get("source_label", source_type.title()),
                    "tier": "web", "date": hit.get("date", ""), "url": hit.get("url", ""),
                    "text": snippet or "Public indexed discovery result; open the source to verify details.",
                    "provider": hit.get("source_label", source_type.title()),
                    "scores": {"relevance": 0.5, "reliability": 0.5, "recency": 0.5,
                               "phrase": 0.0, "total": 0.5},
                })
                detail = f" — {snippet}" if snippet else ""
                lines.append(f"• **{hit.get('source_label', source_type.title())}**: {hit.get('title', 'result')}{detail}  [{n}]")
                # Structured copy for the frontend dropdown: link + quote + source note.
                lead_items.append({
                    "source_type": source_type,
                    "source_label": hit.get("source_label", source_type.title()),
                    "title": hit.get("title", "result"),
                    "url": hit.get("url", ""),
                    "quote": snippet,
                    "fetchable": bool(hit.get("fetchable", False)),
                    "citation": n,
                })
            sections.append(
                f"## 🗂️ Cross-source research leads — {state.get('resolved_place') or dest}\n" + "\n".join(lines)
                + "\n_Public discovery only: Reddit and Instagram results are links/snippets, never login-wall scraping. "
                  "Tripadvisor and Expedia are leads, not live inventory or pricing. Verify each source before acting._"
            )

        # flight info — pricing + travel time for ANY destination (upon request).
        # 'live_schedule' = real route data from flightconnections.com (unverified tier,
        # cited; fares on it are still labeled estimates); 'estimate' = calibrated
        # distance model (labeled, not live); 'schedule' = the sample OAG-style data
        # (Cayman); 'none' = honest no-estimate (offline / unknown place).
        fi = state.get("flight_info") or {}
        fi_dest = fi.get("destination") or dest
        # Home-base suffix for the flight headings — shown ONLY when this run's
        # origin IS the user's preselected home gateway (an explicit 'from X'
        # wins, so a Nevada-based user asking 'from Miami to …' must not be
        # labeled as flying out of Las Vegas).
        _home = goal.get("home") or {}
        _o = str(goal.get("origin") or "").strip().lower()
        _home_active = bool(_home.get("iata")) and bool(_o) and (
            _o == str(_home.get("city") or "").strip().lower() or _o == _home["iata"].lower())
        home_suffix = (
            f" — from {(_home.get('city') or _home.get('airport'))} ({_home['iata']}), your home base"
            if _home_active else "")
        if fi.get("source") == "amadeus" and fi.get("offers"):
            rows = [f"• **${o.get('price_usd')}** · {o.get('stops')} stop(s) · "
                    f"{', '.join(o.get('carriers') or []) or 'carrier n/a'} · "
                    f"{o.get('departure', '')} → {o.get('arrival', '')}" for o in fi["offers"]]
            sections.append(
                f"## ✈️ Flight offers — to {fi_dest}{home_suffix}\n" + "\n".join(rows)
                + "\n_Source: Amadeus Self-Service API. Free-quota integration; offers can change, so verify "
                  "with the carrier before booking._"
            )
        elif fi.get("source") == "live_schedule" and fi.get("routes"):
            lines = []
            for r in fi["routes"]:
                dur_bits = []
                if r.get("duration_range"):
                    dur_bits.append(f"real schedule times **{r['duration_range'][0]}–{r['duration_range'][1]} h**")
                elif r.get("duration_h"):
                    dur_bits.append(f"fastest direct ~**{r['duration_h']} h**")
                via_bits = f" · via {', '.join(r['via_options'])}" if r.get("via_options") else ""
                airline_bits = f" · airlines: {', '.join(r['airlines'][:6])}" if r.get("airlines") else ""
                fare_bits = (f" · economy **${r['economy_low']}–${r['economy_high']}** (estimate)"
                             if r.get("economy_low") else "")
                dist_bits = f"~{r['distance_km']:,} km · " if r.get("distance_km") else ""
                lines.append(
                    f"**From {r['origin']} ({r.get('origin_code', '')})**: {dist_bits}"
                    f"{'nonstop available' if r.get('direct') else r.get('stops', '1 stop')}{via_bits} · "
                    + (" · ".join(dur_bits) if dur_bits else "time n/a")
                    + fare_bits + airline_bits
                )
            notif = (fi["routes"][0].get("notification") or "").strip()
            sections.append(
                f"## ✈️ Flight info — to {fi_dest}{home_suffix}\n" + "\n".join(lines)
                + f"\n_Typical travel time ~**{fi.get('avg_duration_h')} h**; one-way economy across origins "
                  f"**${fi.get('economy_low')}–${fi.get('economy_high')}** (estimate)"
                  + (f"; schedule: {notif}._" if notif else ".")
                + (f" [Source: flightconnections.com route page]({fi.get('schedule_url')}) — community-sourced; "
                   "verify before booking." if fi.get("schedule_url") else "")
                + "\n_⚠️ Flight times / availability / airlines come from the cited route page (key-free). "
                  "Fares are **estimates — not live pricing**: this deployment uses zero API keys, so real "
                  "prices vary by airline, season and dates — check the carrier before booking._"
            )
        elif fi.get("source") == "estimate" and fi.get("routes"):
            lines = [f"**From** **{r['origin']}**: ~{r['distance_km']:,} km · {r['stops']} · ~{r['duration_h']} h · "
                     f"economy **${r['economy_low']}–${r['economy_high']}** "
                     f"(premium ≈ ${r['premium']}, business ≈ ${r['business']})" for r in fi["routes"]]
            sections.append(
                f"## ✈️ Flight info — to {fi_dest}{home_suffix}\n" + "\n".join(lines)
                + f"\n_Typical travel time ~**{fi.get('avg_duration_h')} h**; one-way economy across origins "
                  f"**${fi.get('economy_low')}–${fi.get('economy_high')}**._"
                + "\n_⚠️ **Estimate — not live pricing**: calibrated distance model over a key-free "
                  "Wikipedia geocode (zero API keys). Real fares vary by airline, season and dates — check the "
                  "carrier for actual prices before booking._"
            )
        elif fi.get("source") == "schedule" and fi.get("flights"):
            rows = [f"• **{f.get('airline')} {f.get('flight_no')}** — {f.get('date')} {f.get('dep_time')}→{f.get('arr_time')} "
                    f"({f.get('cabin')}), **${f.get('price_usd')}** ({f.get('origin')}→{f.get('destination')})"
                    for f in fi["flights"][:4]]
            sections.append(
                f"## ✈️ Flight info — to {fi_dest}\n" + "\n".join(rows)
                + "\n_Source: sample schedule — not live pricing._"
            )
        elif fi.get("source") == "none":
            sections.append(
                f"## ✈️ Flight info — to {fi_dest}\n"
                + ("Live research is off in this deployment, so I can't compute a distance-based "
                   "estimate right now (and I won't guess at prices)."
                   if (fi.get("note") or "").startswith("live research is off")
                   else f"I couldn't compute a distance-based estimate for this destination ({fi.get('note', 'unknown place')}). "
                        "Tell me a specific city and I'll try again — I won't invent prices.")
            )

        # budget fit
        if goal.get("budget"):
            flight_cost = min((int(f.get("price_usd", 0)) for f in flights), default=0)
            if not flight_cost and fi.get("source") in ("estimate", "live_schedule", "amadeus") and fi.get("economy_mid"):
                # No mock schedule for this destination — use the labeled estimate's
                # economy midpoint (still flagged as an estimate, not a live fare).
                flight_cost = int(fi["economy_mid"])
            hotel_cost = min((int(h.get("price_usd", 0)) for h in hotels), default=0) * int(goal.get("nights", 3))
            if flight_cost or hotel_cost:
                est = flight_cost + hotel_cost
                fit = "fits" if est <= goal["budget"] else "exceeds"
                est_note = (" (flight leg from the labeled estimate — not live pricing)"
                            if not flights and fi.get("source") in ("estimate", "live_schedule", "amadeus") else "")
                sections.append(
                    f"## 💰 Budget fit\nCheapest grounded estimate: ${flight_cost} flights + ${hotel_cost} lodging "
                    f"= **${est}** — {fit} your ${goal['budget']} budget.{est_note}"
                )

        followups = []
        if "Cayman" in str(dest):
            followups += [
                "Beachfront (Seven Mile Beach) or George Town for dining/shopping?",
                "Want Stingray City snorkeling on day 2?",
                "Any dietary preferences for the fish fry stops?",
            ]
        elif "Kennywood" in str(dest) or "Pittsburgh" in str(dest):
            followups += [
                "Which rides matter most — Racer, The Beast, or Phoenix?",
                "Weekday or weekend visit? (off-peak pricing)",
                "Should I add a Kennywood + Strip District food crawl?",
            ]
        else:
            # request-log cross-sell (single source of truth: data/request_log.jsonl).
            # GATED on EXPLICIT comparison intent (vs/versus/compare/against): auto-injecting
            # a DIFFERENT destination ("You've also researched X before") into a
            # single-destination answer is exactly the off-region leak the user asked us
            # to remove (a Chile search must not surface New Zealand / the Bahamas).
            # Best-effort; never breaks compose.
            if re.search(r"\b(vs\.?|versus|compare|compared|comparison|against|better than|which is better|side.by.side)\b",
                         (goal.get("raw") or "").lower()):
                try:
                    cur_dest = str(goal.get("destination", "") or "").strip()
                    if cur_dest:
                        for r in recent_requests(self.data_dir, 50):
                            d = str((r or {}).get("destination", "") or "").strip()
                            if d and d.lower() != cur_dest.lower():
                                followups.append(f"You've also researched {d} before — want a side-by-side?")
                                break
                except Exception:
                    pass
            if web_pages or research_leads:
                deeper = state.get("resolved_place") or dest
                followups.append(f"Want me to go deeper on {deeper} — history, food, or day-trips?")
                if not state.get("flight_info") and not flights:
                    followups.append(f"Want flight pricing + travel time to {deeper}? (labeled estimate, not live)")
            if (state.get("seasons") or {}).get("covered"):
                sc = state["seasons"].get("country") or dest
                best_m = state["seasons"].get("best_month") or "the best month"
                followups += [
                    f"Any public holidays in {best_m} I should plan around?",
                    f"Want flight pricing + travel time to {sc} for those dates? (labeled estimate, not live)",
                    "Should I compare two months side by side?",
                ]
            followups += ["Where do you want to stay — near the excursions or the city center?",
                          "Any dates I should lock in?", "Budget for the whole trip, or per item?"]
        if state.get("booking") and state.get("best_flight"):
            followups.insert(0, f"Confirm the {state['best_flight'].get('airline')} {state['best_flight'].get('flight_no')} booking?")
        if (state.get("safety") or {}).get("found"):
            s_c = (state.get("safety") or {}).get("country") or dest
            followups.append(f"Want me to compare {s_c}'s safety rating with another destination?")

        # Structured, intentionally conservative data for the interactive
        # bottom-line calculator.  The UI only calculates from retrieved sample
        # rates / labeled flight estimates and user-controlled local mileage;
        # it never turns a missing price into a made-up one.
        flight_cost = min((int(f.get("price_usd", 0)) for f in flights), default=0)
        flight_note = "sample schedule price"
        if not flight_cost and fi.get("source") in ("estimate", "live_schedule", "amadeus"):
            flight_cost = int(fi.get("economy_mid") or 0)
            flight_note = ("Amadeus Self-Service offer; verify before booking"
                           if fi.get("source") == "amadeus"
                           else "distance-based flight estimate, not live pricing")
        trip_summary = {
            "destination": dest,
            "nights": int(goal.get("nights") or 3),
            "budget": goal.get("budget"),
            "flight_cost": flight_cost or None,
            "flight_note": flight_note if flight_cost else "No flight price retrieved",
            "flight_details": self.build_flight_details(dest, flights, fi),
            "hotels": [{
                "name": h.get("name", "Stay option"),
                "price_usd": h.get("price_usd"),
                "area": h.get("area", ""),
                "airport_mi": h.get("distance_to_airport_mi"),
                "excursion_mi": h.get("distance_to_excursion_mi"),
            } for h in hotels[:8]],
            "safety": ({
                "level": safety.get("level"), "rating": safety.get("rating"),
                "advisory": safety.get("advisory"),
            } if safety.get("found") else None),
            "places": [{
                "name": p.get("name", "Local place"), "category": p.get("category", "place"),
                "address": p.get("address", ""),
            } for p in lp_places[:12]],
        }

        # ---- Quick report (Do / Go / Stay) — concise 'what most people recommend' rollup ----
        # Synthesized from evidence already retrieved this run (no LLM in compose).
        # Rendered at the TOP of the answer in the UI (renderQuickReport) and emitted
        # as the first markdown section for CLI/SSE text consumers. Omitted entirely
        # when there's no destination or no grounded evidence for any bucket (no guesses).
        area_brief = self.build_area_brief(goal, state, kb_hits)
        quick_report = self.build_quick_report(goal, state, kb_hits)
        photos = self.fetch_photos(goal, state)

        top_sections: List[str] = []
        if area_brief:
            top_sections.append(area_brief["md"])
        if quick_report:
            _qr = quick_report

            def _qr_line(label: str, items: List[str]) -> str:
                return f"**{label}:** " + " · ".join(items)

            top_sections.append(
                f"## 🎯 GOALS — {_qr['destination']}\n"
                + _qr_line("Do", _qr["do"])
                + "\n" + _qr_line("Go", _qr["go"])
                + "\n" + _qr_line("Stay", _qr["stay"])
                + f"\n_{_qr['provenance']}_"
            )

        # ---- Photos (landmarks + dining/food scene) ----
        # Real Commons images with caption/artist/license (photos.py); omitted
        # entirely when web is off or every lookup failed (honesty model).
        # The UI replaces this markdown in place with renderPhotos (figures with
        # linked thumbnails); the markdown stays for CLI/SSE text consumers.
        if photos:
            _plines = [f"## 📸 Photos — {dest}"]
            for g in photos:
                _plines.append(f"\n**{g['label']}**")
                for im in g["images"]:
                    _cap = im.get("caption") or im.get("title") or "photo"
                    _credit = " · ".join(x for x in (im.get("artist"), im.get("license")) if x)
                    _plines.append(
                        f"• ![{_cap}]({im.get('thumb_url')}) — {_cap}"
                        + (f" _({_credit} · Wikimedia Commons)_" if _credit else " _(_Wikimedia Commons)_)"))
            top_sections.append("\n".join(_plines))
        sections = top_sections + sections

        return {
            "intro": intro,
            "sections": sections,
            "area_brief": area_brief if area_brief else None,
            "quick_report": quick_report,
            # structured photo groups for the UI (renderPhotos); None when there
            # is nothing grounded to show — the markdown section above is the
            # CLI/SSE parity path (same pattern as quick_report/seasons/leads).
            "photos": photos if photos else None,
            "sources": sources,
            "leads": lead_items,
            # structured seasons for the UI (month chips + mini table + holidays) —
            # the markdown section above is still emitted for CLI/SSE text consumers.
            "seasons": seasons if seasons else None,
            "trip_summary": trip_summary,
            "followups": followups,
            "goal": {k: goal[k] for k in ("destination", "origin", "date", "budget", "nights", "focus", "month", "home") if k in goal},

            # CP 6.1 L4: output scoring — deterministic confidence + band (pass/fail
            # tolerance) + evidence summary, shown to the user for reassurance.
            **guardrails.score_confidence(
                kb_top=kb_top, kb_hits=len(kb_hits), flights=len(flights),
                hotels=len(hotels), web_pages=len(web_pages), places=len(research_leads),
                osm_places=len(lp_places),
                dest_covered=dest_covered,
                flight_estimate=(state.get("flight_info") or {}).get("source") == "estimate",
                flight_live=(state.get("flight_info") or {}).get("source") == "live_schedule",
                seasons=bool((state.get("seasons") or {}).get("covered")),
                safety=bool((state.get("safety") or {}).get("found")),
                luxury=bool((state.get("luxury") or {}).get("by_country", {}).get("countries")
                            or (state.get("luxury") or {}).get("stays")
                            or (state.get("luxury") or {}).get("experiences")),
            ),
        }

    @staticmethod
    def _ev(type_: str, **payload) -> Dict[str, Any]:
        return {"type": type_, "ts": time.strftime("%H:%M:%S"), **payload}
