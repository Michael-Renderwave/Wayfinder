"""Planner (CP 2.1 ReAct: "the agent will have to think through a task and
observe, and it will actively seek out information using Acts. Using the
observation from the action to make the second thought, to then inform the
next item it searches for.").

Two engines, same contract `next_step(goal, state) -> (thought, action, input)`:

  HeuristicPlanner  deterministic, offline, observation-driven (default)
  LLMPlanner        optional OpenAI-compatible model (LM Studio friendly)
                    with automatic fallback to the heuristic on any failure
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLM, LLMError
from .web import WEB_ENABLED  # live-web research: disabled by default (offline-first)
from . import world as _world
from .world import CURATED_DESTINATIONS, ascii_norm, dest_matchers

# World dataset (197 countries) — curated seed destinations PLUS every country,
# alias and city from data/world.json. Matching itself is word-boundary and
# longest-first (world.dest_matchers); this list is the canonical-destination
# set used for the dest_covered gate (keeps the web tier a fallback, not the
# default, for all 197 countries).
KNOWN_DESTINATIONS = CURATED_DESTINATIONS + [(c, c) for c in _world.known_canonicals()]

AIRPORTS = {"miami": "MIA", "atlanta": "ATL", "orlando": "MCO", "george town": "GCM"}

MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "sept": "09", "oct": "10", "nov": "11", "dec": "12",
}

FLIGHT_WORDS = {"flight", "flights", "fly", "flyin", "fly", "ticket", "tickets",
                "book", "booking"}
# Flight-INFO intent (pricing + travel time) — the phrases that specifically
# mean "tell me about flying there", as opposed to bare 'price'/'how much'
# which would wrongly pull flights into a hotel-only question.
FLIGHT_INFO_WORDS = {"travel time", "flight time", "duration", "how much to fly",
                     "how much is a flight", "how much are flights", "flight pricing",
                     "flight price", "flight cost", "flights cost", "flying"}
HOTEL_WORDS = {"hotel", "hotels", "stay", "stays", "lodge", "inn", "accommodation", "where to stay"}
BOOK_WORDS = {"book", "booking", "reserve", "reservations", "hold"}
# Best-time intent (the key-free seasons dataset): the phrases that mean "when
# should I go" as opposed to "where should I go". Detection is phrase-based so a
# bare 'weather' word can't accidentally hijack a booking query.
SEASON_PHRASES = {
    "best time", "best month", "best months", "best season", "best seasons",
    "when to go", "when should i go", "when is the best", "when is best",
    "ideal month", "ideal months", "ideal time", "ideal season",
    "should i go in", "what month", "which month", "good month", "good months",
    "time to visit", "time to travel", "time of year",
    "best time to visit", "best time to travel", "best time to go",
    "weather in", "weather like", "weather during", "best weather",
}
MONTH_FULL = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

TOPIC_TERMS = {
    "dining": ["dining", "restaurant", "restaurants", "food", "eat", "coffee", "cafe", "fish fry"],
    "activities": ["things to do", "what to do", "activities", "activity", "attractions",
                   "attraction", "sights", "sightseeing", "museums", "museum", "fun",
                   "beaches", "beach", "landmarks", "landmark"],
    "safety": ["safety", "safe", "crime", "security"],
    "transport": ["transport", "taxi", "bus", "buses", "shuttle", "drive", "driving"],
    "shopping": ["shop", "shops", "shopping", "gifts", "souvenir", "souvenirs", "ceramics"],
}

# Best ranked KB score below this => local evidence is thin for the destination
# => the agent goes to the live web (CP 2.1). Demo destinations score well above this,
# so the offline demo stays KB-grounded unless a real topic gap remains.
WEAK_KB_TOP = 0.35

# Words that mean "not a place" — if the extracted candidate contains one, the user is
# making a vague follow-up ("how about something cheaper?"), so we must NOT treat it as
# a destination (semantic memory can then inherit the prior turn's context, CP 2.1).
_PLACE_NEG = {
    "something", "anything", "somewhere", "else", "other", "others", "alternative",
    "cheaper", "cheapest", "better", "worse", "different", "similar", "family",
    "families", "friends", "friend", "kids", "children", "people", "group",
    "what", "which", "how", "when", "where", "why",
}
_PLACE_STOP = {
    "the", "a", "an", "and", "or", "of", "with", "me", "please", "trip", "vacation", "tour",
    "visit", "stay", "stays", "hotel", "hotels", "know", "good", "best", "next",
    "this", "last", "soon", "summer", "winter", "spring", "fall", "autumn",
    "restaurants", "restaurant", "food", "dining", "museums", "beaches", "bars",
    "attractions", "activities", "things",
    # verbs / booking nouns: 'plan me a trip', 'Book me a flight', 'confirm' are
    # INTENTS without a place — if one of these survives to a 'destination' it is a
    # false extraction (CP 6.1 L1: clarify instead of guessing on an empty region).
    "plan", "plans", "book", "booking", "bookings", "flight", "flights", "fly",
    "confirm", "confirmed", "reservation", "reservations", "resort", "resorts",
    "discount", "discounts", "cheap", "cheaply", "cost", "costs", "price", "prices",
    "tell", "wanted", "want", "looking", "search", "show", "give", "need",
    # stay-location modifiers: 'in Paris NEAR THE AIRPORT or the MAIN SIGHTS' — the
    # place is Paris; these words describe WHERE in the city, not the city itself.
    "near", "airport", "airports", "airfield", "sights", "sight", "sightseeing",
    "main", "center", "centre", "downtown", "cruise",
    # 'capital' alone is a REFERENCE ('the capital' = the capital of what we were just
    # talking about), not a place — resolving it needs context (agent.py), not a
    # bare web search that would land on the concept article.
    "capital", "capitals",
    # luxury-intent / quality descriptors ('a luxury trip', 'upscale stays', 'yacht
    # charter') and the Michelin brand — these describe the TRIP or the dining, never
    # a place. Without them, _guess_place promotes the adjective to a fake destination
    # (e.g. 'Luxury' geocoding to a building) and hijacks the destination-less
    # 'where are the best places for a luxury trip?' global best-places view.
    "luxury", "luxurious", "upscale", "exclusive", "boutique", "penthouse",
    "yacht", "michelin", "experience", "experiences",
    # bare grammar words (prepositions / determiners / adverbs) that are never a
    # destination — so a place-less question ('best experiences for a luxury trip')
    # cleans to empty and yields the global view instead of a garbage 'For'/'Top'.
    "for", "to", "about", "you", "your", "some", "any", "one", "top", "most",
    "much", "many", "favorite", "favourite",
    # trip-noun synonyms of the existing 'trip'/'vacation'/'tour' — a getaway is a
    # kind of trip, not a place ('a luxury getaway' must not geocode 'Getaway').
    "trips", "getaway", "getaways", "escape", "escapes", "adventure",
    "adventures", "holiday", "holidays",
}


def _guess_place(text: str) -> str:
    """Best-effort place extraction for destinations the seed KB doesn't know.

    'Tell me about California' -> 'California'. Sending the whole sentence to
    Wikipedia once matched 'Tell Me You Love Me (album)' — so we extract the place
    itself. Vague follow-ups ('how about something cheaper?') return "" so semantic
    memory can inherit the previous turn's destination (CP 2.1).
    """
    low = " " + text.strip().lower() + " "

    def _clean(words: List[str]) -> List[str]:
        out: List[str] = []
        for w in words:
            w = w.strip(".,!?;:'- ")
            if w and w not in _PLACE_STOP and w not in _PLACE_NEG:
                out.append(w)
        return out[:4]

    # 1) Drop the origin phrase first ("... from Miami ...") — it's the departure, not the place.
    cut = re.split(r"\s+from\s+[a-z][a-z ]*", low)[0]
    # 2) Drop trailing trip params (dates, budgets, 'next summer').
    cut = re.split(r"\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4}\b|\$\s?\d|\bbudget\b|\b(?:next|this)\s+\w+", cut)[0].strip()
    # 3) Trigger-based extraction ("to California", "about California", "in California").
    #    The place usually sits at the END, so scan candidates from the last trigger back.
    for c in reversed(re.findall(r"\b(?:to|about|in|for)\s+([A-Za-z][A-Za-z' .-]*)", cut)):
        if any(w in _PLACE_NEG for w in c.split()):
            continue  # vague candidate — try the one before it
        kept = _clean(c.split())
        if kept:
            return " ".join(w.capitalize() for w in kept)
    # 4) Bare place name ("California") — but if the phrase contains a "not-a-place" word
    #    it's a vague follow-up ("how about something cheaper?") -> "" so semantic
    #    memory can inherit the previous turn's destination (CP 2.1).
    if any(w in _PLACE_NEG for w in cut.split()):
        return ""
    kept = _clean(cut.split())
    return " ".join(w.capitalize() for w in kept)


# Reference follow-ups (CP 2.1: use the conversation context):
# 'the capital' after an NYC conversation means the capital OF New York — a NEW place
# to research, not the old one and not the word 'capital' as a concept.
REF_CAPITAL_RE = re.compile(
    r"\b(?:the|that|this|its|a)\s+capital\b|\bthe state capital\b|\bthe national capital\b", re.I)
REF_PLACE_RE = re.compile(r"\b(?:that place|this place|the place)\b", re.I)
EXPLICIT_CAPITAL_RE = re.compile(r"\bcapital of ([A-Za-z][A-Za-z' .-]{2,30})\b", re.I)

# Web research depth (CP 1.1 cap, raised from 1): 2 searches + 2 fetches per run when
# the KB is thin — pass 1 = the place, pass 2 = facts/history — so compose() can
# SUMMARIZE what it found instead of quoting a single page.
WEB_MAX_PASSES = 2


def parse_goal(text: str) -> Dict[str, Any]:
    low = text.lower()
    destination = ""
    # World-dataset matching (word-boundary, longest-needle-first, accent-normalized,
    # kills the old 'mali inside malaysia' substring bug,
    # and an explicit 'city, country' phrase anchors on the CITY — 'Rome, Italy'
    # -> Rome, not Italy). 'from X' stays the ORIGIN, never the destination
    # ("...to California from Miami" must not set destination=Miami).
    norm = ascii_norm(low)
    origin_m = re.search(r"\bfrom\s+([a-z ]{3,20})\b", norm)
    if origin_m:
        # 'from X to Y': the origin is X — cap the span at the 'to' boundary so the
        # destination on the other side of it stays matchable ('from Canada to
        # Quebec City' must still find Quebec City).
        o = re.split(r"\s+to\s+", origin_m.group(1))[0]
        origin_span = (origin_m.start(1), origin_m.start(1) + len(o))
    else:
        origin_span = None
    destination = _world.match_destination(text, origin_span)
    if not destination:
        # Unknown place (e.g. 'California'): guess it so the live-web step searches the
        # PLACE, not the whole sentence. It is still 'not covered' (not in
        # KNOWN_DESTINATIONS), so the web path fires (CP 2.1) — and semantic memory won't
        # leak the previous turn's destination into it.
        destination = _guess_place(text)
    origin = ""
    m = re.search(r"\bfrom\s+([A-Za-z ]{3,20})\b", low)
    if m:
        for needle, code in AIRPORTS.items():
            if needle in m.group(1).lower():
                origin = code
                break
    date_str = ""
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        date_str = m.group(0)
    else:
        m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})?\b", low)
        if m:
            month, day = MONTHS[m.group(1)[:3]], m.group(2)
            year = m.group(3) or "2026"
            date_str = f"{year}-{month}-{day.zfill(2)}"
    budget = None
    m = re.search(r"\$\s?(\d[\d,]*)|(\d{3,6})\s*(?:dollars|usd|budget)", low)
    if m:
        budget = int((m.group(1) or m.group(2)).replace(",", ""))
    nights = 3
    m = re.search(r"(\d+)\s*(?:-?\s*day|nights?)\b", low)
    if m:
        nights = max(1, int(m.group(1)) - (1 if "day" in m.group(0).lower() else 0))
    elif "weekend" in low:
        nights = 2
    focus: List[str] = []
    is_trip = any(re.search(rf"\b{w}\b", low) for w in {"trip", "vacation", "itinerary", "getaway"})
    if any(re.search(rf"\b{w}\b", low) for w in FLIGHT_WORDS) or \
            any(re.search(rf"\b{re.escape(w)}\b", low) for w in FLIGHT_INFO_WORDS):
        focus.append("flights")
    if any(re.search(rf"\b{w}\b", low) for w in HOTEL_WORDS):
        focus.append("hotels")
    wants_booking = any(re.search(rf"\b{w}\b", low) for w in BOOK_WORDS) and "flight" in low
    if "dining" in low or "food" in low or "restaurant" in low or "where to eat" in low:
        focus.append("dining")
    if "shop" in low or "gift" in low or "souvenir" in low or "ceramics" in low:
        focus.append("shopping")
    if "safety" in low or "safe" in low:
        focus.append("safety")
    if "transport" in low or "taxi" in low or "bus" in low:
        focus.append("transport")
    # Medical / urgent-care intent -> a distinct 'health' focus (pharmacies, hospitals,
    # clinics, doctors). Kept to explicit medical terms so a general 'is it safe' query
    # still routes to 'safety' rather than suddenly pulling in hospitals.
    # 'police' / 'fire' / 'ambulance' also route here: the safety pass now pulls
    # local authorities (police/fire/ambulance/coast guard) alongside medical care.
    if any(re.search(rf"\b{re.escape(w)}\b", low) for w in (
            "pharmacy", "pharmacies", "hospital", "hospitals", "doctor", "doctors",
            "clinic", "clinics", "medical", "medicine", "urgent care", "emergency room",
            "police", "police station", "fire station", "ambulance", "emergency services")):
        focus.append("health")
    # Luxury intent -> a distinct 'luxury' focus (4-5★ stays + premium experiences
    # from OSM + Michelin three-star fine dining from the Wikipedia list). Kept to
    # explicit upscale words so a general 'best restaurants' query stays standard dining.
    if any(re.search(rf"\b{re.escape(w)}\b", low) for w in (
            "luxury", "luxurious", "five-star", "five star", "four-star", "four star",
            "5-star", "5 star", "4-star", "4 star", "high-end", "high end", "upscale",
            "exclusive", "michelin", "fine dining", "penthouse", "yacht",
            "boutique hotel", "boutique hotels")):
        focus.append("luxury")
    if not focus:
        focus = ["dining", "shopping", "safety", "transport"]
    # CP 1.1: trip planning implies dining/safety/transport follow-through — and
    # where-to-stay is core trip info: surface hotels near the airport & the main
    # destinations people go (shown at the TOP of the answer).
    if is_trip:
        for t in ("dining", "shopping", "safety", "transport"):
            if t not in focus:
                focus.append(t)
        if "hotels" not in focus:
            focus.append("hotels")
    if wants_booking:
        focus.insert(0, "flights")
    # Best-time intent (seasons dataset, key-free): 'best time to visit Japan',
    # 'when should I go to Brazil', 'is March a good month for Rome?' -> the whole
    # run is about WHEN, so focus collapses to ['seasons'] and the web path stays
    # out (the dataset is the source of truth; if it lacks the place, we say so).
    if any(re.search(rf"\b{re.escape(w)}\b", low) for w in SEASON_PHRASES):
        focus = ["seasons"]
        month_m = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october"
            r"|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b", low)
        if month_m:
            g = month_m.group(1)
            month_num = MONTH_FULL.get(g) or MONTHS.get(g) or MONTHS.get(g[:3])
            if month_num:
                return {
                    "raw": text, "destination": destination, "origin": origin,
                    "date": date_str, "budget": budget, "nights": nights,
                    "focus": focus, "month": month_num, "wants_booking": wants_booking,
                }
    return {
        "raw": text,
        "destination": destination,
        "origin": origin,
        "date": date_str,
        "budget": budget,
        "nights": nights,
        "focus": focus,
        "month": None,
        "wants_booking": wants_booking,
    }


class HeuristicPlanner:
    name = "heuristic"

    def __init__(self, llm: LLM):
        self._llm = llm  # kept for mode reporting only

    # -- internal helpers -------------------------------------------------
    # destination -> extra region-hint words for the ranking bias guard:
    # country -> 'capital + top destinations'; city -> its country; the four
    # curated demo entries keep their hand-tuned context (world.dest_context).
    DEST_CONTEXT = _world.dest_context()

    def _kb_query(self, goal: Dict[str, Any], topic: Optional[str] = None) -> str:
        dest = goal.get("destination") or "travel"
        if "seasons" in goal.get("focus", []):
            # Grounding pass for a season run: pull the sNNN best-time doc (and the
            # cNNN guide's 'Good to know' Best line) for citation + a second opinion.
            parts = [dest, "best time to visit", "season", "climate", "weather", "ideal months"]
            if goal.get("month"):
                parts.append("month")
            return " ".join(dict.fromkeys(parts))
        parts = [dest]
        context = self.DEST_CONTEXT.get(dest, "")
        if context:
            parts.extend(context.split())
        if topic == "dining":
            parts.append("dining restaurants food where to eat")
        elif topic == "safety":
            parts.append("safety crime safe")
        elif topic == "transport":
            parts.append("transport taxi bus getting around")
        elif topic == "shopping":
            parts.append("shopping shops gifts souvenirs")
        else:
            topics = [t for t in goal["focus"] if t in TOPIC_TERMS][:3]
            if not topics:
                topics = ["dining", "safety", "transport"]
            for t in topics:
                parts.extend(TOPIC_TERMS[t][:3])
        if goal.get("budget"):
            parts.append("budget prices cost")
        return " ".join(dict.fromkeys(parts))

    @staticmethod
    def _covered(goal: Dict[str, Any], state: Dict[str, Any]) -> set:
        """Which focus topics already have evidence in the observations."""
        covered = set()
        text = ""
        for res in state.get("kb_results", []):
            for r in res.get("results", []):
                text += " " + (r.get("text", "") + " " + r.get("title", "")).lower()
        for topic, terms in TOPIC_TERMS.items():
            if any(t in text for t in terms):
                covered.add(topic)
        if state.get("flights"):
            covered.add("flights")
        if state.get("hotels"):
            covered.add("hotels")
        # real OSM places cover the dining/activities topics (names + addresses
        # ARE the evidence — no KB gap pass needed on top of them)
        if (state.get("local_places") or {}).get("places"):
            covered.add("dining")
            covered.add("activities")
        # luxury results (4-5★ stays + experiences + Michelin three-star) are
        # themselves the dining/activities evidence for a luxury run
        lux = state.get("luxury") or {}
        if lux.get("stays") or lux.get("experiences") or (lux.get("michelin") or {}).get("count"):
            covered.add("dining")
            covered.add("activities")
        return covered

    # -- contract ---------------------------------------------------------
    def next_step(self, goal: Dict[str, Any], state: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        steps = len(state.get("steps", []))
        covered = self._covered(goal, state)
        focus = goal["focus"]
        dest = goal.get("destination") or "your destination"
        recall_hit = state.get("recall_hit")

        # 1. flights first when requested — but only for destinations the mock schedule
        #    actually serves (GCM). Unknown destinations (e.g. 'California') skip to the
        #    live web instead of leaking Cayman flights (CP 1.1: no misleading info).
        #    AND only when the origin is one of the schedule's hubs (or unnamed): the
        #    sample data exists ONLY for MIA/ATL/MCO, so a preselected home base with a
        #    different gateway (Nevada -> Las Vegas, Turkey -> Istanbul) must take the
        #    live/estimate path instead of a mismatched sample row.
        dest_scheduled = goal.get("destination") in {"Cayman Islands", "George Town"}
        _sched_ok = not goal.get("origin") or goal.get("origin") in ("MIA", "ATL", "MCO")
        schedule_usable = dest_scheduled and _sched_ok
        if "flights" in focus and not state.get("flights_done") and schedule_usable:
            thought = (
                f"The user wants {goal.get('origin') or 'a'} -> {dest} flight options"
                + (f" around {goal['date']}" if goal.get("date") else "")
                + (f" within a ${goal['budget']} budget" if goal.get("budget") else "")
                + ". First act: run SearchFlight() against the schedule."
            )
            return thought, "search_flights", {
                "origin": goal.get("origin") or "",
                "destination": "GCM" if goal.get("origin") in ("MIA", "ATL", "MCO") else "",
                "date": goal.get("date") or "",
            }
        # 1b. flight INFO for destinations the mock schedule does NOT serve: the user
        #     asked for pricing + travel time, so give a clearly-labeled distance-based
        #     estimate (key-free geocode) — never live fares, never a silent gap.
        #     Skipped in reference-resolution mode ('the capital' + flight words): the
        #     place is not known yet, so let the web resolve it first.
        if ("flights" in focus and not state.get("flight_info_done") and not schedule_usable
                and not state.get("resolve_query")):
            thought = (
                f"The user wants flight pricing + travel time to {dest}, but the sample schedule "
                "only serves GCM. Act: flight_info() — live route data where available "
                "(flightconnections.com, key-free, cited), else a distance-based fare/duration "
                "estimate. Fares are always labeled estimates, never live pricing."
            )
            return thought, "flight_info", {
                "origin": goal.get("origin") or "",
                "destination": goal.get("destination") or "",
                "date": goal.get("date") or "",
            }

        # 1c. best time to travel (the key-free seasons dataset) — BEFORE the
        #     primary KB pass: the answer comes from data/seasons.json (real
        #     2023-2025 climate normals + Wikipedia holidays + Wikivoyage notes),
        #     never the live web. One grounding KB pass follows for citation, then
        #     compose. If the user named a month, best_time also returns a verdict.
        if "seasons" in focus and not state.get("seasons_done"):
            month = goal.get("month")
            thought = (
                f"The user asks about the BEST TIME for {dest}" + (f" in {month and 'month ' + str(month)}" if month else "")
                + ". Act: best_time() — the key-free seasons dataset (Open-Meteo 2023-2025 climate normals "
                "+ Wikipedia public holidays + Wikivoyage climate notes). Season runs never go to the "
                "live web: the dataset is the source of truth, and if it has no data I say so honestly."
            )
            inp = {"destination": goal.get("destination") or ""}
            if month:
                inp["month"] = int(month)
            return thought, "best_time", inp

        if "flights" in focus and state.get("flights_done") and goal.get("wants_booking") and not state.get("booking_asked"):
            thought = (
                "The user asked me to book. CP 1.1 guardrail: booking happens only with the owner's "
                "permission per request, so I call BookTicket() in PENDING mode and confirm in the answer."
            )
            return thought, "book_ticket", {"flight": state.get("best_flight") or {}, "confirmed": False}

        # 2. primary KB retrieval
        if not state.get("kb_done"):
            thought = (
                f"I need grounded, traceable facts about {dest} for: {', '.join(focus)}. "
                "Act: hard search the vector store (recursive-chunk index) for the best evidence."
            )
            if recall_hit:
                thought += " Episodic memory has a near-duplicate prior query, so I keep this to a single broad pass (beam budget, CP 4.1)."
            return thought, "search_kb", {"query": self._kb_query(goal), "k": 8}

        # 3. hotels — ALSO on luxury runs: "find the higher-priced hotels, or tier a more
        #     expensive room" needs the full price range in hand, so a luxury run pulls the
        #     schedule WITHOUT a budget cap (a cap would hide exactly the splurge tier the
        #     question asks for) and compose() surfaces the pricier 4-5★ tier as a tier-up.
        if ("hotels" in focus or "luxury" in focus) and not state.get("hotels_done"):
            max_price = None
            if goal.get("budget") and "luxury" not in focus:
                max_price = max(80, goal["budget"] // 3)
            if max_price:
                cap_note = f" with a ~${max_price}/night cap derived from the ${goal['budget']} budget"
            elif "luxury" in focus:
                cap_note = (" with NO price cap — the luxury run wants the full price range so the "
                            "pricier 4-5★ tier is available for the splurge tier-up")
            else:
                cap_note = ""
            thought = (
                f"Next: analyze stays near {dest}"
                + cap_note
                + ", ranked by proximity to the airport & the main destinations people go (CP 1.1)."
            )
            return thought, "search_hotels", {
                "destination": dest,
                "max_price": max_price,
                "near": "" if "Cayman" not in dest else "George Town",
            }

        # 3b. LOCAL PLACES (key-free OpenStreetMap): REAL restaurant/cafe/bar names +
        #     street addresses and tourist attractions near the destination — the API
        #     answer to 'dining options, places to go, activities'. One radius pass
        #     per run (L3 budget); 7-day disk cache keeps repeats instant. Triggered
        #     when the user asks for dining/activities, or for general 'tell me about
        #     X' runs (not pure booking/season runs, which stay fast and focused).
        # A luxury run ALSO gets the ATTRACTIONS pass ('things to do'): the splurge
        # entries (premium experiences + Michelin three-star) fold INTO that section,
        # so high-cost things to do land in 'Things to see & do' (CP 1.1: same
        # key-free, never-invented sources as the standalone luxury section).
        wants_places = bool(set(focus) & {"dining", "activities", "health"}) or "luxury" in focus
        general_query = (not any(k in focus for k in ("flights", "seasons"))
                         and "luxury" not in focus
                         and not goal.get("wants_booking"))
        if (WEB_ENABLED and goal.get("destination") and not state.get("local_places_done")
                and (wants_places or general_query)):
            # Include mapped lodging in broad trip runs.  The small local hotel
            # fixture is only a demo; this key-free OSM pass makes unfamiliar and
            # less-touristed destinations useful too. Luxury runs take the attractions
            # cut — the upscale dining/stays come from the luxury pass (Michelin + 4-5★).
            f = ("safety" if "health" in focus else
                 "attractions" if "luxury" in focus else
                 "all" if "hotels" in focus else
                 ("attractions" if ("activities" in focus and "dining" not in focus) else "dining"))
            thought = (
                f"Next: real local places for {dest} — the key-free OpenStreetMap layer "
                f"(Nominatim geocode + Overpass radius query): actual mapped restaurant/cafe/bar "
                f"names with street addresses, attractions, mapped stays, and (when asked) "
                f"pharmacies/hospitals. Names are community-mapped "
                f"data, never invented — if OSM has no match I say so plainly (honesty model)."
            )
            return thought, "local_places", {"place": goal.get("destination") or "", "focus": f, "k": 16}

        # 3b2. LUXURY EXPERIENCES (key-free): 4-5★ stays + premium experiences from
        #      OpenStreetMap (star_rating/golf/spa/marina/winery) + Michelin three-star
        #      fine dining from the Wikipedia list (official API). One pass per run;
        #      30-day disk cache; every bucket degrades honestly when empty (CP 1.1).
        if "luxury" in focus and not state.get("luxury_done"):
            thought = (
                f"Next: LUXURY experiences for {dest} — the key-free luxury layer: "
                "OpenStreetMap 4-5★ hotels/resorts + golf/spa/marina/winery near the anchor "
                "(community-mapped, real names, never invented), plus the destination's "
                "Michelin three-star restaurants from the Wikipedia list (official API). "
                "Any empty bucket is reported honestly instead of being filled in (CP 1.1)."
            )
            return thought, "luxury", {"destination": goal.get("destination") or ""}

        # 3c. TRAVEL SAFETY (official State Dept advisory): green->yellow->red meter.
        #     Multi-signal: OFFICIAL travel.state.gov RSS feed (primary live) + DDG
        #     official-page fallback + GitHub snapshot of the same feed; disagreement ->
        #     the HIGHER risk level
        #     wins and both are disclosed (safety-first). No rating -> honest fallback,
        #     never invented. One call per run (budget 1); 7-day disk cache on the snapshot.
        wants_safety = "safety" in focus
        if (WEB_ENABLED and goal.get("destination") and not state.get("travel_safety_done")
                and (wants_safety or general_query)):
            thought = (
                f"Next: official travel safety for {dest} — the green->yellow->red meter from "
                f"the U.S. Department of State travel advisory (levels 1-4). Safety is the #1 "
                f"traveler question, so this runs before hotels. If sources disagree I show the "
                f"more cautious rating and say so; if I can't confirm one I won't invent a rating."
            )
            return thought, "travel_safety", {"place": goal.get("destination") or ""}

        # 4. targeted gap-filling passes (the 'hard' multi-pass search; depth <= MAX_DEPTH).
        #    SKIPPED in reference-resolution mode ('the capital' after an NYC conversation):
        #    the KB is about the OLD place by definition — gap passes would only burn the
        #    step budget the deeper web research + leads need (CP 1.1 guardrails).
        #    Also SKIPPED for unknown destinations: off-region KB chunks can't fill a real
        #    gap, so escalate straight to the source-aware web research below.
        gaps = [t for t in focus if t in TOPIC_TERMS and t not in covered]
        if gaps and steps < 5 and not state.get("resolve_query") \
                and goal.get("destination") in {c for _, c in KNOWN_DESTINATIONS}:
            topic = gaps[0]
            thought = (
                f"Observation check: I still lack {topic} evidence. ReAct says use the observation to "
                f"inform the next search, so I run a targeted {topic} pass for {dest} (pass {len(state.get('kb_results', [])) + 1})."
            )
            return thought, "search_kb", {"query": self._kb_query(goal, topic), "k": 5}

        # 4b. LIVE WEB (CP 2.1: 'actively seek out information using Acts'; CP 3.1: provenance + unverified tier)
        #     Only when the KB doesn't really cover this destination, local evidence is thin,
        #     OR a topic gap remains. Deeper now (CP 1.1 cap): up to WEB_MAX_PASSES searches +
        #     fetches per run — pass 1 = the place, pass 2 = facts/history — so compose() can
        #     summarize what it found. REFERENCE mode ('the capital' after an NYC conversation):
        #     the search query is the RESOLUTION target ('capital of New York City') and the
        #     fetch guard relaxes — the search itself is the disambiguator (CP 2.1 context).
        kb_top = max(
            (r.get("scores", {}).get("total", 0.0)
             for p in state.get("kb_results", [])
             for r in p.get("results", [])),
            default=0.0,
        )
        known_dests = {c for _, c in KNOWN_DESTINATIONS}
        # An unknown destination means the KB has no dedicated coverage (its top hits are
        # off-region neighbours scoring by lexical overlap, not real evidence — bias guard).
        dest_covered = bool(goal.get("destination")) and goal["destination"] in known_dests
        kb_thin = (not dest_covered) or (kb_top < WEAK_KB_TOP)
        resolve_query = state.get("resolve_query") or ""
        base_place = (state.get("resolved_place") or goal.get("destination")
                      or _guess_place(goal.get("raw") or ""))
        web_query = (resolve_query or base_place)[:80]
        web_searches = state.get("web_searches", [])
        web_passes = state.get("web_passes", 0)
        tried = state.get("web_tried", [])
        fetched_urls = state.get("web_fetched_urls", [])
        # No identifiable place -> no web search: a generic 'travel information' fetch would
        # only pull up junk (CP 3.1 bias guard). Compose will say so plainly instead.
        # Ordering (CP 2.1): search pass 1 -> FETCH that page -> only then the deeper
        # pass 2. Searching twice back-to-back let pass 2's result set displace pass 1's
        # page before it was ever fetched (observed bug: 'the capital' run pulled 'Fact').
        searched = bool(web_searches)
        fetched = web_passes >= 1 or bool(tried)
        if WEB_ENABLED and (kb_thin or bool(gaps)) and web_query and not state.get("web_failed") \
                and len(web_searches) < WEB_MAX_PASSES and "seasons" not in focus:
            if not searched:
                q = web_query
                why = (
                    f"not covered by the local knowledge base (destination '{goal.get('destination') or 'unknown'}')"
                    if not dest_covered
                    else f"weak for this destination (best KB score {kb_top:.2f} < {WEAK_KB_TOP})" if kb_thin
                    else f"gapped on: {', '.join(gaps)}"
                )
                thought = (
                    f"Observation check: local KB evidence is {why}. ReAct says act, not guess — "
                    f"run source-aware research for {q!r}: Wikipedia/Wikivoyage for context, "
                    "Tripadvisor/Expedia for travel leads, and Reddit/Instagram for public community leads. "
                    "The results remain unverified until a public page is fetched and cited."
                )
                return thought, "advanced_search", {"query": q, "k_per_source": 3}
            if fetched and base_place:
                q = f"{base_place} facts history highlights"[:90]
                thought = (
                    f"One page is not enough depth — the user wants insight, not a single article. "
                    f"Act: run a deeper pass, {q!r}, so I can summarize what I actually found "
                    "(CP 2.1: use the observation to drive the next act)."
                )
                return thought, "web_search", {"query": q, "k": 6}
            # Searched but not fetched yet (or no place for a deeper query) -> the
            # fetch gate below consumes the current results first.
        if WEB_ENABLED and state.get("web_results") and not state.get("web_failed") \
                and web_passes < WEB_MAX_PASSES and len(tried) < 3:
            results = state["web_results"]
            candidates = [r for r in results
                          if r.get("url") not in tried and r.get("url") not in fetched_urls
                          and r.get("fetchable", True)]  # Reddit/Instagram leads are never fetched (login walls)
            top = None
            if state.get("resolve_query"):
                # REFERENCE RESOLUTION: pick the best concrete PLACE article — not the concept
                # article ('Capital'), not lookalikes ('Capital punishment in France'), not
                # lists/disambiguation, not flags/maps, and not the BASE entity itself
                # ('France' is not the capital of France — Paris is). The user asked for the
                # NEW place the reference points at (CP 2.1 context).
                prev_words = [w for w in (state.get("prev_dest") or "").lower().split() if len(w) > 3]
                rq = (state.get("resolve_query") or "").strip()
                base_title = rq[len("capital of "):].strip().lower() \
                    if rq.lower().startswith("capital of ") else ""

                _CONCEPTS = {"fact", "facts", "capital", "capital (city)", "capital city",
                            "government", "state", "country", "city", "seat of government",
                            "seats of government", "national capital"}

                def _is_place(r: Dict[str, Any]) -> bool:
                    t = (r.get("title") or "").lower().strip()
                    if t in _CONCEPTS or "disambiguation" in t or "capital punishment" in t \
                            or t.startswith(("list of", "map of", "category:", "flags of")):
                        return False
                    if base_title and t == base_title:
                        return False  # the base has a capital; it is not its own capital
                    if prev_words and all(w in t for w in prev_words):
                        return False
                    return True

                pool = [r for r in candidates if _is_place(r)] or candidates
                top = next((r for r in pool if r.get("provider") == "wikipedia"), pool[0])
            else:
                if candidates:
                    # Relevance guard (CP 3.1 bias guard): only fetch a result that actually
                    # matches the place being researched — otherwise a 'California' search
                    # could pull up 'Tell Me You Love Me (album)' and poison the answer.
                    q_words = {w for w in re.findall(r"[a-z]{4,}",
                                                    (goal.get("destination") or _guess_place(goal.get("raw") or "")).lower())}
                    if q_words:
                        on_target = [r for r in candidates
                                     if any(w in (r.get("title", "") + " " + r.get("url", "")).lower() for w in q_words)]
                        if on_target:
                            candidates = on_target
                    if candidates:
                        # Prefer structured, reliable sources (Wikipedia/Wikivoyage) over
                        # community leads (CP 3.1 reliability). source_type comes from
                        # advanced_search hits; provider from web_search hits.
                        top = next((r for r in candidates
                                    if r.get("source_type") in {"wikipedia", "wikivoyage"}
                                    or r.get("provider") == "wikipedia"), candidates[0])
            if top:
                thought = (
                    f"Web returned {len(results)} result(s). Act: fetch '{top.get('title')}' and index it — "
                    "extract text, chunk into the web crawl store, keep full provenance (url + provider + date)."
                    + (" (reference resolution — the place the user meant)" if state.get("resolve_query") else "")
                    + (" (deeper pass — second page for the summary)" if web_passes else "")
                    + (" (fallback candidate — the previous fetch failed)" if tried else "")
                )
                return thought, "web_fetch", {"url": top["url"], "provider": top.get("provider", "")}

        # 5. done
        thought = (
            f"Evidence is sufficient ({len(state.get('kb_results', []))} search pass(es), "
            f"{len(state.get('hotels') or [])} hotels, {len(state.get('flights') or [])} flights). "
            "I prune weak branches and compose the answer with citations and follow-up questions (CP 1.1)."
        )
        return thought, "compose_answer", {}


class LLMPlanner:
    name = "llm"

    SYSTEM = (
        "You are Wayfinder, a travel research agent implementing a ReAct loop. "
        "You plan step by step: think, act (one tool call), observe, then decide the next step. "
        "Respond with ONLY a JSON object: {\"thought\": string (<= 40 words), "
        "\"action\": one of [search_kb, best_time, search_flights, search_hotels, book_ticket, flight_info, local_places, luxury, travel_safety, web_search, advanced_search, web_fetch, compose_answer], "
        "\"input\": object}. Use compose_answer when you have enough grounded evidence. "
        + "Use best_time when the user asks about the BEST TIME / best month / when to go — it reads the "
        + "key-free seasons dataset (real 2023-2025 climate normals + Wikipedia holidays); pass month (1-12) "
        + "when the user names a month; it NEVER invents months for uncovered places. "
        + (
            "Use flight_info when the user asks for flight pricing or travel time to a destination the "
            "sample schedule doesn't serve — it returns a clearly-labeled distance-based ESTIMATE (never "
            "live pricing). Use web_search/advanced_search/web_fetch only when the KB has little evidence "
            "for the destination (max 2 searches + 2 fetches per run); advanced_search gathers "
            "source-aware leads (Reddit, Instagram, Tripadvisor, Expedia, Wikivoyage, Wikipedia) "
            "that stay unverified until a public page is fetched; web content is unverified (tier 'web'). "
            if WEB_ENABLED else
            "Web tools are disabled in this deployment (WAYFINDER_WEB=0, offline mode) — rely on "
            "search_kb for covered destinations and compose_answer otherwise; never invent facts for "
            "places the knowledge base doesn't cover. "
        )
        + "Never invent facts; rely only on tool observations."
    )
    TOOLS = {
        "search_kb": '{"query": string, "k": int}',
        "best_time": '{"destination": string, "month": int|null} — best time to travel from the key-free seasons dataset (climate normals 2023-2025 + public holidays); month 1-12 when the user names one',
        "search_flights": '{"origin": "MIA|ATL|MCO", "destination": "GCM", "date": "YYYY-MM-DD"}',
        "flight_info": '{"origin": string|null, "destination": string} — flight pricing + travel time for ANY destination (labeled distance-based estimate, not live pricing)',
        "search_hotels": '{"destination": string, "max_price": int|null, "near": string}',
        "book_ticket": '{"flight": object, "confirmed": false}',
        "web_search": '{"query": string, "k": int} — live web (key-free DDG + Wikipedia)',
        "advanced_search": '{"query": string, "k_per_source": int} — structured public leads from Reddit, Instagram, Tripadvisor, Expedia, Wikivoyage, Wikipedia',
        "local_places": '{"place": string, "focus": "dining"|"attractions"|"hotels"|"health"|"luxury"|"authorities"|"safety"|"all", "k": int} — REAL restaurant/cafe/attraction/accommodation names + street addresses near a place (key-free OpenStreetMap; use for dining, stays, things to do, and for safety runs focus="safety" = hospitals/clinics/pharmacies PLUS police/fire/ambulance stations)',
        "luxury": '{"destination": string} — LUXURY experiences: 4-5★ stays + golf/spa/marina/winery near the destination (key-free OpenStreetMap) + Michelin three-star fine dining (Wikipedia official API). Use when the user asks for luxury/upscale/fine-dining experiences, or where the best luxury places are (empty destination = honest global view)',
        "travel_safety": '{"place": string} — official green->yellow->red safety meter from the U.S. State Department travel advisory (levels 1-4); use when the user asks about safety, or for any destination overview',
        "web_fetch": '{"url": string, "provider": string} — fetch + index the best web result',
        "compose_answer": "{}",
    }

    def __init__(self, llm: LLM):
        self.llm = llm
        self._fallback = HeuristicPlanner(llm)

    def next_step(self, goal: Dict[str, Any], state: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        ref_hint = ""
        if state.get("resolve_query"):
            ref_hint = (
                f"REFERENCE RESOLUTION (CP 2.1): the user's phrase '{state.get('resolve_ref')}' is a "
                f"reference to a NEW place. Do NOT research the previous place again — search the web for "
                f"'{state['resolve_query']}', fetch the concrete place article it surfaces, then compose a "
                f"summary of what you found.\n"
            )
        user = (
            f"GOAL: {goal['raw']}\n"
            + ref_hint
            + f"Parsed: destination={goal.get('destination')} origin={goal.get('origin')} date={goal.get('date')} "
            f"budget={goal.get('budget')} nights={goal.get('nights')} focus={goal['focus']}\n"
            f"TOOLS:\n" + "\n".join(f"  {k}: {v}" for k, v in self.TOOLS.items()) + "\n"
            "OBSERVATIONS SO FAR:\n"
            + "\n".join(f"  - {s['observation']}" for s in state.get("steps", [])[-6:])
            + "\n\nWhat is your next step? (JSON only)"
        )
        try:
            data = self.llm.complete_json(self.SYSTEM, user)
            if data and data.get("action") in self.TOOLS:
                thought = str(data.get("thought") or "").strip()[:400] or "(no thought text)"
                action = data["action"]
                action_input = data.get("input") or {}
                if not isinstance(action_input, dict):
                    action_input = {}
                if action == "search_kb" and not action_input.get("query"):
                    action_input["query"] = self._fallback._kb_query(goal)
                if action == "search_hotels" and not action_input.get("destination"):
                    action_input["destination"] = goal.get("destination") or ""
                if action == "web_search" and not action_input.get("query"):
                    action_input["query"] = (goal.get("destination") or _guess_place(goal.get("raw") or ""))[:80] or "travel information"
                if action == "advanced_search" and not action_input.get("query"):
                    action_input["query"] = (goal.get("destination") or _guess_place(goal.get("raw") or ""))[:80] or "travel information"
                if action == "best_time" and not action_input.get("destination"):
                    action_input["destination"] = goal.get("destination") or ""
                if action == "best_time" and action_input.get("month") is None and goal.get("month"):
                    action_input["month"] = int(goal["month"])
                return thought, action, action_input
        except (LLMError, Exception):  # noqa: BLE001 - any LLM failure -> heuristic fallback
            pass
        thought, action, action_input = self._fallback.next_step(goal, state)
        return thought + " [LLM unavailable/invalid -> heuristic fallback]", action, action_input
