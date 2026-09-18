"""Rubric ranking + beam pruning (CP 4.1: "The candidate nodes are scored based
on a set of rules: Date of information, Source reliability (PRIMARY->SECONDARY->
TERTIARY), and relevance to the user prompt." + beam search limiting the branch
width; "If a source is outdated, it gets pruned. If the information is
irrelevant to the goal, it is pruned.")

score = 0.55 * relevance + 0.25 * reliability + 0.10 * recency + 0.10 * phrase
"""

from __future__ import annotations

import re
from datetime import date
from typing import List

from .vectorstore import Chunk, ScoredChunk, TIER_WEIGHT
from .world import region_home  # hard bias guard: chunk.region -> known home place

# rubric weights
W_RELEVANCE = 0.55
W_RELIABILITY = 0.25
W_RECENCY = 0.10
W_PHRASE = 0.10

# beam parameters (CP 4.1: branch limit 3 candidates, depth 3-4)
BEAM_WIDTH = 3
MAX_DEPTH = 3
PRUNE_FLOOR = 0.20          # absolute score floor
PRUNE_RELATIVE = 0.45       # prune candidates scoring < 45% of the best in a pass
STALE_DAYS = 730            # information older than ~2 years is "outdated"

# destination anchoring (bias guard, CP 3.1): with a destination in play, chunks
# inside the requested region rise and everything else sinks — so a thin
# Tuvalu answer stays Tuvalu instead of leaking Cayman/Kennywood prose
REGION_MATCH_BOOST = 1.30
REGION_MISMATCH_PENALTY = 0.30


def recency_score(date_str: str, today: date) -> float:
    """1.0 for fresh info, decaying to 0 at STALE_DAYS. Undated -> neutral 0.5."""
    if not date_str:
        return 0.5
    try:
        published = date.fromisoformat(date_str[:10])
    except ValueError:
        return 0.5
    age = (today - published).days
    if age < 0:
        age = 0
    return max(0.0, 1.0 - age / STALE_DAYS)


def phrase_boost(query: str, chunk: Chunk) -> float:
    """Exact multi-word phrase matches in the chunk (the 'hard' part of the
    search: vector similarity + literal evidence). Word-boundary matched so
    'eat' doesn't count inside 'meat'."""
    q = query.lower()
    words = [w for w in re.findall(r"[a-z0-9]{3,}", q) if w not in _STOP]
    if not words:
        return 0.0
    text = f" {chunk.text.lower()} "
    hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", text))
    return min(1.0, hits / max(3, len(words)))


_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "you", "your", "what", "which", "where", "when", "how", "plan", "plans",
    "trip", "trips", "trip", "me", "my", "a", "an", "to", "of", "in", "on",
    "want", "wants", "looking", "best", "good", "please", "find", "show",
    "tell", "give", "info", "information", "details",
}


def _region_match(chunk: Chunk, hint_words) -> bool:
    """True if the chunk is inside the hinted region.

    Primary signal is the chunk's own `region` metadata (document-level
    front matter, CP 3.1 provenance). Fallback for untagged chunks: exact
    word match in title/text using only strong hint words (>=5 chars) so
    e.g. hint 'west' can't match 'Westin' or 'West End Shopping Village'."""
    if chunk.region:
        region_words = {w for w in re.findall(r"[a-z0-9]+", chunk.region.lower()) if len(w) > 2}
        if region_words:
            return any(w in region_words for w in hint_words)
    strong = {w for w in hint_words if len(w) >= 5}
    if not strong:
        return True  # no discriminative signal -> don't penalize
    hay_words = set(re.findall(r"[a-z0-9]+", f"{chunk.title} {chunk.text}".lower()))
    return any(w in hay_words for w in strong)


def rank(scored: List[ScoredChunk], today: date, region_hint: str = "", query: str = "",
         home_country: str = "") -> List[ScoredChunk]:
    """Apply the rubric, then beam-prune. Returns survivors first (best
    score), then pruned candidates with prune_reason (for the trace UI).

    region_hint (destination words): chunks outside the requested region are
    softly down-weighted so Cayman dining facts never leak into a Kennywood
    answer (bias guard, CP 3.1: 'bias and inaccurate data sources can poison
    the final results').

    home_country (normalized destination home, '' when undecidable / curated
    demo): HARD bias guard — a chunk whose `region` names a DIFFERENT known
    place is dropped outright instead of merely penalized, so off-destination
    content (New Zealand / Bahamas for a Chile search) never reaches the beam
    (CP 1.1: omit rather than misattribute). '' keeps the soft path above.

    query: used for the exact-phrase component of the rubric (the 'hard'
    part of the search: vector similarity + literal evidence)."""
    hint_words = {w for w in region_hint.lower().split() if len(w) > 3}
    for s in scored:
        s.reliability = TIER_WEIGHT.get(s.chunk.tier, 0.5)
        s.recency = recency_score(s.chunk.date, today)
        s.phrase = phrase_boost(query, s.chunk)
        s.score = (
            W_RELEVANCE * s.relevance
            + W_RELIABILITY * s.reliability
            + W_RECENCY * s.recency
            + W_PHRASE * s.phrase
        )
        # HARD bias guard (CP 3.1): destination has a determinable home AND this
        # chunk's region names a different known place -> drop it outright.
        # region_home('') (untagged/undecidable region) or home_country==''
        # (curated demos) fall through to the soft boost/penalty below.
        if home_country:
            rh = region_home(s.chunk.region)
            if rh and rh != home_country:
                s.pruned = True
                s.prune_reason = (f"off-destination: source is about '{rh}', "
                                 f"not '{home_country}' (region hard-prune)")
                continue
        if hint_words:
            if _region_match(s.chunk, hint_words):
                s.score *= REGION_MATCH_BOOST
            else:
                s.score *= REGION_MISMATCH_PENALTY
    scored.sort(key=lambda s: s.score, reverse=True)
    if not scored:
        return []
    unpicked = [s for s in scored if not (s.pruned and s.prune_reason)]
    if not unpicked:
        # every candidate is off-destination (hard-pruned): an honest empty
        # beam, never a foreign source (CP 1.1 omit over misattribute).
        return list(scored)
    best = unpicked[0].score
    survivors: List[ScoredChunk] = []
    for s in unpicked:  # hard-pruned off-destination chunks are excluded here
        if len(survivors) >= BEAM_WIDTH:
            s.pruned = True
            s.prune_reason = "kept the 3 best candidates for this pass"
            continue
        if s.score < PRUNE_FLOOR:
            s.pruned = True
            s.prune_reason = "score too low for this search"
            continue
        if best > 0 and s.score < best * PRUNE_RELATIVE:
            s.pruned = True
            s.prune_reason = "weaker than the top results"
            continue
        if s.chunk.tier == "tertiary" and survivors and s.score < survivors[0].score - 0.08:
            s.pruned = True
            s.prune_reason = "less reliable source"
            continue
        survivors.append(s)
    if not survivors:  # never starve a destination that has on-destination sources
        survivors = [unpicked[0]]
    survivors.sort(key=lambda s: s.score, reverse=True)
    return survivors + [s for s in scored if s.pruned]
