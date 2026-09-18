"""World dataset access — 197-country destination matching + capital resolution.

Loads `data/world.json` (built by scripts/build_world_kb.py from the five
country files in data/countries/). Pure local data: zero API keys, fully
offline. If the file is missing (fresh checkout before the first build), the
curated seed destinations in planner.py keep working exactly as before.

What this module gives the rest of the app:

  DEST_MATCHERS     word-boundary regex -> canonical destination, longest-first
                    (countries + aliases + cities; the "mali in malaysia" bug is
                    gone because matching is on word boundaries)
  KNOWN_CANONICALS  every canonical the KB really covers (dest_covered gate)
  DEST_CONTEXT      destination -> extra region-hint words (capital + top
                    destinations / the city's country) for the ranking bias guard
  resolve_capital   'California' -> 'Sacramento', 'France' -> 'Paris' ...
                    COUNTRY beats US state ('Georgia' -> 'Tbilisi'), so
                    "the capital" follow-ups resolve OFFLINE (no web needed)
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD_PATH = os.path.join(ROOT, "data", "world.json")

# Curated seed destinations (always present, even before the world build).
CURATED_DESTINATIONS: List[Tuple[str, str]] = [
    ("cayman islands", "Cayman Islands"),
    ("grand cayman", "Cayman Islands"),
    ("george town", "George Town"),
    ("cayman", "Cayman Islands"),
    ("kennywood", "Kennywood"),
    ("west mifflin", "Kennywood"),
    ("pittsburgh", "Pittsburgh"),
    ("miami", "Miami"),
    ("atlanta", "Atlanta"),
    ("orlando", "Orlando"),
]

# Aliases that are too generic to match as bare words (would eat pronouns /
# region names): 'america' (South America), 'us' (the pronoun), 'car', 'png'.
_DANGEROUS_ALIAS_NEEDLES = {"america", "us", "car", "png"}

# US state capitals (50 states + DC). Country beats state: 'Georgia' resolves
# to Tbilisi first; only if the base is not a known country do we look here —
# so 'California' -> Sacramento and 'Georgia' (state) -> Atlanta both work
# when the user names them in a capital question.
US_STATE_CAPITALS: Dict[str, str] = {
    "alabama": "Montgomery", "alaska": "Juneau", "arizona": "Phoenix",
    "arkansas": "Little Rock", "california": "Sacramento", "colorado": "Denver",
    "connecticut": "Hartford", "delaware": "Dover",
    "district of columbia": "Washington, D.C.", "florida": "Tallahassee",
    "georgia": "Atlanta", "hawaii": "Honolulu", "idaho": "Boise",
    "illinois": "Springfield", "indiana": "Indianapolis", "iowa": "Des Moines",
    "kansas": "Topeka", "kentucky": "Frankfort", "louisiana": "Baton Rouge",
    "maine": "Augusta", "maryland": "Annapolis", "massachusetts": "Boston",
    "michigan": "Lansing", "minnesota": "Saint Paul", "mississippi": "Jackson",
    "missouri": "Jefferson City", "montana": "Helena", "nebraska": "Lincoln",
    "nevada": "Carson City", "new hampshire": "Concord", "new jersey": "Trenton",
    "new mexico": "Santa Fe", "new york": "Albany", "new york city": "Albany",
    "north carolina": "Raleigh", "north dakota": "Bismarck", "ohio": "Columbus",
    "oklahoma": "Oklahoma City", "oregon": "Salem", "pennsylvania": "Harrisburg",
    "rhode island": "Providence", "south carolina": "Columbia",
    "south dakota": "Pierre", "tennessee": "Nashville", "texas": "Austin",
    "utah": "Salt Lake City", "vermont": "Montpelier", "virginia": "Richmond",
    "washington": "Olympia", "west virginia": "Charleston", "wisconsin": "Madison",
    "wyoming": "Cheyenne",
    # the country itself, for 'capital of the United States'
    "united states": "Washington, D.C.", "usa": "Washington, D.C.",
}


def ascii_norm(s: str) -> str:
    """NFKD accent-strip: 'Türkiye' -> 'Turkey', 'São Paulo' -> 'Sao Paulo'.
    Matching everywhere is on this normalized form, so accented and unaccented
    spellings both work (ranking.py tokenizes [a-z0-9]+ — accents would break
    the region-hint match otherwise)."""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _clean_city(name: str) -> str:
    """'St. John's (Antigua)' -> 'St. John's'; 'Washington, D.C.' -> 'Washington';
    'Patagonia (El Calafate, Ushuaia)' -> 'Patagonia'."""
    s = re.sub(r"\s*\([^)]*\)", " ", name)          # strip parentheticals
    s = s.split(",")[0]                               # 'Washington, D.C.' -> Washington
    s = re.sub(r"\s+", " ", s).strip(" .-'")
    return s


# ---------------------------------------------------------------------------
# lazy load
# ---------------------------------------------------------------------------
_WORLD: Optional[dict] = None
_LOADED = False


def _load_world() -> dict:
    global _WORLD, _LOADED
    if _LOADED:
        return _WORLD or {}
    _LOADED = True
    if os.path.exists(WORLD_PATH):
        try:
            with open(WORLD_PATH, "r", encoding="utf-8") as f:
                _WORLD = json.load(f)
        except Exception:  # noqa: BLE001 — corrupt file -> curated seed only
            _WORLD = None
    return _WORLD or {}


# ---------------------------------------------------------------------------
# destination matching
# ---------------------------------------------------------------------------
def _valid_needle(n: str) -> bool:
    n = n.strip().lower()
    if not (2 <= len(n) <= 40):
        return False
    if not re.search(r"[a-z]{2}", n):
        return False
    if n in _DANGEROUS_ALIAS_NEEDLES:
        return False
    return True


def _city_entry(world: dict, city: str):
    e = (world.get("cities") or {}).get(city)
    if isinstance(e, dict):
        return e.get("name") or city.title(), e.get("country", "")
    if isinstance(e, str):
        return city.title(), e
    return city.title(), ""


def city_display(world: dict, city: str) -> str:
    name, _country = _city_entry(world, city)
    return name


def build_matchers(world: Optional[dict] = None) -> List[Tuple[re.Pattern, str]]:
    """(word-boundary regex, canonical) pairs, longest needle first.

    Precedence for an identical needle string: curated seed > city > country.
    Word boundaries kill the old substring bug ('mali' matched 'Malaysia').
    Needles are ASCII-normalized, so accented input ('São Paulo') matches too —
    callers should normalize the query text the same way.
    """
    world = world if world is not None else _load_world()
    best: Dict[str, Tuple[str, int]] = {}   # needle -> (canonical, rank)

    def put(needle: str, canonical: str, rank: int) -> None:
        n = ascii_norm(needle).strip().lower()
        if not _valid_needle(n) or not canonical:
            return
        cur = best.get(n)
        if cur is None or rank < cur[1]:
            best[n] = (canonical, rank)

    for c in world.get("countries", []):
        canonical = c.get("canonical") or c.get("name", "")
        put(c.get("name", ""), canonical, 2)
        for a in c.get("aliases", []):
            put(a, canonical, 2)
    for city in (world.get("cities") or {}).keys():
        # city canonicals are the CITIES themselves (better hotel/KB targeting);
        # the country is carried in DEST_CONTEXT below.
        put(city, city_display(world, city), 1)
    for needle, canonical in CURATED_DESTINATIONS:
        put(needle, canonical, 0)

    pairs = sorted(best.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    out: List[Tuple[re.Pattern, str]] = []
    for needle, (canonical, _rank) in pairs:
        out.append((re.compile(r"\b" + re.escape(needle) + r"\b", re.I), canonical))
    return out


_DEST_MATCHERS: Optional[List[Tuple[re.Pattern, str]]] = None


def dest_matchers() -> List[Tuple[re.Pattern, str]]:
    global _DEST_MATCHERS
    if _DEST_MATCHERS is None:
        _DEST_MATCHERS = build_matchers()
    return _DEST_MATCHERS


def known_canonicals() -> set:
    """Every canonical destination the KB covers (planner's dest_covered gate)."""
    world = _load_world()
    out = {c for _n, c in CURATED_DESTINATIONS}
    for c in world.get("countries", []):
        if c.get("canonical"):
            out.add(c["canonical"])
    for city in (world.get("cities") or {}).keys():
        out.add(city_display(world, city))
    return out


def dest_context() -> Dict[str, str]:
    """destination -> extra region-hint words for the ranking bias guard.

    country -> 'capital top-3 destinations'; city -> its country. The four
    curated demo entries keep their hand-tuned context (last, so they win).
    """
    world = _load_world()
    ctx: Dict[str, str] = {}
    for c in world.get("countries", []):
        canonical = c.get("canonical") or ""
        if not canonical:
            continue
        bits = [b for b in [c.get("capital", ""), *c.get("top_destinations", [])] if b]
        ctx[canonical] = " ".join(ascii_norm(b).lower() for b in bits[:4])
    for city, entry in (world.get("cities") or {}).items():
        name, country = _city_entry(world, city)
        if name not in ctx:
            ctx[name] = ascii_norm(country).lower()
    ctx.update({
        "Cayman Islands": "George Town Grand Cayman",
        "George Town": "Grand Cayman",
        "Kennywood": "Pittsburgh West Mifflin rides discounts",
        "Pittsburgh": "Strip District North Hills",
    })
    return ctx


# ---------------------------------------------------------------------------
# provenance / home-region (photo & content anchoring, CP 3.1 bias guard)
# ---------------------------------------------------------------------------
# The photo layer needs to tell a real "Rio de Janeiro" food photo from a
# "Brazilian food, Quebec city" one: the second names a foreign CITY (Quebec
# City, Canada) whose country is not the destination's home country (Brazil), so
# it is a foreign-place leak and is dropped. We key off dataset CITIES (not
# country names or top-destination landmarks) because a city name is the reliable
# 'this photo is OF/IN a foreign city' signal, while a shared-border landmark
# (e.g. 'Iguazu Falls', which this dataset files under Argentina) should not tank
# a genuine Brazil-side photo ('...Foz do Iguacu, Brazil'). Pure local data
# (data/world.json), fully offline, no new keys.
_CITY_COUNTRY: Optional[Dict[str, str]] = None


def known_city_countries() -> Dict[str, str]:
    """ASCII-normalized CITY name -> ASCII-normalized country, for every city in
    the dataset (the `cities` map, by key and by display name).

    This is the 'named place -> country' lookup the photo filter uses to detect
    foreign-city leaks. The destination's own cities all map to its home country,
    so a photo that names only local cities is never mistaken for a foreign leak."""
    global _CITY_COUNTRY
    if _CITY_COUNTRY is not None:
        return _CITY_COUNTRY
    m: Dict[str, str] = {}
    for city, entry in (_load_world().get("cities") or {}).items():
        if not isinstance(entry, dict):
            continue
        country = ascii_norm(entry.get("country", "")).strip().lower()
        if not country:
            continue
        for key in [city, entry.get("name", "")]:
            n = ascii_norm(key).strip().lower()
            if n and n not in m:
                m[n] = country
    _CITY_COUNTRY = m
    return m


def home_countries(dest: str) -> set:
    """The destination's home country (normalized), or an EMPTY set when we can't
    determine it.

    A dataset country -> itself; a dataset city -> its country; a curated/unknown
    destination (Kennywood, Cayman Islands, George Town) -> empty. Callers MUST
    treat an empty result as 'provenance undecidable' and NOT drop anything —
    that is what keeps the curated demo destinations (Kennywood/Cayman) working
    exactly as before, since their photos legitimately name US cities or a
    territory that isn't in the 197-country set."""
    d = ascii_norm(dest or "").strip().lower()
    if not d:
        return set()
    world = _load_world()
    for c in world.get("countries", []):
        names = {ascii_norm(c.get("canonical") or "").lower(),
                 ascii_norm(c.get("name") or "").lower(),
                 *(ascii_norm(a).lower() for a in (c.get("aliases") or []))}
        if d in names:
            country = ascii_norm(c.get("canonical") or c.get("name") or "").strip().lower()
            return {country} if country else set()
    for city, entry in (world.get("cities") or {}).items():
        if not isinstance(entry, dict):
            continue
        if ascii_norm(city).strip().lower() == d or \
                ascii_norm(entry.get("name", "")).strip().lower() == d:
            country = ascii_norm(entry.get("country", "")).strip().lower()
            return {country} if country else set()
    return set()


def _norm_place(s: str) -> str:
    """Canonical place-token normalization used for BOTH the home country and the
    region signatures, so 'congo (republic)' and 'congo republic' compare equal
    (accents stripped, punctuation dropped, whitespace collapsed, lowercased)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", ascii_norm(s or "").lower())).strip()


def home_country(dest: str) -> str:
    """The destination's home country (normalized), or '' when undecidable
    (a curated/unknown destination such as Kennywood or the Cayman Islands).
    A '' result is the 'provenance undecidable' signal: callers keep their
    soft bias behavior so the curated demo destinations never starve."""
    hs = home_countries(dest or "")
    return _norm_place(next(iter(hs), ""))


def country_facts(dest: str) -> dict:
    """Grounded quick facts for a destination from data/world.json (CP 3.1):
    name, capital, continent, top destinations. Returns {} when the destination is
    not a known dataset country (curated demos / live-web-only places) — callers
    simply omit the fact rather than invent one. Matching is on the accent-stripped,
    punctuation-dropped canonical, so 'Congo (Republic)' and 'congo republic' both hit."""
    d = _norm_place(dest or "")
    if not d:
        return {}
    world = _load_world()
    for c in world.get("countries", []):
        names = {_norm_place(c.get("canonical") or ""),
                 _norm_place(c.get("name") or ""),
                 *(_norm_place(a) for a in (c.get("aliases") or []))}
        if d in names:
            top = [t for t in (c.get("top_destinations") or []) if t]
            return {
                "name": (c.get("name") or c.get("canonical") or "").strip(),
                "capital": (c.get("capital") or "").strip(),
                "continent": (c.get("continent") or "").strip().title(),
                "top_destinations": top[:6],
            }
    return {}


# Every curated demo destination (needle AND canonical): the ranking bias guard
# must treat these as SOFT (provenance undecidable), never hard-prune them.
_DEMO_DESTS: set = set()
for _n, _c in CURATED_DESTINATIONS:
    _DEMO_DESTS.add(ascii_norm(_n).strip().lower())
    _DEMO_DESTS.add(ascii_norm(_c).strip().lower())


def effective_home(dest: str) -> str:
    """home_country(dest), with a curated-demo exception: a curated demo
    destination (Kennywood, Cayman Islands, George Town, Pittsburgh, Miami, ...)
    returns '' so the ranking guard stays SOFT (boost/penalty, never hard-prune).

    Why: home_country() is semantically correct but over-eager for the demos —
    Pittsburgh is a real US city (-> 'united states') and George Town is also a
    city in Malaysia (-> 'malaysia'). Hard-pruning on those would drop the demos'
    own legitimately-bundled content (Kennywood sits in West Mifflin next to
    Pittsburgh; George Town IS the Cayman capital; a Miami answer day-trips to
    Orlando). '' is the 'provenance undecidable' signal that keeps the curated
    demos behaving exactly as before."""
    d = ascii_norm(dest or "").strip().lower()
    if d in _DEMO_DESTS:
        return ""
    return home_country(dest)


# Place-signature table for the ranking bias guard: a normalized space-joined
# phrase -> the normalized 'home' (country canonical, or a curated-demo tag) it
# belongs to. Built once from world.json (countries: name/canonical/alias; cities:
# name->country) plus the curated demos. Longest signature wins in region_home(),
# which is what disambiguates near-name pairs (Congo (Republic) vs DR Congo).
_PLACE_SIGS: Optional[List[Tuple[str, str]]] = None


def _place_signatures() -> List[Tuple[str, str]]:
    global _PLACE_SIGS
    if _PLACE_SIGS is not None:
        return _PLACE_SIGS
    out: List[Tuple[str, str]] = []
    seen = set()

    def add(phrase: str, home: str) -> None:
        p = _norm_place(phrase)
        if not p or len(p) < 3 or p in _DANGEROUS_ALIAS_NEEDLES:
            return
        key = (p, home)
        if key in seen:
            return
        seen.add(key)
        out.append((p, home))

    w = _load_world()
    for c in w.get("countries", []):
        canon = _norm_place(c.get("canonical") or c.get("name") or "")
        if not canon:
            continue
        for ph in [c.get("name", ""), c.get("canonical", ""), *(c.get("aliases") or [])]:
            add(ph, canon)
    for city, entry in (w.get("cities") or {}).items():
        country = _norm_place((entry or {}).get("country", ""))
        if not country:
            continue
        add(city, country)
        add((entry or {}).get("name", ""), country)
    # curated demo destinations (territories/parks not in the 197-country set)
    for demo, tag in [("cayman islands", "cayman"), ("grand cayman", "cayman"),
                     ("george town", "cayman"), ("cayman", "cayman"),
                     ("kennywood", "kennywood"), ("west mifflin", "kennywood"),
                     ("pittsburgh", "kennywood"),
                     ("miami", "united states"), ("atlanta", "united states"), ("orlando", "united states")]:
        add(demo, tag)
    # longest phrases first so region_home() can skip shorter ones safely
    out.sort(key=lambda t: len(t[0]), reverse=True)
    _PLACE_SIGS = out
    return out


def region_home(region_text: str) -> str:
    """The primary known country / curated-demo a KB `region` string belongs to,
    by LONGEST place-signature match; '' when it names no known place.

    Used by the ranking bias guard (CP 3.1) to hard-prune off-destination
    chunks: a 'New Zealand' region is 'new zealand', a 'cayman islands grand
    cayman george town' region is 'cayman', and the near-name pair is split
    correctly because the longer signature (e.g. 'democratic republic of the
    congo') outranks the shared substring ('republic of the congo')."""
    if not region_text:
        return ""
    rt = _norm_place(region_text)
    if not rt:
        return ""
    padded = f" {rt} "
    for phrase, home in _place_signatures():
        if f" {phrase} " in padded:
            return home
    return ""


_CITIES_CANON: Optional[set] = None
_COUNTRIES_CANON: Optional[set] = None


def city_canonicals() -> set:
    """Canonical destination names that are dataset CITIES (not countries)."""
    global _CITIES_CANON
    if _CITIES_CANON is None:
        world = _load_world()
        _CITIES_CANON = {city_display(world, c) for c in (world.get("cities") or {})}
    return _CITIES_CANON


def country_canonicals() -> set:
    """Canonical destination names that are dataset COUNTRIES."""
    global _COUNTRIES_CANON
    if _COUNTRIES_CANON is None:
        _COUNTRIES_CANON = {
            c["canonical"] for c in _load_world().get("countries", []) if c.get("canonical")
        }
    return _COUNTRIES_CANON


def _matcher_hits(segment: str, allowed: set) -> List[Tuple[int, str]]:
    """(start, canonical) for every matcher whose canonical is in `allowed` and
    matches within `segment` (indices are in `segment` coordinates)."""
    out: List[Tuple[int, str]] = []
    for pat, canonical in dest_matchers():
        if canonical not in allowed:
            continue
        for m in pat.finditer(segment):
            out.append((m.start(), canonical))
    return out


def match_destination(text: str, origin_span: Optional[Tuple[int, int]] = None
                      ) -> str:
    """Best canonical destination in `text` (word-boundary, longest-first),
    skipping any match that falls inside the 'from X' origin span.

    An explicit 'city, country' phrase anchors on the CITY — the country word
    is only disambiguation context ('Rome, Italy' -> Rome, not Italy; without
    this the longer country needle would win the longest-first order).

    The query is accent-normalized first ('São Paulo' -> 'Sao Paulo') so ASCII
    needles still match; the origin span is honored when given (planner.py
    computes it on the same normalized text)."""
    low = ascii_norm(text).lower()
    if "," in low:
        for cm in re.finditer(r",", low):
            left = low[:cm.start()]
            right = low[cm.end():]
            left_hits = [
                h for h in _matcher_hits(left, city_canonicals())
                if not (origin_span and origin_span[0] <= h[0] < origin_span[1])
            ]
            if left_hits and _matcher_hits(right, country_canonicals()):
                # the city closest to the comma is the one paired with the country
                return max(left_hits, key=lambda h: h[0])[1]
    for pat, canonical in dest_matchers():
        m = pat.search(low)
        if not m:
            continue
        if origin_span and origin_span[0] <= m.start() < origin_span[1]:
            continue
        return canonical
    return ""


# ---------------------------------------------------------------------------
# capital resolution (offline)
# ---------------------------------------------------------------------------
_COUNTRY_CAPITALS: Optional[Dict[str, str]] = None


def _country_capitals() -> Dict[str, str]:
    global _COUNTRY_CAPITALS
    if _COUNTRY_CAPITALS is None:
        m: Dict[str, str] = {}
        for c in _load_world().get("countries", []):
            cap = (c.get("capital") or "").strip()
            if not cap:
                continue
            keys = [c.get("name", "")] + list(c.get("aliases", []))
            for k in keys:
                n = ascii_norm(k).strip().lower()
                if n and n not in m:
                    m[n] = cap
        _COUNTRY_CAPITALS = m
    return _COUNTRY_CAPITALS


def resolve_capital(base: str) -> Tuple[Optional[str], str]:
    """Resolve 'the capital of {base}' OFFLINE.

    Returns (capital, kind) where kind is 'country' | 'us_state' | '' (no hit).
    Country beats US state (Georgia -> Tbilisi); 'new york city' -> Albany;
    'united states' -> Washington, D.C.
    """
    b = ascii_norm(base).strip().lower()
    b = b.rstrip(".")
    if not b:
        return None, ""
    cap = _country_capitals().get(b)
    if cap:
        return cap, "country"
    if b in US_STATE_CAPITALS:
        return US_STATE_CAPITALS[b], "us_state"
    # 'state of X' / 'X state' phrasing
    m = re.fullmatch(r"state of (.+)", b) or re.fullmatch(r"(.+) state", b)
    if m:
        inner = m.group(1).strip().lower()
        if inner in US_STATE_CAPITALS:
            return US_STATE_CAPITALS[inner], "us_state"
    return None, ""


# Curated anchors beyond the 197-country dataset, plus names that Nominatim resolves
# to a BIGGER region than the user means. 'Local places' near a region must anchor on
# the city people actually visit — a region centroid can sit in open water ('Cayman
# Islands' geocodes to the Caribbean Sea; 'New York City' geocodes to the whole state).
ANCHOR_CITIES: Dict[str, str] = {
    "cayman islands": "George Town",        # in the 197-country dataset; anchor keeps 'local places' on the city, not the island centroid
    "new york city": "Manhattan",           # the city people visit, not the state centroid
    "new york": "Manhattan",                # famous city — NOT the state (Albany would be a wrong anchor)
    "washington": "Washington, D.C.",       # famous city — NOT the state (Olympia would be a wrong anchor)
}


def anchor_city(region: str) -> str:
    """Region name -> the major city that anchors 'local places' near it.

    Order: curated seed (territories + tricky names) -> 197-country dataset
    (top destination first — the city people actually visit — then capital) ->
    US state capital. Returns '' when the dataset has no city for the region;
    callers then keep the region centroid (honest: fewer places, never invented).
    """
    b = ascii_norm(region or "").strip().lower().rstrip(".")
    if not b:
        return ""
    if b in ANCHOR_CITIES:
        return ANCHOR_CITIES[b]
    for c in _load_world().get("countries", []):
        names = {ascii_norm(c.get("canonical", "") or "").lower(),
                 *(ascii_norm(a).lower() for a in (c.get("aliases") or []))}
        if b in names:
            top = c.get("top_destinations") or []
            return (top[0] if top else (c.get("capital") or "")).strip()
    return US_STATE_CAPITALS.get(b, "")


# ---------------------------------------------------------------------------
# sanity check (used by tests / the build script)
# ---------------------------------------------------------------------------
def stats() -> dict:
    world = _load_world()
    countries = world.get("countries", [])
    caps = sum(1 for c in countries if (c.get("capital") or "").strip())
    return {
        "file_present": bool(world),
        "countries": len(countries),
        "with_capital": caps,
        "cities": len(world.get("cities", {})) or len(world.get("city_of", {})),
        "matchers": len(dest_matchers()),
    }
