"""Home-base airport dataset (data/airports.json) — runtime API.

The dataset is built OFFLINE by scripts/fetch_airports.py from Wikipedia's
"List of airports in …" pages (official MediaWiki API, key-free) and committed
to data/airports.json, so the app itself makes ZERO network calls for it.

This module answers "which airport should this person fly out of?":

  * `best_departure(country, region)` — the best departure airport for a
    preselected home country/region (Nevada -> Las Vegas/LAS, Turkey ->
    Istanbul/IST, UK -> London/Heathrow). Ranking is a DISCLOSED heuristic
    (the same pattern as LUXURY_BRANDS): the dataset's `hub_rank` (position
    in the country's ordered curated hub list — index 0 is its primary
    gateway) first, then the `major` flag (FAA primary class / hub list),
    then airports in the capital city (world.json), then name. It never
    invents an airport — every row was parsed from a real page.
  * `iata_for(city)` — offline city -> IATA resolution for web.py's
    resolve_iata_code, consulted before the Wikipedia fallback.
  * `country_list()` / `regions(country)` / `airports_for(...)` — the UI
    dropdown data (GET /api/airports).

Honesty model (CP 1.1): if the dataset is missing or a country/region has no
rows, the functions return empty/None and the caller degrades to the no-home
fallback — never a guessed airport.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "airports.json")
WORLD_PATH = os.path.join(ROOT, "data", "world.json")

_lock = threading.Lock()
_cache: Optional[Dict[str, Any]] = None
_capitals: Optional[Dict[str, str]] = None


def load() -> Dict[str, Any]:
    """Load + cache data/airports.json. Returns {"countries": {}} when absent."""
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            _cache = {"countries": {}}
        return _cache


def _load_capitals() -> Dict[str, str]:
    """world.json `capital_of` (lowercase country -> capital city), cached."""
    global _capitals
    with _lock:
        if _capitals is not None:
            return _capitals
        caps: Dict[str, str] = {}
        try:
            with open(WORLD_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f).get("capital_of") or {}
            if isinstance(raw, dict):
                caps = {str(k).lower(): str(v) for k, v in raw.items()}
        except (OSError, json.JSONDecodeError):
            caps = {}
        _capitals = caps
        return _capitals


def _country_entry(country: str) -> Dict[str, Any]:
    entry = load().get("countries", {}).get(country)
    return entry if isinstance(entry, dict) else {"regions": [], "airports": []}


def country_list() -> List[str]:
    """Country names with at least one airport, alphabetical (for the UI)."""
    out = []
    for name, entry in load().get("countries", {}).items():
        if isinstance(entry, dict) and entry.get("airports"):
            out.append(name)
    return sorted(out, key=str.lower)


def regions(country: str) -> List[str]:
    """Region names for a country (US state, UK county, province, …); [] if the
    country has no regional breakdown (country-level selection still works)."""
    entry = _country_entry(country)
    return sorted({r for r in entry.get("regions", []) if r}, key=str.lower)


def airports_for(country: str, region: Optional[str] = None) -> List[Dict[str, Any]]:
    """Airport rows for a country, optionally narrowed to one region.
    Rows: {name, city, region, iata, icao, major, hub_rank} — city may be
    None; hub_rank is None unless the airport is on the curated hub list."""
    entry = _country_entry(country)
    rows = entry.get("airports", [])
    if region:
        r = str(region).strip()
        rows = [row for row in rows if (row.get("region") or "").strip().lower() == r.lower()]
    return list(rows)


def best_departure(country: str, region: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Best departure airport for a home country (and optional region).

    Ranking (DISCLOSED heuristic, not a fact):
      1. rows with a `hub_rank` (ordered curated hub list — index 0 is the
         country's primary gateway: JFK, LHR, IST, …), lower rank wins
      2. rows with the dataset `major` flag (FAA primary / hub list)
      3. airports in the country's CAPITAL city (world.json capital_of)
      4. airports WITH a city (rankable) before city-less rows
      5. name (stable, alphabetical)
    Returns {name, city, region, iata, icao, major, hub_rank} or None (honest gap).
    """
    rows = [r for r in airports_for(country, region) if r.get("iata")]
    if not rows:
        return None
    capital = _load_capitals().get(str(country).strip().lower(), "")
    cap_norm = " ".join(capital.lower().split())
    cap_tail = cap_norm.split()[-1] if cap_norm else ""   # 'St. John's' -> 'john's'

    def capital_match(row: Dict[str, Any]) -> bool:
        city = " ".join(str(row.get("city") or "").lower().split())
        if not city:
            return False
        name = str(row.get("name") or "").lower()
        return (bool(cap_norm) and (cap_norm in city or city in cap_norm or cap_tail in city)
                or (bool(capital) and capital.lower() in name))

    rows.sort(key=lambda r: (
        r.get("hub_rank") is None,
        r.get("hub_rank") or 0,
        not r.get("major"),
        0 if capital_match(r) else 1,
        0 if r.get("city") else 1,
        str(r.get("name") or "").lower(),
    ))
    return dict(rows[0])


def _norm_city(city: Optional[str]) -> str:
    return " ".join(str(city or "").lower().split())


def iata_for(city: str, country: Optional[str] = None) -> Optional[str]:
    """Offline city -> IATA from the dataset (for web.resolve_iata_code).

    Exact city match first (optionally country-scoped), then a prefix match
    that is unambiguous on the CITY name. Returns None when nothing fits —
    the caller keeps its own fallback. Never guesses: dataset rows only.
    """
    key = _norm_city(city)
    if not key or len(key) < 3:
        return None
    if country:
        rows = airports_for(country)
    else:
        rows = [r for c in load().get("countries", {}).values()
                for r in (c.get("airports") if isinstance(c, dict) else [])]
    exact = [r for r in rows if r.get("iata") and _norm_city(r.get("city")) == key]
    if exact:
        # same city, several airports (e.g. Istanbul: IST vs SAW) -> curated
        # hub first, then major, then name — never a random pick
        exact.sort(key=lambda r: (
            r.get("hub_rank") is None,
            r.get("hub_rank") or 0,
            not r.get("major"),
            str(r.get("name") or "").lower(),
        ))
        return exact[0]["iata"]
    pref = [r for r in rows if r.get("iata") and str(r.get("city") or "").lower().startswith(key)]
    if len({_norm_city(r.get("city")) for r in pref}) == 1 and pref:
        return pref[0]["iata"]
    return None


def describe_home(home: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Normalize a request's `home` ({country, region?}) against the dataset.

    Returns a ready-to-display dict:
      {country, region, city, iata, airport, region_applied: "true"/"false"}
    or None when nothing usable was given (caller behaves as no-home).

    * country is required; a region that doesn't exist for that country is
      dropped (region_applied=false) rather than silently mis-routed.
    * best_departure must resolve, otherwise None (honest gap, CP 1.1).
    """
    if not isinstance(home, dict):
        return None
    country = str(home.get("country") or "").strip()
    if not country:
        return None
    region = str(home.get("region") or "").strip()
    region_applied = False
    if region and region not in regions(country):
        region, region_applied = "", False
    else:
        region_applied = bool(region)
    dep = best_departure(country, region or None)
    if not dep or not dep.get("iata"):
        return None
    return {
        "country": country,
        "region": region,
        "city": dep.get("city") or "",
        "iata": dep["iata"],
        "airport": dep.get("name") or "",
        "region_applied": "true" if region_applied else "false",
    }
