"""Key-free OpenStreetMap discovery — Nominatim geocoding + Overpass radius queries.

Wayfinder's "local places" layer (the API answer to "dining options, places to go,
activities"): REAL restaurant/cafe/bar names + street addresses and tourist
attractions (museums, viewpoints, beaches, parks) near a destination, from the
community-maintained OpenStreetMap database.

  - Nominatim  https://nominatim.openstreetmap.org  (geocode a place -> lat/lon)
  - Overpass   https://overpass-api.de               (what's near that point?)

Both are key-free public APIs (the approved exception to the zero-API-key promise).
Usage-policy compliance: one meaningful User-Agent, a hard 1 request/second
throttle shared across the whole process, bounded result sizes, and a 7-day
on-disk cache (data/osm_cache.json) so repeat questions are instant and never
re-hammer OSM.

Honesty model: every failure (no geocode hit, API down, timeout) raises
OSMError; the tool layer converts that to an empty result + a plain note.
This module NEVER invents a place name or address.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

USER_AGENT = "Wayfinder/1.0 (capstone travel-research demo; key-free OSM discovery)"
TIMEOUT = 20            # Nominatim / Wikidata: fast
OVERPASS_TIMEOUT = 35   # Overpass interpreters can queue; bound the wait
CACHE_TTL_DAYS = 7      # OSM changes slowly; a week of cache keeps us polite + fast
MIN_SPACING = 1.1       # seconds between ANY outbound OSM-family request (policy: <=1 req/s)

# Cache-shape version: baked into the near-places cache key. Bump this whenever the
# shaped place-dict gains/changes a field (e.g. adding `stars`/`operator`), so a
# pre-change entry is invalidated on read instead of silently serving the old shape.
SCHEMA = "v2"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",  # 3rd fallback (dense-city 504/429 on the other two)
]
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"


class OSMError(RuntimeError):
    """A key-free lookup failed (offline, no match, API error) — caller degrades honestly."""


# ---------------------------------------------------------------------------
# polite HTTP: shared 1 req/s throttle + User-Agent (Nominatim usage policy)
# ---------------------------------------------------------------------------
_last_out = 0.0
_throttle = threading.Lock()


def _polite_get(url: str, data: Optional[bytes] = None, timeout: int = TIMEOUT) -> bytes:
    global _last_out
    with _throttle:
        wait = _last_out + MIN_SPACING - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_out = time.time()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise OSMError(f"HTTP {e.code} from {url.split('/')[2]}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise OSMError(f"unreachable: {url.split('/')[2]} ({getattr(e, 'reason', type(e).__name__)})") from None


# ---------------------------------------------------------------------------
# disk cache (data/osm_cache.json): key = sha1(kind|place|focus|radius)
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()


def _cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, "osm_cache.json") if data_dir else ""


def _cache_key(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _cache_get(data_dir: str, key: str) -> Optional[dict]:
    path = _cache_path(data_dir)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f).get(key)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(entry, dict):
        return None
    if time.time() - float(entry.get("ts", 0)) > CACHE_TTL_DAYS * 86400:
        return None
    return entry.get("value")


def _cache_put(data_dir: str, key: str, value: dict) -> None:
    path = _cache_path(data_dir)
    if not path:
        return
    try:
        with _cache_lock:
            store: Dict[str, Any] = {}
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        store = json.load(f)
                    if not isinstance(store, dict):
                        store = {}
                except (OSError, json.JSONDecodeError):
                    store = {}
            store[key] = {"ts": time.time(), "value": value}
            # cap growth: keep the newest 60 entries
            if len(store) > 60:
                for k in sorted(store, key=lambda k: float(store[k].get("ts", 0)))[: len(store) - 60]:
                    store.pop(k, None)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(store, f, ensure_ascii=False)
    except OSError:
        pass  # cache is an optimization, never a dependency


# ---------------------------------------------------------------------------
# geocode (Nominatim)
# ---------------------------------------------------------------------------
def geocode(place: str, data_dir: str = "") -> Dict[str, Any]:
    """'George Town, Cayman Islands' -> {lat, lon, display_name, class, type}.

    Raises OSMError when the place cannot be resolved (caller degrades honestly)."""
    place = (place or "").strip()
    if len(place) < 2:
        raise OSMError("no place to geocode")
    key = _cache_key("geo", place)
    hit = _cache_get(data_dir, key)
    if hit:
        return dict(hit)
    q = urllib.parse.urlencode({"q": place, "format": "json", "limit": 1})
    raw = _polite_get(f"{NOMINATIM_URL}?{q}")
    rows = json.loads(raw)
    if not rows:
        raise OSMError(f"Nominatim has no match for {place!r}")
    top = rows[0]
    try:
        result = {
            "lat": float(top["lat"]),
            "lon": float(top["lon"]),
            "display_name": top.get("display_name", place),
            "class": top.get("class", ""),
            "type": top.get("type", ""),
        }
    except (KeyError, ValueError, TypeError) as e:
        raise OSMError(f"malformed geocode response for {place!r}") from e
    _cache_put(data_dir, key, result)
    return result


# ---------------------------------------------------------------------------
# nearby places (Overpass)
# ---------------------------------------------------------------------------
# focus -> Overpass filters (community tags). Each focus is a LIST of filter blocks;
# the query builder joins them as separate statements inside one set-builder
# parenthesis — the canonical valid Overpass QL union pattern. 'all' = dining + attractions.
FOCUS_FILTERS = {
    "dining": ['["amenity"~"^(restaurant|cafe|bar|fast_food|pub)$"]'],
    "attractions": (
        '["tourism"~"^(attraction|viewpoint|museum|zoo|theme_park|gallery|artwork)$"]',
        '["leisure"~"^(park|garden)$"]',
    ),
    "hotels": ['["tourism"~"^(hotel|guest_house|hostel|camp_site)$"]'],
    # Medical / urgent care: pharmacies, hospitals, clinics, emergency centres.
    # Two filter blocks cover both common OSM tag schemes (amenity=* and
    # healthcare=*); a hospital or clinic is tagged one way or the other.
    "health": (
        '["amenity"~"^(pharmacy|hospital|doctors)$"]',
        '["healthcare"~"^(hospital|centre|clinic)$"]',
    ),
    # Upscale stays + premium experiences: hotels/resorts carrying a community
    # star_rating of 4-5, resorts, golf courses, spas, marinas and wineries.
    # (spa/winery appear under both tag schemes in the wild — both are queried.)
    "luxury": (
        '["tourism"~"^(hotel|resort)$"]["star_rating"~"^[45]$"]',
        '["tourism"="resort"]',
        '["leisure"="golf_course"]',
        '["leisure"="spa"]',
        '["amenity"="spa"]',
        '["tourism"="winery"]',
        '["amenity"="winery"]',
        '["amenity"="marina"]',
    ),
    # Local authorities / emergency responders: police stations, fire stations,
    # ambulance stations, coast guard — both tag schemes (amenity=* and
    # emergency=*) appear in the wild.
    "authorities": (
        '["amenity"~"^(police|fire_station|coast_guard)$"]',
        '["emergency"~"^(police|fire_station|ambulance_station)$"]',
    ),
    # The filter is assembled from the concrete categories above; keeping these keys
    # makes `focus="all"` and `focus="safety"` explicit, validated public options.
    "all": [],
    "safety": [],  # public composite: health + authorities (see foci expansion below)
}
DINING_CATEGORIES = {"restaurant", "cafe", "bar", "fast_food", "pub"}
# OSM categories that read as 'luxury' (4-5★ stays + premium experiences), so
# compose can split them from generic dining/sights and label them distinctly.
LUXURY_CATEGORIES = {"hotel", "resort", "golf_course", "spa", "winery", "marina"}
# OSM categories that read as 'health / urgent care' (pharmacy, hospital, clinic…).
# A positive set, like DINING_CATEGORIES, so compose can label them distinctly
# instead of filing a hospital under 'things to see'.
HEALTH_CATEGORIES = {"pharmacy", "hospital", "doctors", "clinic", "centre"}
# OSM categories that read as 'local authorities / emergency responders'
# (police, fire, ambulance, coast guard), so compose labels them distinctly too.
AUTHORITY_CATEGORIES = {"police", "fire_station", "ambulance_station", "coast_guard"}


def _anchor_city(region: str) -> str:
    """'Cayman Islands' -> 'George Town' (world dataset + curated seed; '' if unknown).

    Local import: osm.py stays usable standalone if the world dataset is absent."""
    try:
        from . import world as _world
        return _world.anchor_city(region) or ""
    except Exception:  # noqa: BLE001 — anchoring is an enhancement, never a dependency
        return ""


def _addr(tags: Dict[str, str]) -> str:
    bits = [tags.get("addr:street", ""), tags.get("addr:city", "")]
    return ", ".join(b for b in bits if b)


def _cuisine(tags: Dict[str, str]) -> str:
    c = (tags.get("cuisine") or "").split(";")
    return "; ".join(x.strip() for x in c[:2] if x.strip())


def nearby_places(place: str, focus: str = "dining", radius_m: int = 3000,
                  limit: int = 24, data_dir: str = "") -> Dict[str, Any]:
    """Real named places near `place` from OpenStreetMap (key-free).

    Returns {place, resolved_as, lat, lon, radius_m, places: [...], counts, source}.
    Raises OSMError on any failure — this function never fabricates places.
    """
    if focus not in FOCUS_FILTERS:
        raise OSMError(f"unknown focus {focus!r} (dining|attractions|hotels|health|luxury|authorities|safety|all)")
    radius_m = max(250, min(int(radius_m), 25000))
    # Cap at 200 = the Overpass `out body 200` result limit, i.e. the full set the
    # query can return. (Was 48: an arbitrary display slice that, combined with the
    # by-name sort below, silently dropped brand-specific stays in dense cities —
    # 'Marriott/St. Regis/Westin' sort past the 48th entry and were never seen.)
    limit = max(2, min(int(limit), 200))

    geo = geocode(place, data_dir)
    # Region-level anchors sit in the wrong spot for a radius query: 'Cayman
    # Islands' geocodes to a point in open water, 'New York City' to the whole
    # state. Re-anchor to the region's major city (world dataset: top destination
    # / capital + a small curated seed) — a named city's boundary centroid IS its
    # center, so the radius covers real places instead of the sea or farmland.
    if geo.get("class") == "boundary" or geo.get("type") in ("country", "province", "state", "region"):
        city = _anchor_city(place)
        if city:
            try:
                geo2 = geocode(f"{city}, {place}", data_dir)
                if geo2.get("lat") is not None:
                    geo = geo2
            except OSMError:
                pass  # keep the region anchor; an honest thin result beats a crash
    key = _cache_key("near", SCHEMA, place.lower(), focus, str(radius_m))
    hit = _cache_get(data_dir, key)
    if hit:
        return dict(hit)

    # A broad trip-planning pass should surface stays as well as food and sights.
    # This is particularly useful outside the small local hotel demo dataset: the
    # caller gets real, mapped accommodation names rather than an empty list.
    foci = ["dining", "attractions", "hotels"] if focus == "all" else \
           (["health", "authorities"] if focus == "safety" else
            ["dining", "attractions"] if focus == "dining" else [focus])
    elements: List[dict] = []
    last_err: Optional[str] = None
    for f in foci:
        # NOTE: live-verified quirk — this Overpass build (0.7.62.x) requires a ';'
        # after the LAST statement inside the union parenthesis: '(...);)' not '(...))'.
        body = "".join(
            f"nwr{flt}(around:{radius_m},{geo['lat']},{geo['lon']});"
            for flt in FOCUS_FILTERS[f]
        )
        query = f"[out:json][timeout:25];({body});out body 200;"
        data = urllib.parse.urlencode({"data": query}).encode("utf-8")
        tried = []
        for mirror in OVERPASS_MIRRORS:
            try:
                raw = _polite_get(mirror, data=data, timeout=OVERPASS_TIMEOUT)
                elements.extend(json.loads(raw).get("elements", []))
                last_err = None
                break
            except OSMError as e:
                tried.append(str(e))
                last_err = str(e)
        else:
            raise OSMError(f"Overpass unreachable ({'; '.join(tried)})")

    # dedupe + shape: named places first, then the rest
    seen = set()
    places: List[Dict[str, Any]] = []
    for el in elements:
        tags = el.get("tags", {})
        name = (tags.get("name") or tags.get("operator") or "").strip()
        if not name:
            continue
        cat = (tags.get("amenity") or tags.get("tourism") or tags.get("leisure")
               or tags.get("healthcare") or "place")
        sig = (name.lower(), cat)
        if sig in seen:
            continue
        seen.add(sig)
        places.append({
            "name": name[:80],
            "category": cat,
            "address": _addr(tags)[:100],
            "cuisine": _cuisine(tags),
            "stars": (tags.get("star_rating") or "").strip(),
            "operator": (tags.get("operator") or tags.get("brand") or "").strip()[:60],
            "lat": el.get("lat"),
            "lon": el.get("lon"),
        })
    named = [p for p in places if p["category"] != "place"]
    named.sort(key=lambda p: p["name"].lower())
    if len(places) > limit:
        named = named[:limit]

    counts = {}
    for p in places:
        if p["category"] in DINING_CATEGORIES:
            bucket = "dining"
        elif p["category"] in HEALTH_CATEGORIES:
            bucket = "health"
        elif p["category"] in AUTHORITY_CATEGORIES:
            bucket = "authorities"
        elif p["category"] in LUXURY_CATEGORIES:
            bucket = "luxury"
        else:
            bucket = "other"
        counts[bucket] = counts.get(bucket, 0) + 1

    result = {
        "place": place,
        "resolved_as": geo["display_name"],
        "lat": geo["lat"],
        "lon": geo["lon"],
        "radius_m": radius_m,
        "focus": focus,
        "places": named,
        "counts": counts,
        "source": "OpenStreetMap — community data via key-free Nominatim geocode + Overpass radius query "
                  "(names/addresses as mapped; verify hours & prices before visiting)",
    }
    # Never pin an EMPTY result for the 7-day cache window: a zero-hit can be a
    # transient Overpass state (mirror queue, partial sync) or simply thin mapping
    # at that moment. Re-verifying is cheap — the geocode is cached separately and
    # Overpass stays under the 1 req/s throttle — so a genuinely empty region just
    # costs one polite re-query next time instead of 7 days of a stale 'none'.
    if result["places"]:
        _cache_put(data_dir, key, result)
    return result


# ---------------------------------------------------------------------------
# Wikidata sidecar: one-line official description of the place (key-free)
# ---------------------------------------------------------------------------
def wikidata_fact(place: str, data_dir: str = "") -> str:
    """'Cayman Islands' -> 'island territory of the United Kingdom' (or '').

    Best-effort enrichment: any failure returns '' and is never fatal."""
    place = (place or "").strip()
    if not place:
        return ""
    key = _cache_key("wd", place.lower())
    hit = _cache_get(data_dir, key)
    if hit is not None:
        return str(hit)
    try:
        q = urllib.parse.urlencode({
            "action": "wbsearchentities", "search": place, "language": "en",
            "limit": 1, "type": "item", "format": "json",
        })
        rows = json.loads(_polite_get(f"{WIKIDATA_SEARCH}?{q}"))
        top = (rows.get("search") or [{}])[0]
        desc = (top.get("description") or "").strip()
        if desc:
            _cache_put(data_dir, key, desc)
            return desc[:160]
        _cache_put(data_dir, key, "")
        return ""
    except OSMError:
        return ""
