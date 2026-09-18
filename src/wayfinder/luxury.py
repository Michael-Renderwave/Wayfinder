"""Key-free luxury experiences — real 4/5-star stays & premium experiences
(OpenStreetMap) + Michelin three-star fine dining (Wikipedia, official API).

This is the API answer to "where are the best luxury experiences?":

  * STAYS & EXPERIENCES — OpenStreetMap (the same key-free Nominatim + Overpass
    pair as osm.py), cut to upscale tags: hotels/resorts with a community
    `star_rating` of 4 or 5, plus golf courses, spas, marinas and wineries near
    the destination. Real mapped names + street addresses; never invented.
  * FINE DINING — the Wikipedia "List of Michelin 3-star restaurants" article,
    read per-country through the OFFICIAL MediaWiki API (key-free, no scraping
    of a protected site): (restaurant, chef, city, awarded-since) rows.
  * GLOBAL VIEW — the same article, section by section, gives an honest
    "where the world's three-star dining concentrates" ranking for
    destination-less questions ("where are the best places for a luxury trip").

Honesty model (CP 1.1): every failure degrades to an empty bucket + a plain
note; this module NEVER invents a restaurant, hotel, star or ranking. Results
are cached 30 days in data/luxury_cache.json — the Michelin list updates about
once a year and rows carry their "since <year>", so a month of cache is both
polite to the API and current enough.

Zero new dependencies: stdlib urllib + re + json only.
"""

from __future__ import annotations

import hashlib
import html as _html
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

USER_AGENT = "Wayfinder/1.0 (capstone travel-research demo; key-free Wikipedia/OSM reads)"
TIMEOUT = 20
CACHE_TTL_DAYS = 30          # Michelin list changes ~yearly; a month keeps us polite + current
WIKI_API = "https://en.wikipedia.org/w/api.php"
MICHINEL_PAGE = "List of Michelin 3-star restaurants"
MICHINEL_URL = "https://en.wikipedia.org/wiki/List_of_Michelin_3-star_restaurants"


class LuxuryError(RuntimeError):
    """A key-free luxury lookup failed (offline, no match) — caller degrades honestly."""


# ---------------------------------------------------------------------------
# cache (data/luxury_cache.json) — same shape as osm.py's, independent file
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()


def _cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, "luxury_cache.json") if data_dir else ""


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
            if len(store) > 40:
                for k in sorted(store, key=lambda k: float(store[k].get("ts", 0)))[: len(store) - 40]:
                    store.pop(k, None)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(store, f, ensure_ascii=False)
    except OSError:
        pass  # cache is an optimization, never a dependency


# ---------------------------------------------------------------------------
# Wikipedia API (official, key-free)
# ---------------------------------------------------------------------------
def _api_json(params: Dict[str, Any]) -> dict:
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise LuxuryError(f"HTTP {e.code} from en.wikipedia.org") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LuxuryError(f"en.wikipedia.org unreachable ({getattr(e, 'reason', type(e).__name__)})") from None
    if isinstance(raw.get("error"), dict):
        raise LuxuryError(f"Wikipedia API error: {raw['error'].get('info', raw['error'].get('code', '?'))}")
    return raw


def _strip_tags(fragment: str) -> str:
    s = re.sub(r"<[^>]+>", " ", fragment or "")
    s = _html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _target_country(destination: str) -> str:
    """Destination -> the dataset country it belongs to (display name, e.g. 'Italy').

    A dataset country -> itself; a dataset city -> its country; anything else
    -> '' (the caller then reports the country is not listed — honest, no guess)."""
    try:
        from . import world as _world
    except Exception:  # noqa: BLE001 — world dataset is optional
        return ""
    # 'Rome, Italy' -> try the city part first, then the whole phrase
    candidates = [p.strip() for p in re.split(r",|\band\b", destination or "") if p.strip()]
    if not candidates:
        return ""
    world = _world._load_world()
    for d in candidates:
        d = _world.ascii_norm(d).strip().lower()
        if not d:
            continue
        for c in world.get("countries", []):
            names = {_norm(c.get("canonical")), _norm(c.get("name"))}
            names.update(_norm(a) for a in (c.get("aliases") or []))
            if d in names and c.get("name"):
                return c["name"]
        for city, entry in (world.get("cities") or {}).items():
            if not isinstance(entry, dict):
                continue
            if d == _norm(city) or d == _norm(entry.get("name", "")):
                return (entry.get("country") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Michelin three-star (per country)
# ---------------------------------------------------------------------------
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_ROWHEAD_RE = re.compile(r'<th[^>]*scope="row"[^>]*>(.*?)</th>', re.S)
_TD_RE = re.compile(r"<td>(.*?)</td>", re.S)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _parse_michelin_table(html: str) -> List[Dict[str, str]]:
    """wikitable rows -> [{name, chef, city, since}]. Header rows (scope=col)
    are skipped naturally: they have no scope=row cell."""
    out: List[Dict[str, str]] = []
    for row in _ROW_RE.finditer(html or ""):
        inner = row.group(1)
        name_m = _ROWHEAD_RE.search(inner)
        if not name_m:
            continue
        name = _strip_tags(name_m.group(1))
        if not name:
            continue
        tds = [_strip_tags(t.group(1)) for t in _TD_RE.finditer(inner)]
        chef = tds[0] if len(tds) > 0 else ""
        city = tds[1] if len(tds) > 1 else ""
        since_m = _YEAR_RE.search(tds[2] if len(tds) > 2 else "")
        out.append({
            "name": name[:80],
            "chef": chef[:60],
            "city": city[:60],
            "since": since_m.group(0) if since_m else "",
        })
    return out


def michelin_three_star(destination: str, data_dir: str = "") -> Dict[str, Any]:
    """Michelin three-star restaurants for the destination's COUNTRY (Wikipedia
    list, official API). City destinations keep their country's list with the
    destination's own-city rows first.

    Returns {country, restaurants, count, source, note, city_priority}.
    Never fabricates: no country section -> count 0 + an honest note."""
    country = _target_country(destination)
    if not country:
        return {
            "country": "", "restaurants": [], "count": 0, "source": MICHINEL_URL,
            "note": f"'{destination}' is not in the local world dataset, so I can't tie it to a "
                    "country's Michelin three-star list — no invented rankings.",
            "city_priority": [],
        }
    key = _cache_key("mich", _norm(country))
    hit = _cache_get(data_dir, key)
    if hit:
        return dict(hit)

    sections = _api_json({
        "action": "parse", "page": MICHINEL_PAGE,
        "prop": "sections", "format": "json", "formatversion": "2",
    })["parse"]["sections"]
    want = _norm(country)
    sec = None
    for s in sections:
        line = _norm(_html.unescape(s.get("line", "")))
        if not line:
            continue
        # exact first, then containment ('France & Monaco' holds 'Monaco')
        if line == want:
            sec = s
            break
        if sec is None and want in line:
            sec = s
    if sec is None:
        result = {
            "country": country, "restaurants": [], "count": 0, "source": MICHINEL_URL,
            "note": f"{country} has no section in the Wikipedia three-star list "
                    "(no currently-listed three-star restaurants) — said plainly, not filled in.",
            "city_priority": [],
        }
        _cache_put(data_dir, key, result)
        return result

    text = _api_json({
        "action": "parse", "page": MICHINEL_PAGE, "section": str(sec["index"]),
        "prop": "text", "format": "json", "formatversion": "2",
    })["parse"]["text"]
    rows = _parse_michelin_table(text)

    # city destinations: surface their own city's restaurants first
    city_priority: List[str] = []
    dest_city = _norm(destination)
    if dest_city and dest_city != want:
        own = [r for r in rows if _norm(r["city"]) == dest_city or _norm(r["city"]).startswith(dest_city + " ")]
        if own:
            city_priority = [r["name"] for r in own]
            others = [r for r in rows if r["name"] not in set(city_priority)]
            rows = own + others

    result = {
        "country": country,
        "restaurants": rows,
        "count": len(rows),
        "source": MICHINEL_URL,
        "note": "Wikipedia's curated list of Michelin three-star restaurants (community-maintained; "
                "verify current stars before booking).",
        "city_priority": city_priority,
    }
    _cache_put(data_dir, key, result)
    return result


# ---------------------------------------------------------------------------
# global view: where the world's three-star dining concentrates
# ---------------------------------------------------------------------------
_H3_RE = re.compile(r'<h3[^>]*id="([^"]+)"[^>]*>([^<]*)</h3>')


def michelin_by_country(data_dir: str = "") -> Dict[str, Any]:
    """Honest global ranking: per-country counts of listed three-star restaurants,
    one fetch of the full article (cached 30 days). For destination-less
    'best places for a luxury experience' questions — real data, no opinion
    dressed up as fact."""
    key = _cache_key("mich_by_country")
    hit = _cache_get(data_dir, key)
    if hit:
        return dict(hit)
    text = _api_json({
        "action": "parse", "page": MICHINEL_PAGE,
        "prop": "text", "format": "json", "formatversion": "2",
    })["parse"]["text"]

    # split the article on level-3 country headings; count row-header cells per slice
    parts = _H3_RE.split(text)   # [pre, id1, title1, body1, id2, title2, body2, ...]
    counts: List[Tuple[str, int]] = []
    i = 1
    while i + 2 < len(parts):
        title = _strip_tags(parts[i + 1])
        body = parts[i + 2]
        n = len(_ROWHEAD_RE.findall(body))
        if title and n:
            counts.append((title, n))
        i += 3
    counts.sort(key=lambda t: (-t[1], t[0].lower()))
    total = sum(n for _t, n in counts)
    result = {
        "countries": [{"country": t, "three_star": n} for t, n in counts],
        "total": total,
        "source": MICHINEL_URL,
        "note": "Counts of currently-listed Michelin three-star restaurants per country "
                "(Wikipedia list, community-maintained) — a factual concentration, "
                "not a subjective 'best' ranking.",
    }
    _cache_put(data_dir, key, result)
    return result
