"""Tool suite (CP 1.1 + CP 2.1 tool calling).

  search_kb(query, k)        RAG retrieval over the vector store (the 'hard search')
  search_flights(...)        SearchFlight()  — sample OAG-style schedule
  search_hotels(...)         hotel analysis w/ distance-to-excursions (CP 1.1)
  book_ticket(...)           BookTicket()    — mock booking, owner permission per request
  lookup(entity)             entity lookup across KB + structured data
  flight_info(origin, dest)  pricing + travel time for ANY destination (upon request):
                             sample schedule if local, else a labeled distance-based estimate
  web_search(query, k)       live-web search (key-free DDG + Wikipedia) — the internet act
  advanced_search(query)     source-aware discovery across travel/community sources
  local_places(place, focus) REAL restaurants/attractions/hotels near a place (key-free
                             OpenStreetMap: Nominatim geocode + Overpass radius query)
  travel_safety(place)       official green->yellow->red safety meter (U.S. State Dept
                             travel advisory: live official page via key-free search
                             cross-checked against an official data snapshot)
  web_fetch(url, web_store)  fetch a page, chunk it into the web index (2nd vector store)

Every tool returns (result_dict, observation_text) so the agent's observation
step has both data for the planner and a human-readable line for the trace.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from . import web as _web
from .ranking import rank
from .vectorstore import ScoredChunk, VectorStore
from .world import effective_home, region_home  # hard bias guard: destination -> home; hotel.region -> known home

Observation = Tuple[Dict[str, Any], str]


def search_kb(store: VectorStore, query: str, k: int = 8, today: Optional[date] = None, region_hint: str = "", anchor: str = "") -> Observation:
    today = today or date.today()
    raw = store.search(query, k=k, anchor=anchor)
    if not raw:
        return {"results": []}, f"search_kb('{query}') -> 0 chunks matched"
    ranked = rank(raw, today, region_hint=region_hint, query=query,
                  home_country=effective_home(anchor))
    survivors = [s for s in ranked if not s.pruned]
    results = [
        {
            "id": s.chunk.id,
            "title": s.chunk.title,
            "text": s.chunk.text,
            "source": s.chunk.source,
            "tier": s.chunk.tier,
            "date": s.chunk.date,
            "url": s.chunk.url,
            "category": s.chunk.category,
            "scores": {
                "relevance": round(s.relevance, 4),
                "reliability": round(s.reliability, 3),
                "recency": round(s.recency, 3),
                "phrase": round(s.phrase, 3),
                "total": round(s.score, 4),
            },
        }
        for s in survivors
    ]
    pruned = [
        {"id": s.chunk.id, "title": s.chunk.title, "score": round(s.score, 4), "reason": s.prune_reason}
        for s in ranked
        if s.pruned
    ]
    if results:
        best = f"; best: '{results[0]['title']}' (score {results[0]['scores']['total']})"
    else:
        best = "; all matches set aside — no surviving evidence"
    obs = (
        f"search_kb('{query}') -> {len(results)} kept, {len(pruned)} set aside{best}"
    )
    return {"results": results, "pruned": pruned, "raw_count": len(raw)}, obs


def _load_json(path: str, default):
    import json, os
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_flights(db: List[dict], origin: str = "", destination: str = "", date: str = "", max_price: Optional[int] = None) -> Observation:
    origin = origin.upper().strip()
    destination = destination.upper().strip()
    hits: List[dict] = []
    for f in db:
        if origin and f.get("origin", "").upper() != origin:
            continue
        if destination and f.get("destination", "").upper() != destination:
            continue
        if date and f.get("date", "") != date:
            continue
        if max_price and int(f.get("price_usd", 0)) > max_price:
            continue
        hits.append(f)
    hits.sort(key=lambda f: (f.get("date", ""), int(f.get("price_usd", 0))))
    hits = hits[:8]
    obs = f"search_flights({origin or '*'} -> {destination or '*'}, {date or 'any date'}) -> {len(hits)} flights"
    return {"flights": hits}, obs


# Departure hubs used when the user names no origin of their own: three major
# US gateways, so an estimate always has a concrete "from" to measure against.
FLIGHT_HUBS = ("New York City", "Atlanta", "Los Angeles")
# parse_goal maps known origins to airport codes (MIA/ATL/MCO) — map them back
# to a geocodable city name.
_AIRPORT_CITY = {"MIA": "Miami", "ATL": "Atlanta", "MCO": "Orlando", "GCM": "George Town"}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# Calibrated one-way economy base fare (USD) by great-circle distance, as
# piecewise-linear knots (km, $). Tuned against typical real-world one-way
# economy fares: MIA-ATL ~$120–250, MIA-LAX ~$250–450, MIA-LHR ~$600–950,
# MIA-NRT ~$800–1,300, MIA-GRU ~$450–850, MIA-BOM ~$900–1,600, MIA-SYD ~$1,100–1,800.
_FARE_KNOTS: Tuple[Tuple[float, float], ...] = (
    (0, 150), (1200, 200), (2500, 300), (4000, 430),
    (6500, 700), (9000, 950), (12000, 1150), (15000, 1350), (18000, 1500),
)


def _calibrated_base_fare_usd(km: float) -> float:
    """Piecewise-linear interpolated base fare for a route of `km` great-circle distance."""
    if km <= _FARE_KNOTS[0][0]:
        return float(_FARE_KNOTS[0][1])
    if km >= _FARE_KNOTS[-1][0]:
        return float(_FARE_KNOTS[-1][1])
    for (k0, f0), (k1, f1) in zip(_FARE_KNOTS, _FARE_KNOTS[1:]):
        if k0 <= km <= k1:
            t = (km - k0) / (k1 - k0)
            return f0 + t * (f1 - f0)
    return float(_FARE_KNOTS[-1][1])  # pragma: no cover


def _calibrated_duration_h(km: float, lon_from: float, lon_to: float) -> float:
    """Block-flight hours for a route of `km` (deterministic, calibrated).

    Short hops are dominated by climb/descent + taxi; long-haul cruises closer to
    ~850 km/h effective. Westbound transoceanic legs fight the jet stream
    (NYC->Tokyo ~13.5 h vs Tokyo->NYC ~11.5 h in reality) -> a headwind penalty
    on genuinely westward long routes (normalized longitude difference).
    """
    if km < 2000:
        hours = km / 700.0 + 0.6
    elif km <= 4500:
        hours = km / 800.0 + 0.55
    else:
        hours = km / 850.0 + 0.5
    dlon = ((lon_to - lon_from + 540.0) % 360.0) - 180.0
    if km > 6000 and dlon < -15:
        hours *= 1.08  # westbound headwind correction
    return hours


def _route_fare_band(base: float, seed: str, label: str) -> Dict[str, int]:
    """Deterministic ±8% jitter (seeded by the route — same route, same numbers)
    plus the honest 0.72x–1.30x economy band around the calibrated base."""
    h = int(hashlib.sha256(f"{label}|{seed}".encode("utf-8")).hexdigest()[:8], 16)
    jitter = 1.0 + ((h / 0xFFFFFFFF) - 0.5) * 0.16
    adj = base * jitter
    return {
        "economy_low": int(round(adj * 0.72 / 5) * 5),
        "economy_high": int(round(adj * 1.30 / 5) * 5),
        "premium": int(round(adj * 2.05 / 10) * 10),
        "business": int(round(adj * 4.1 / 10) * 10),
    }


def _live_route_lookup(origin: str, dest: str) -> Optional[Observation]:
    """Live layer: REAL route data from flightconnections.com (key-free).

    Resolves both ends to IATA codes (built-in table -> Wikipedia fallback),
    fetches the route page, and returns a labeled result with direct-flight
    availability, real per-stopover durations, operating airlines and the
    route distance. Fares are NOT on the page (and this deployment uses zero
    API keys), so they come from the calibrated distance model — clearly
    separated: schedules = real (unverified tier, cited), fares = estimate.
    Returns None when codes can't be resolved or no route page exists, so the
    caller falls back to the offline model.
    """
    if not _web.WEB_ENABLED:
        return None
    d_code = _web.resolve_iata_code(dest)
    if not d_code:
        return None
    if origin:
        entries = [(origin, _web.resolve_iata_code(origin))]
    else:
        entries = [(hub, _web.resolve_iata_code(hub)) for hub in FLIGHT_HUBS]
    routes: List[dict] = []
    url = ""
    for name, code in entries:
        if not code:
            continue
        sched = _web.fetch_route_schedule(code, d_code)
        if not sched:
            continue
        routes.append(_route_from_schedule(name, code, dest, sched))
        url = url or sched.get("url", "")
    if not routes:
        return None
    econ = [r["economy_low"] for r in routes] + [r["economy_high"] for r in routes]
    result = {
        "source": "live_schedule",
        "destination": dest,
        "schedule_source": "flightconnections",
        "schedule_url": url,
        "fetched_at": time.strftime("%Y-%m-%d"),
        "routes": routes,
        "economy_low": min(econ) if econ else 0,
        "economy_high": max(econ) if econ else 0,
        "economy_mid": int(round((min(econ) + max(econ)) / 2 / 5) * 5) if econ else 0,
        "avg_duration_h": round(sum(r["duration_h"] for r in routes) / len(routes), 1),
        "note": ("Real route data (direct/stop availability, real flight times, operating "
                 "airlines) from flightconnections.com — key-free, double-check before booking. "
                 "Fares are distance-calibrated estimates: no live pricing without paid APIs."),
    }
    best = routes[0]
    direct_txt = "nonstop available" if best.get("direct") else best.get("stops", "1 stop")
    obs = (f"flight_info({origin or '3 US hubs'} -> {dest}) -> LIVE route data (flightconnections.com): "
           f"{direct_txt}; ~{result['avg_duration_h']} h typical; "
           f"economy ${result['economy_low']}–${result['economy_high']} (ESTIMATE — fares are not on the page)")
    return result, obs


def _route_from_schedule(origin_name: str, origin_code: str, dest: str, sched: Dict[str, object]) -> Dict[str, object]:
    """Merge one flightconnections route page into a per-origin route record.

    Distance: prefer the page's own figure; fall back to key-free geocode +
    haversine so the fare estimate always has a real distance.
    """
    d_km = sched.get("distance_km")
    if not d_km:
        o_city = _AIRPORT_CITY.get(origin_code, origin_name)  # 'MIA' -> 'Miami' etc.
        o_geo = _web.geocode_place(o_city)
        d_geo = _web.geocode_place(dest)
        if o_geo and d_geo:
            d_km = int(round(_haversine_km(float(o_geo["lat"]), float(o_geo["lon"]),
                                           float(d_geo["lat"]), float(d_geo["lon"]))))
    route: Dict[str, object] = {
        "origin": origin_name,
        "origin_code": origin_code,
        "distance_km": int(d_km) if d_km else None,
        "direct": bool(sched.get("direct")),
        "notification": sched.get("notification"),
        "airlines": [],
        "via_options": [],
        "stops": "nonstop" if sched.get("direct") else "1 stop",
        "duration_h": None,
        "duration_range": None,
    }
    if sched.get("direct"):
        fastest = sched.get("fastest_direct_h")
        route["duration_h"] = round(float(fastest), 1) if fastest else None
        route["airlines"] = list(sched.get("airlines") or [])[:6]
    else:
        stops_routes = [r for r in (sched.get("routes") or []) if r.get("duration_h")]
        if stops_routes:
            durs = sorted(r["duration_h"] for r in stops_routes)
            route["duration_h"] = durs[len(durs) // 2]  # median of real per-route durations
            route["duration_range"] = [durs[0], durs[-1]]
            route["stops"] = f"{max(r['stops'] for r in stops_routes)} stop(s)" if max(r["stops"] for r in stops_routes) > 1 else "1 stop"
        codes: List[str] = []
        for r in (sched.get("routes") or [])[:12]:
            for c in r.get("airline_codes") or []:
                if c not in codes:
                    codes.append(c)
        names: List[str] = []
        for c in codes:
            n = _web.airline_display(c)
            if n and n not in names:
                names.append(n)
        route["airlines"] = names[:6]
        vias: List[str] = []
        for r in (sched.get("routes") or []):
            v = (r.get("via") or "").strip()
            if v.lower().startswith("via "):
                v = v[4:].strip()
            if v and v not in vias:
                vias.append(v)
        route["via_options"] = vias[:3]
    if d_km:
        fare = _route_fare_band(_calibrated_base_fare_usd(float(d_km)), f"{origin_code}|{dest}", "flight_info")
        route.update(fare)
    return route


def flight_info(flights_db: List[dict], origin: str = "", destination: str = "", date: str = "") -> Observation:
    """Flight pricing + average travel time for ANY destination (upon user request).

    Three honest paths (CP 1.1: never present a guess as live data):
      1. If the local sample schedule serves this destination (GCM today), return
         those flights — the same data SearchFlight() uses, so the Cayman
         booking flow stays intact.
      2. LIVE route data (best-effort, key-free): IATA code resolution +
         flightconnections.com route page — real direct/stop availability, real
         per-stopover flight times, operating airlines (cited, unverified tier).
         Fares on this path are still labeled estimates (no live pricing APIs).
      3. DETERMINISTIC distance model: key-free Wikipedia geocode -> great-circle
         distance -> calibrated piecewise fare/duration curves. Clearly labeled
         'estimate — not live pricing'. Offline or un-geocodable -> an honest
         'no estimate' — never invent a fare or a position.
    """
    dest = (destination or "").strip()
    if not dest:
        return ({"source": "none", "destination": "", "routes": [],
                 "note": "no destination given"},
                "flight_info() -> no destination to estimate")

    # Optional free-quota Amadeus integration. It needs a real origin, IATA
    # destination, date, and user-supplied credentials; otherwise the existing
    # key-free paths below remain the deterministic default.
    if _web.WEB_ENABLED and date:
        try:
            from . import amadeus as _amadeus
            a_origin = _web.resolve_iata_code(origin) if origin else None
            a_dest = _web.resolve_iata_code(dest)
            offers = _amadeus.flight_offers(a_origin or "", a_dest or "", date)
            if offers:
                offers.update(destination=dest, origin=a_origin, destination_code=a_dest, date=date)
                return offers, (f"flight_info({a_origin}->{a_dest}, {date}) -> {len(offers['offers'])} "
                                "Amadeus offer(s) (free-quota integration; verify before booking)")
        except Exception:
            pass  # credentials/quota/network failures always fall through safely

    # 1) Local sample schedule first (keeps the Cayman booking flow intact).
    #    The schedule is keyed by airport code (GCM): check the raw destination
    #    name, and GCM ONLY when the destination is Cayman-related — never for
    #    other places (an MIA origin would otherwise leak Cayman flights into a
    #    Tokyo answer — CP 1.1: no misleading info).
    local_dests = [destination]
    if re.search(r"cayman|george town", destination, re.I):
        local_dests.append("GCM")
    local_flights: List[dict] = []
    for d in local_dests:
        local_res, _ = search_flights(flights_db, origin=origin, destination=d, date="")
        local_flights = local_res.get("flights", [])
        if local_flights:
            break
    if local_flights:
        return ({"source": "schedule", "destination": dest, "flights": local_flights,
                 "note": "sample schedule (sample data) — not live pricing"},
                f"flight_info({origin or 'hub'} -> {dest}) -> {len(local_flights)} sample-schedule flight(s)")

    # 2) LIVE route data (best-effort): real direct/stop availability, real
    #    per-stopover flight times, operating airlines — from flightconnections.com
    #    (key-free, cited, unverified tier). Fares here are still labeled
    #    estimates (the page carries no fares; this deployment uses zero API keys).
    live = _live_route_lookup(origin, dest)
    if live is not None:
        return live

    # 3) Recalibrated distance model (deterministic, offline-safe).
    dest_geo = _web.geocode_place(dest)
    if not dest_geo:
        return ({"source": "none", "destination": dest, "routes": [],
                 "note": ("live research is off (offline mode)" if not _web.WEB_ENABLED
                          else f"couldn't pin {dest!r} on a map for a distance-based estimate")},
                f"flight_info({origin or 'hub'} -> {dest}) -> no estimate: "
                + ("offline mode" if not _web.WEB_ENABLED else f"{dest!r} couldn't be geocoded")
                + " — not guessing at prices")

    if origin and origin in _AIRPORT_CITY:
        origin_list = [(_AIRPORT_CITY[origin], _web.geocode_place(_AIRPORT_CITY[origin]))]
    elif origin:
        origin_list = [(origin, _web.geocode_place(origin))]
    else:
        origin_list = [(hub, _web.geocode_place(hub)) for hub in FLIGHT_HUBS]

    routes: List[dict] = []
    for name, og in origin_list:
        if not og:
            continue
        km = _haversine_km(float(og["lat"]), float(og["lon"]),
                           float(dest_geo["lat"]), float(dest_geo["lon"]))
        if not (100 <= km <= 21000):  # sanity: same-city geocode noise or a bad match
            continue
        # Calibrated base fare (piecewise-linear in distance) with a small
        # deterministic ±8% jitter seeded by the route — same route, same numbers.
        base = _calibrated_base_fare_usd(km)
        fare = _route_fare_band(base, f"{name}|{dest}", "flight_info")
        duration = _calibrated_duration_h(km, float(og["lon"]), float(dest_geo["lon"]))
        stops = "nonstop" if duration <= 14.5 else "1 stop"  # real nonstops run to ~14.5 h
        if stops == "1 stop":
            duration += 2.0  # typical layover
        routes.append({
            "origin": name,
            "distance_km": int(round(km)),
            "duration_h": round(duration, 1),
            "stops": stops,
            **fare,
        })
    if not routes:
        return ({"source": "none", "destination": dest, "routes": [],
                 "note": "no geocodable origin available for this estimate"},
                f"flight_info({origin or 'hub'} -> {dest}) -> no estimate: no geocodable origin")

    econ = [r["economy_low"] for r in routes] + [r["economy_high"] for r in routes]
    result = {
        "source": "estimate",
        "schedule_source": "model",
        "destination": dest,
        "dest_geo": {"lat": float(dest_geo["lat"]), "lon": float(dest_geo["lon"]),
                     "title": dest_geo.get("title", "")},
        "routes": routes,
        "economy_low": min(econ),
        "economy_high": max(econ),
        "economy_mid": int(round((min(econ) + max(econ)) / 2 / 5) * 5),
        "avg_duration_h": round(sum(r["duration_h"] for r in routes) / len(routes), 1),
        "note": ("Estimate from a calibrated distance model (key-free geocode + tuned fare/duration "
                 "curves) — NOT live pricing; zero API keys used. Fares vary by airline, season and dates."),
    }
    obs = (f"flight_info({origin or '3 US hubs'} -> {dest}) -> {len(routes)} route estimate(s); "
           f"economy ${min(econ)}–${max(econ)}, ~{result['avg_duration_h']} h typical "
           "(ESTIMATE — not live pricing)")
    return result, obs


# ---------------------------------------------------------------------------
# best_time — the "best time to travel" tool (key-free seasons dataset)
# ---------------------------------------------------------------------------
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]

_seasons_cache: Optional[Dict[str, Any]] = None  # module cache of data/seasons.json
_world_cache: Optional[Dict[str, Any]] = None    # module cache of data/world.json


def _seasons_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_seasons() -> Dict[str, Any]:
    global _seasons_cache
    if _seasons_cache is None:
        path = os.path.join(_seasons_root(), "data", "seasons.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _seasons_cache = json.load(f)
        else:
            _seasons_cache = {"data": {}, "countries": []}
    return _seasons_cache


def _load_world() -> Dict[str, Any]:
    global _world_cache
    if _world_cache is None:
        path = os.path.join(_seasons_root(), "data", "world.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _world_cache = json.load(f)
        else:
            _world_cache = {"countries": [], "cities": {}}
    return _world_cache


def _resolve_season_country(destination: str) -> tuple[str, str]:
    """destination -> (canonical_country, how). ASCII-folded matching:
    country name -> alias -> city (world.json cities map) -> top destination.
    ('', '') when nothing resolves — the caller then returns an honest fallback."""
    from .world import ascii_norm
    target = ascii_norm(destination or "").lower().strip()
    if not target:
        return "", ""
    seasons = _load_seasons()
    world = _load_world()
    # 1) exact country name in the seasons dataset
    for name in seasons.get("countries", []):
        if ascii_norm(name).lower() == target:
            return name, "country name"
    # 2) country alias (world.json)
    for c in world.get("countries", []):
        if ascii_norm(c.get("name", "")).lower() == target or \
                any(ascii_norm(a).lower() == target for a in c.get("aliases", [])):
            return c.get("name", ""), "alias"
    # 3) city / place -> its country
    city = (world.get("cities", {}) or {}).get(target)
    if city and city.get("country"):
        return city["country"], f"city '{(destination or '').strip()}'"
    # 4) top-destination token (e.g. 'Patagonia' -> Argentina)
    for c in world.get("countries", []):
        for d in c.get("top_destinations", []):
            if ascii_norm(d).lower() == target:
                return c.get("name", ""), f"destination '{(destination or '').strip()}'"
    return "", ""


def _month_verdict(rec: dict, month: int) -> Dict[str, Any]:
    """A plain verdict for ONE month the user asked about ('is March good for Rome?')."""
    m = next((x for x in rec.get("months", []) if x["m"] == month), None)
    if not m:
        return {"month": month, "verdict": "I don't have month-level data for this place yet."}
    band = m["band"]
    best = rec.get("best_month", "")
    if band == "ideal":
        verdict = (f"{_month_names(month)} is in the IDEAL window for {rec['country']} — "
                   f"one of the top-3 months by comfort and dryness"
                   + (f" (the single best month is {best})" if best != m["name"] else " — in fact it's the best month") + ".")
    elif band == "shoulder":
        verdict = (f"{_month_names(month)} is a SHOULDER month for {rec['country']} — workable, "
                   f"but the top-3 window ({', '.join(rec.get('ideal', []))}) scores better on "
                   "comfort and dryness.")
    else:
        verdict = (f"{_month_names(month)} falls in the OFF season for {rec['country']} by the "
                   f"comfort+dryness score — the top-3 window is {', '.join(rec.get('ideal', []))}. "
                   "It's still possible to travel then; just expect less ideal weather.")
    return {"month": month, "name": m["name"], "band": band, "verdict": verdict, "data": m}


def _month_names(n: int) -> str:
    return _MONTH_NAMES[n - 1] if 1 <= n <= 12 else "that month"


def best_time(destination: str, month: Optional[int] = None) -> Observation:
    """Best time to visit — seasonal climate + holidays from the key-free dataset.

    data/seasons.json is built by scripts/build_seasons.py from Open-Meteo climate
    normals (2023-2025) + Wikipedia public holidays + Wikivoyage climate notes —
    zero API keys. Honesty model: a country without climate coverage returns
    covered:false and a plain note — never invented months.
    """
    dest = (destination or "").strip()
    country, how = _resolve_season_country(dest)
    if not country:
        result = {
            "country": "",
            "destination": dest,
            "covered": False,
            "note": (f"I've searched far and wide, but I just couldn't find what you might have "
                     f"been looking for — {dest or 'this place'} isn't in my "
                     f"{len((_load_seasons().get('data') or {}))}-country seasons "
                     "dataset, so I won't guess at months."),
        }
        return result, f"best_time({dest or '*'}) -> not in the seasons dataset — honest fallback, no invented months"

    rec = _load_seasons().get("data", {}).get(country, {})
    result = {
        "country": country,
        "destination": dest,
        "matched_from": how,
        "covered": bool(rec.get("covered")),
        "reference_point": rec.get("reference_point", ""),
        "climate_years": rec.get("climate_years", ""),
        "months": rec.get("months", []),
        "ideal": rec.get("ideal", []),
        "shoulder": rec.get("shoulder", []),
        "off": rec.get("off", []),
        "best_month": rec.get("best_month", ""),
        "best_month_num": rec.get("best_month_num"),
        "prose": rec.get("prose", ""),
        "prose_url": rec.get("prose_url", ""),
        "holidays": rec.get("holidays", []),
        "holidays_url": rec.get("holidays_url", ""),
        "sources": rec.get("sources", []),
    }
    if not rec.get("covered"):
        result["note"] = rec.get("note", "No verified climate data for this country yet — I won't guess at months.")
        obs = (f"best_time({dest or '*'}) -> {country}: no climate coverage in the dataset "
               "— honest fallback (never invented months)")
        return result, obs
    if month is not None and 1 <= int(month) <= 12:
        result["month_verdict"] = _month_verdict(rec, int(month))
    obs = (f"best_time({dest or '*'}) -> {country} (matched via {how}): "
           f"ideal {', '.join(result['ideal'])}, best month {result['best_month']}, "
           f"{len(result['holidays'])} holiday(s)"
           + (f"; verdict for month {int(month)}: {result['month_verdict']['band']}"
              if result.get("month_verdict") else ""))
    return result, obs


_GENERIC_HOTEL_WORDS = {"islands", "island", "city", "national", "province", "region", "the"}


def search_hotels(db: List[dict], destination: str = "", max_price: Optional[int] = None, near: str = "") -> Observation:
    dest = destination.lower().strip()
    dest_words = {w for w in dest.split() if len(w) > 2}
    # Generic words ('islands', 'city', ...) appear in many destination names and
    # must not by themselves qualify a hotel: a 'Cayman Islands' search must not
    # match a Solomon 'Islands' hotel. Qualify on a distinctive word instead.
    matchers = {w for w in dest_words if w not in _GENERIC_HOTEL_WORDS} or dest_words
    home = effective_home(destination)  # '' for demos/unknown -> keep soft matching only
    hits: List[dict] = []
    for h in db:
        if dest:
            hay = " ".join(str(h.get(k, "")) for k in ("regions", "area", "city"))
            hay = hay.lower().replace("[", " ").replace("]", "").replace(",", " ")
            if not any(w in hay for w in matchers):
                continue
            # HARD bias guard (CP 3.1): a bare substring ('north', 'korea') matches the
            # WRONG hotels — 'North Macedonia' would pull in Pyongyang (north korea),
            # Pittsburgh (north hills) and Grand Cayman (north shore). If the destination
            # has a determinable home and this hotel's own `regions` name a DIFFERENT
            # known place, drop it (omit over misattribute, CP 1.1). region_home('')
            # (undecidable) or home=='' (curated demos) keep the soft match above.
            if home:
                regions_str = " ".join(str(r) for r in (h.get("regions") or []))
                rh = region_home(regions_str)
                if rh and rh != home:
                    continue
        if max_price and int(h.get("price_usd", 0)) > max_price:
            continue
        hits.append(h)
    # CP 1.1: rank by proximity to the airport OR the main destinations people
    # actually go (the info the user needs: near airport / near the sights).
    if near:
        hits.sort(key=lambda h: (h.get("distance_to_excursion_mi", 99), int(h.get("price_usd", 0))))
    else:
        hits.sort(key=lambda h: (min(h.get("distance_to_airport_mi", 99), h.get("distance_to_excursion_mi", 99)),
                                 int(h.get("price_usd", 0))))
    hits = hits[:8]
    obs = (f"search_hotels({destination or '*'}, max ${max_price or 'any'}{' near ' + near if near else ''}) -> "
           f"{len(hits)} hotels (near airport / main destinations first)")
    return {"hotels": hits}, obs


def book_ticket(bookings: List[dict], flight: dict, passenger: str = "Traveler", confirmed: bool = False) -> Observation:
    if not confirmed:
        obs = f"book_ticket({flight.get('flight_no')}) -> PENDING: your confirmation is needed"
        return {"status": "pending_confirmation", "flight": flight}, obs
    ref = f"WF-{len(bookings) + 1:04d}"
    booking = {
        "ref": ref,
        "flight": flight,
        "passenger": passenger,
        "status": "confirmed (sample — no real charge)",
        "price_usd": flight.get("price_usd"),
    }
    bookings.append(booking)
    obs = f"book_ticket({flight.get('flight_no')}) -> CONFIRMED ref {ref} (${flight.get('price_usd')} sample)"
    return booking, obs


WEB_INDEX_DOCS_MAX = 120  # cap on cached web pages (CP 1.1: guard rails on useless info)


def web_search_tool(query: str, k: int = 5) -> Observation:
    """Act: search the live web (CP 2.1 — the agent actively seeks information).

    Merges DuckDuckGo (HTML endpoint) + Wikipedia (MediaWiki API); both key-free.
    Never raises: offline/blocked -> {'results': [], 'error': ...} so the agent
    degrades gracefully to KB-only (CP 1.1 guardrail: never dead-end).
    """
    if not _web.WEB_ENABLED:
        return ({"results": [], "error": "web research disabled (offline-first mode)"},
                "web_search is disabled in this deployment (offline-first mode) — "
                "answering from the local knowledge base only")
    try:
        results = _web.web_search(query, k)
    except _web.WebError as exc:
        return ({"results": [], "error": str(exc)},
                f"web_search({query!r}) -> OFFLINE: {exc} — falling back to the guide library")
    best = results[0] if results else {}
    obs = (f"web_search({query!r}) -> {len(results)} result(s); "
           f"best: '{best.get('title', '')}' via {best.get('provider', '?')}")
    return {"results": results}, obs


def advanced_search_tool(query: str, k_per_source: int = 3) -> Observation:
    """Gather structured public research leads beyond a general web search.

    This is discovery only: Reddit/Instagram are never scraped or fetched behind a
    login wall, while public results from travel/reference sites may be selected for
    the normal SSRF-protected fetch-and-index step.
    """
    data = _web.advanced_search(query, max(1, min(int(k_per_source), 5)))
    results = data.get("results", [])
    errors = data.get("errors", {})
    by_source: Dict[str, int] = {}
    for item in results:
        source = item.get("source_type", "other")
        by_source[source] = by_source.get(source, 0) + 1
    summary = ", ".join(f"{name}: {count}" for name, count in sorted(by_source.items())) or "no public leads"
    failed = f"; unavailable: {', '.join(sorted(errors))}" if errors else ""
    return ({"results": results, "by_source": by_source, "errors": errors,
             "query": query, "sources": data.get("sources", [])},
            f"advanced_search({query!r}) -> {len(results)} structured lead(s) ({summary}){failed}")


def _usable_place_name(name: str) -> bool:
    """A real mapped name has at least one letter. OpenStreetMap sometimes carries
    a bare number in `name` (a house number / an unnamed node — e.g. '305') and
    that reads as garbage in a recommendation ('eat at 305'), so we drop any name
    with no alphabetic character. Real names like '389 Burguer' keep their digits."""
    return any(ch.isalpha() for ch in (name or ""))


def local_places_tool(place: str, focus: str = "dining", k: int = 12,
                      data_dir: str = "", raw_context: str = "") -> Observation:
    """Act: find REAL named places near a destination (dining, attractions, hotels,
    or pharmacies/hospitals/police-fire when the user asks about medical/urgent
    care or safety services).

    Key-free OpenStreetMap (Nominatim geocode + Overpass radius query) — the API
    answer to 'dining options, places to go, activities'. Returns actual mapped
    names + street addresses, never invented ones. Honesty model: offline, no
    geocode hit, or API failure -> {'places': [], 'error': ...} + a plain note,
    so the agent composes honestly instead of guessing (CP 1.1 / never invent).

    `raw_context` (the user's own sentence) is used ONLY to disambiguate a bare
    city name: 'George Town' geocodes to Malaysia, so when the sentence also
    names a known country the qualified 'George Town, Cayman Islands' form is
    tried first — never to invent a destination.
    """
    if not _web.WEB_ENABLED:
        return ({"places": [], "error": "web research disabled (offline-first mode)"},
                "local_places is disabled in this deployment (offline-first mode) — "
                "answering from the local knowledge base only")
    from . import osm as _osm  # local import: keeps tools.py import-light when web is off
    from . import world as _w

    # Disambiguation candidates: the qualified 'city, country' form FIRST when the
    # user's sentence names a known country that the bare place does not carry —
    # bare city names are exactly the ones Nominatim resolves to the wrong country
    # ('George Town' -> Pulau Pinang, Malaysia, not the Cayman Islands). The bare
    # form stays as the fallback when the qualified one doesn't resolve.
    candidates = [place]
    ctx = _w.ascii_norm(raw_context or "").lower()
    if ctx and place:
        low_place = _w.ascii_norm(place).lower()
        for c in sorted(_w.country_canonicals(), key=len, reverse=True):
            nc = _w.ascii_norm(c).lower()
            if len(nc) < 4 or nc in low_place:
                continue
            if re.search(rf"\b{re.escape(nc)}\b", ctx):
                candidates.insert(0, f"{place}, {c}")
                break
    data = None
    last_exc: Optional[str] = None
    for cand in candidates:
        try:
            data = _osm.nearby_places(cand, focus, limit=max(2, min(int(k), 48)), data_dir=data_dir)
            break
        except _osm.OSMError as exc:
            last_exc = str(exc)  # try the next candidate (qualified -> bare)
    if data is None:
        return ({"place": place, "focus": focus, "places": [], "error": last_exc or "no data"},
                f"local_places({place!r}, {focus}) -> no data: {last_exc} — "
                "I won't invent place names")
    # Drop unmappable names (a bare number like '305') so neither the quick report
    # nor the Local-places section ever recommends a house number as a place.
    places = [p for p in data.get("places", []) if _usable_place_name(p.get("name", ""))]
    dining = [p for p in places if p["category"] in _osm.DINING_CATEGORIES]
    health = [p for p in places if p["category"] in _osm.HEALTH_CATEGORIES]
    others = [p for p in places if p["category"] not in _osm.DINING_CATEGORIES
              and p["category"] not in _osm.HEALTH_CATEGORIES]
    fact = _osm.wikidata_fact(place, data_dir)  # best-effort one-line description
    best = places[0]["name"] if places else "(none found)"
    obs = (f"local_places({place!r}, {focus}) -> {len(places)} real place(s) near "
           f"{data.get('resolved_as', place)} "
           f"({len(dining)} dining, {len(health)} health, {len(others)} other); "
           f"first: {best!r} — OpenStreetMap (key-free)")
    result = {"place": place, "focus": focus, "places": places, **{k2: data.get(k2)
                                                                    for k2 in ("resolved_as", "lat", "lon", "radius_m", "counts")}}
    if fact:
        result["fact"] = fact
    return result, obs


def luxury_experiences_tool(destination: str, data_dir: str = "") -> Observation:
    """Act: LUXURY experiences for a destination (upon request).

    Three key-free layers, each degrading honestly on its own:
      * STAYS      — OSM hotels/resorts with a community star_rating of 4-5
      * EXPERIENCES — OSM golf courses, spas, marinas, wineries near the anchor
      * FINE DINING — Michelin three-star restaurants from the Wikipedia list
                      (official API, per the destination's country; the destination's
                      own city's rows first)
    With no destination, it answers 'where are the best places for a luxury
    experience?' with the honest global view: per-country counts of listed
    three-star restaurants (factual concentration, not a subjective ranking).
    Honesty model (CP 1.1): offline / no data -> empty buckets + plain notes;
    this tool NEVER invents a hotel, experience or star rating.
    """
    destination = (destination or "").strip()
    if not _web.WEB_ENABLED:
        return ({"destination": destination, "error": "web research disabled (offline-first mode)"},
                "luxury_experiences is disabled in this deployment (offline-first mode) — "
                "answering from the local knowledge base only")
    from . import osm as _osm
    from . import luxury as _lux
    from .world import ascii_norm

    # Globally recognized luxury/upscale hotel brands (OSM `operator`/`brand` tag).
    # Disclosed heuristic: the NAMES + operators are real OpenStreetMap data; the
    # 'upscale' CLASSIFICATION is Wayfinder's curated brand list (kept to chains
    # people actually book as luxury — mid-scale brands like Holiday Inn are out).
    def _n(s: str) -> str:
        return ascii_norm(s or "").lower().strip()

    # Matched as a SUBSTRING of the hotel's normalized name OR operator, because
    # OSM usually stores the corporate PARENT as `operator` — "Mandarin Oriental
    # Hotel Group", "Hyatt Hotels", "Mandarin Oriental" — not the exact guest-facing
    # brand. Substring matching resolves those to their luxury brands while still
    # EXCLUDING mid-scale chains (Holiday Inn, Best Western, Park Inn, Adagio, Exe,
    # Hilton) — none of those tokens are present, so they stay out. Deliberately
    # kept to chains people actually book as luxury; a disclosed heuristic.
    LUXURY_BRANDS = {
        "ritz-carlton", "ritz carlton", "st. regis", "st regis", "four seasons",
        "aman", "belmond", "rocco forte", "mandarin oriental", "six senses",
        "bulgari", "langham", "waldorf", "shangri-la", "peninsula",
        "conrad", "hyatt", "dorchester", "claridge", "kempinski", "banyan tree",
        "jw marriott", "westin", "marriott", "melia", "sofitel", "meridien",
    }
    _brand_tokens = {t for t in (_n(b) for b in LUXURY_BRANDS) if t}

    # Mid-scale sub-brands whose NAME contains a luxury parent token (e.g. "Courtyard
    # by Marriott" contains "marriott"). If one is in the hotel name, it is NOT a
    # luxury stay even though the parent brand matches — keeps the list honest.
    MIDSCALE = {
        "courtyard", "four points", "springhill", "element", "fairfield",
        "holiday inn", "hampton", "aloft", "home2", "traveleodge",
    }
    _mid_tokens = {t for t in (_n(b) for b in MIDSCALE) if t}

    def _is_upscale_stay(p: dict) -> bool:
        if str(p.get("stars") or "").strip() in ("4", "5"):
            return True
        name = _n(p.get("name"))
        if any(t in name for t in _mid_tokens):
            return False
        hay = (name + " " + _n(p.get("operator"))).strip()
        if not hay:
            return False
        return any(tok in hay for tok in _brand_tokens)

    if not destination:
        try:
            byc = _lux.michelin_by_country(data_dir)
            obs = (f"luxury_experiences(global) -> {byc.get('total', 0)} three-star restaurants "
                   f"across {len(byc.get('countries', []))} countries — Wikipedia list (key-free)")
            return ({"destination": "", "by_country": byc}, obs)
        except _lux.LuxuryError as exc:
            return ({"destination": "", "by_country": {"countries": [], "total": 0, "note": str(exc)}},
                    f"luxury_experiences(global) -> no data: {exc} — I won't invent a ranking")

    notes: List[str] = []
    stays: List[dict] = []
    exps: List[dict] = []
    resolved = ""
    seen = set()

    def _add_stay(p: dict) -> None:
        if p["name"].lower() in seen:
            return
        seen.add(p["name"].lower())
        stays.append(p)

    try:
        # 8km: golf courses & wineries usually sit just outside the core.
        data = _osm.nearby_places(destination, focus="luxury", radius_m=8000,
                                  limit=40, data_dir=data_dir)
        resolved = data.get("resolved_as", "") or ""
        for p in data.get("places", []):
            if not _usable_place_name(p.get("name", "")):
                continue
            if p["category"] in ("hotel", "resort"):
                _add_stay(p)
            else:
                exps.append(p)
        # second pass: star_rating is sparsely mapped, so ALSO take mapped stays run
        # by a recognized luxury brand (real OSM name + operator, disclosed heuristic).
        # limit=200: fetch the FULL hotel set (matches the Overpass body cap) so the
        # brand filter searches everything — a 48-name slice would miss M/S/W brands.
        data2 = _osm.nearby_places(destination, focus="hotels", radius_m=8000,
                                   limit=200, data_dir=data_dir)
        for p in data2.get("places", []):
            if not _usable_place_name(p.get("name", "")):
                continue
            if p["category"] in ("hotel", "resort") and _is_upscale_stay(p):
                _add_stay(p)
    except _osm.OSMError as exc:
        notes.append(f"OpenStreetMap had no 4-5★ stays/experiences nearby (or was unreachable): {exc}")

    try:
        mich = _lux.michelin_three_star(destination, data_dir)
    except _lux.LuxuryError as exc:
        mich = {"country": "", "restaurants": [], "count": 0, "source": "",
                "note": f"Wikipedia unreachable: {exc}"}
        notes.append(mich["note"])

    # 5★ first, then 4★, then brand stays, then name — the user asked for the BEST
    def _stay_rank(p: dict):
        s = str(p.get("stars") or "").strip()
        if s == "5":
            return 0
        if s == "4":
            return 1
        return 2  # recognized-luxury-brand stay (no star mapped)

    stays.sort(key=lambda p: (_stay_rank(p), p["name"].lower()))
    exps.sort(key=lambda p: p["name"].lower())

    result = {
        "destination": destination,
        "resolved_as": resolved,
        "stays": stays,
        "experiences": exps,
        "michelin": mich,
        "notes": notes,
    }
    obs = (f"luxury_experiences({destination!r}) -> {len(stays)} upscale stay(s), "
           f"{len(exps)} experience(s) "
           + (f"near {resolved} " if resolved else "")
           + f"+ {mich.get('count', 0)} Michelin three-star ({mich.get('country') or 'n/a'}) — "
           "OSM + Wikipedia (key-free)")
    return result, obs


def travel_safety_tool(place: str, data_dir: str = "") -> Observation:
    """Act: official travel-safety rating for a destination (green->yellow->red meter).

    U.S. Department of State travel advisory (levels 1-4), key-free: the OFFICIAL
    travel.state.gov RSS feed is the primary live signal (fallback: the official page
    via the project's DuckDuckGo search), cross-checked against an official data
    snapshot (7-day cache). Reconciliation is SAFETY-FIRST: if the sources disagree
    the MORE CAUTIOUS rating wins and the conflict is disclosed. Honesty model: all
    sources unreachable -> {'found': False, 'error': ...} + a plain note; this tool
    NEVER invents a safety rating (CP 1.1 / never invent).
    """
    if not _web.WEB_ENABLED:
        return ({"found": False, "error": "web research disabled (offline-first mode)"},
                "travel_safety is disabled in this deployment (offline-first mode) — "
                "no safety rating, and I won't guess at one")
    from . import safety as _safety  # local import: keeps tools.py import-light when web is off
    try:
        data = _safety.travel_safety(place, data_dir)
    except _safety.SafetyError as exc:
        return ({"found": False, "place": place, "error": str(exc)},
                f"travel_safety({place!r}) -> no verifiable advisory: {exc} — "
                "I won't invent a safety rating")
    obs = (f"travel_safety({place!r}) -> {data['country']} ({data['iso']}) "
           f"Level {data['level']} ({data['advisory']}) [{data['status']}]")
    return data, obs


def web_fetch_tool(url: str, web_store: VectorStore, save_path: str, provider: str = "") -> Observation:
    """Act: fetch one page, extract text, and index it in the web crawl store.

    The page becomes a document in a second VectorStore (data/web_index.json)
    with full provenance (url as id + provider + fetch date) — i.e. the web is
    part of the RAG index, tier 'web' (unverified), per CP 3.1's provenance rule.
    Already-cached URLs are a cache hit: stored text is reused (CP 2.1: 'cache').
    """
    if not _web.WEB_ENABLED:
        return ({"page": None, "error": "web research disabled (offline-first mode)"},
                "web_fetch is disabled in this deployment (offline-first mode)")
    # CP 2.1 cache: same URL already indexed -> reuse stored text, no re-fetch
    if url in web_store.docs:
        doc = web_store.docs[url]
        chunks = web_store.chunks_of(url)
        if chunks:
            # Leading chunks may be infobox/nav boilerplate — prefer chunks that
            # contain real prose (lines >= 100 chars) so we have material to quote.
            prose = [c for c in chunks if any(len(ln.strip()) >= 100 for ln in c.text.splitlines())]
            text = "\n".join(c.text for c in (prose[:2] or chunks[:3]))
            return (
                {"page": {"id": url, "title": doc.get("title", url), "url": url, "text": text,
                          "chars": len(text), "date": doc.get("date", ""),
                          "provider": doc.get("provider", "web"), "cached": True},
                 "cached": True},
                f"web_fetch({url}) -> cache hit: '{doc.get('title', url)}' already in the web index — "
                f"reusing stored text (cached)",
            )

    try:
        page = _web.fetch_page(url)  # SSRF guard / blocked / too little text -> WebError
    except _web.WebError as exc:
        return ({"page": None, "error": str(exc)},
                f"web_fetch({url}) -> FAILED: {exc} — continuing without this page")
    n_chunks = web_store.add_document(
        {
            "id": url,  # dedupe key
            "title": page["title"],
            "source": provider or "web",
            "tier": "web",
            "date": page["fetched_at"],
            "url": url,
            "category": "web",
            "region": "",
            "provider": provider or "web",
        },
        page["text"],
    )
    dropped = web_store.trim_to(WEB_INDEX_DOCS_MAX)
    web_store.save(save_path)
    page_out = dict(page, id=url, chunks=n_chunks, provider=provider or "web", cached=False)
    obs = (f"web_fetch({url}) -> indexed {n_chunks} chunk(s) of {page['chars']} chars "
           f"as '{page['title']}' (source: web, provenance kept)" + (f"; trimmed {dropped} older page(s)" if dropped else ""))
    return {"page": page_out}, obs


def lookup(store: VectorStore, entity: str) -> Observation:
    """Light entity probe: best single chunk mentioning the entity."""
    res, obs = search_kb(store, entity, k=4)
    results = res.get("results", [])
    if not results:
        return {"entity": entity, "found": False, "detail": None}, f"lookup('{entity}') -> no match"
    top = results[0]
    return (
        {"entity": entity, "found": True, "detail": {"title": top["title"], "text": top["text"], "source": top["source"]}},
        f"lookup('{entity}') -> found in '{top['title']}'",
    )
