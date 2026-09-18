"""Optional GeoNames resolver for places outside the curated world dataset.

GeoNames is a free, attribution-required service. It is deliberately optional:
set GEONAMES_USERNAME to enable it; without credentials callers receive None and
continue with Wikimedia/OpenStreetMap resolution.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Dict, Optional

_CACHE: Dict[str, Optional[Dict[str, object]]] = {}
_UA = "Wayfinder/1.0 (travel research; GeoNames attribution required)"


def geocode(query: str) -> Optional[Dict[str, object]]:
    username = os.environ.get("GEONAMES_USERNAME", "").strip()
    key = " ".join((query or "").lower().split())
    if not username or not key:
        return None
    if key in _CACHE:
        return _CACHE[key]
    try:
        params = urllib.parse.urlencode({"q": query, "maxRows": 1, "featureClass": "P",
                                         "username": username, "style": "FULL"})
        req = urllib.request.Request("https://secure.geonames.org/searchJSON?" + params,
                                     headers={"User-Agent": _UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as response:
            row = (json.loads(response.read()).get("geonames") or [])[0]
        lat, lon = float(row["lat"]), float(row["lng"])
        hit = {"lat": lat, "lon": lon, "title": row.get("name", query),
               "country": row.get("countryName", ""), "provider": "GeoNames"}
    except (IndexError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        hit = None
    _CACHE[key] = hit
    return hit
