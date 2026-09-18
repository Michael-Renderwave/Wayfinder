"""Optional Amadeus Self-Service flight offers integration.

It remains inactive until AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET are set.
The free quota is suitable for development; failures never prevent Wayfinder's
existing key-free route and distance-estimate fallbacks.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

_TOKEN: Dict[str, object] = {}
_BASE = os.environ.get("AMADEUS_BASE_URL", "https://test.api.amadeus.com").rstrip("/")


def configured() -> bool:
    return bool(os.environ.get("AMADEUS_CLIENT_ID") and os.environ.get("AMADEUS_CLIENT_SECRET"))


def _token() -> Optional[str]:
    if _TOKEN.get("expires", 0) > time.time() + 30:
        return str(_TOKEN.get("value"))
    payload = urllib.parse.urlencode({"grant_type": "client_credentials",
                                      "client_id": os.environ.get("AMADEUS_CLIENT_ID", ""),
                                      "client_secret": os.environ.get("AMADEUS_CLIENT_SECRET", "")}).encode()
    try:
        req = urllib.request.Request(_BASE + "/v1/security/oauth2/token", data=payload,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
        _TOKEN.update(value=data["access_token"], expires=time.time() + int(data.get("expires_in", 900)))
        return str(_TOKEN["value"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def flight_offers(origin: str, destination: str, departure_date: str) -> Optional[Dict[str, object]]:
    """Return up to three live/test offers, or None when unavailable."""
    if not (configured() and origin and destination and departure_date):
        return None
    token = _token()
    if not token:
        return None
    params = urllib.parse.urlencode({"originLocationCode": origin, "destinationLocationCode": destination,
                                     "departureDate": departure_date, "adults": 1, "max": 3,
                                     "currencyCode": "USD"})
    try:
        req = urllib.request.Request(_BASE + "/v2/shopping/flight-offers?" + params,
                                     headers={"Authorization": "Bearer " + token, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.loads(response.read())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    offers: List[Dict[str, object]] = []
    for row in (data.get("data") or [])[:3]:
        itineraries = row.get("itineraries") or []
        segments = (itineraries[0].get("segments") or []) if itineraries else []
        if not segments:
            continue
        offers.append({"price_usd": int(round(float((row.get("price") or {}).get("grandTotal", 0)))),
                       "currency": (row.get("price") or {}).get("currency", "USD"),
                       "departure": (segments[0].get("departure") or {}).get("at", ""),
                       "arrival": (segments[-1].get("arrival") or {}).get("at", ""),
                       "stops": max(0, len(segments) - 1),
                       "carriers": list(dict.fromkeys(s.get("carrierCode", "") for s in segments if s.get("carrierCode"))),
                       "duration": (itineraries[0].get("duration") if itineraries else "")})
    costs = [int(x["price_usd"]) for x in offers if x.get("price_usd")]
    if not offers or not costs:
        return None
    return {"source": "amadeus", "offers": offers, "economy_low": min(costs), "economy_high": max(costs),
            "economy_mid": int(round(sum(costs) / len(costs))), "note": "Amadeus Self-Service offer; verify before booking."}
