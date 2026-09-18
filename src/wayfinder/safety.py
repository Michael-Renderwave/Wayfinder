"""Key-free travel safety ratings — U.S. Department of State travel advisories.

The green-to-yellow-to-red "is it safe for tourists?" meter. Three key-free signals,
reconciled SAFETY-FIRST (never show a lower risk than a source indicates):

  LIVE      the OFFICIAL travel.state.gov RSS feed of travel advisories
            (/_res/rss/TAsTWs.xml — direct from the State Department, fresh daily,
            NOT Cloudflare-gated; verified live). Each item title carries the
            country + current level ("North Korea - Level 4: Do Not Travel").
            1-day disk cache. If the feed is unreachable, the live official page
            is read via the project's key-free DuckDuckGo search instead.
  SNAPSHOT  an official-data mirror of the State Department advisory feed
            (helios1014/US_State_Department_Travel_Advisories — a bot-synced
            snapshot of the same feed; weeks to a month stale, occasional feed
            glitches — e.g. a 2026-07-07 batch that wrongly flipped North Korea
            L4->L1, which the live feed + safety-first rule below neutralize).
  FALLBACK  when NEITHER live signal is reachable, the snapshot alone is shown
            and explicitly labeled "verify at travel.state.gov".

Reconciliation:
  both agree    -> rating, status "confirmed"
  live only     -> rating, status "live" (source: official page via search)
  snapshot only -> rating, status "snapshot" (strong "verify" note)
  disagree      -> the MORE CAUTIOUS level, status "conflict" + plain disclosure
                   (a feed glitch that "downgrades" a Do-Not-Travel country must
                   never produce a green meter)
  neither       -> SafetyError -> the tool layer says so plainly. NEVER a guess.

Honesty model: this module never invents a rating. Every failure path degrades to
"no advisory found"; every rating carries its source, status and level provenance.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import time
import urllib.request
from typing import Any, Dict, Optional, Tuple

SNAPSHOT_URL = ("https://raw.githubusercontent.com/helios1014/"
                "US_State_Department_Travel_Advisories/HEAD/USSD_TAS.csv")
RSS_URL = "https://travel.state.gov/_res/rss/TAsTWs.xml"  # OFFICIAL live feed (not gated)
USER_AGENT = "Wayfinder/1.0 (capstone travel-research demo; key-free safety data)"
TIMEOUT = 25
CACHE_TTL_DAYS = 7
RSS_TTL_DAYS = 1  # the live feed is official and fresh — cache 1 day, not 7

# ---------------------------------------------------------------------------
# RSS-specific country names -> ISO. The feed's titles use common names that
# differ from the ISO spellings in NAME_TO_ISO ('Burma' vs 'Myanmar', 'Cote d
# Ivoire' without the apostrophe, 'The Gambia', 'Kingdom of Denmark', the
# two Congos as full phrases, ...). Matched after HTML-unescape + stripping
# ' Travel Advisory' / ' - See Summaries' suffixes.
# ---------------------------------------------------------------------------
RSS_ALIASES = {
    "macau": "MO", "west bank": "PS", "gaza": "PS",
    "sint eustatius": "BQ", "bonaire": "BQ", "saba": "BQ",
    "british virgin islands": "VG", "burma": "MM", "cote d ivoire": "CI",
    "the kyrgyz republic": "KG", "the gambia": "GM", "russia": "RU",
    "cabo verde": "CV", "taiwan": "TW", "kingdom of denmark": "DK",
    "the bahamas": "BS", "federated states of micronesia": "FM",
    "sint maarten": "SX",
    "democratic republic of the congo": "CD", "republic of the congo": "CG",
}

# ---------------------------------------------------------------------------
# Ambiguous city names -> this app's canonical country. 'George Town' on its own
# means the Cayman one in this project (the KB deep-dive is George Town, Cayman
# Islands); a raw geocode would pick Penang, Malaysia instead. Ask 'George Town,
# Bermuda' explicitly for the other — comma parts resolve it on their own.
# ---------------------------------------------------------------------------
CITY_HINTS = {
    "george town": "KY",
}

# ---------------------------------------------------------------------------
# Official State Department advisory scale (fixed 1-4). The meter shifts color
# green -> yellow -> orange -> red with the level; the word is what the user sees.
# ---------------------------------------------------------------------------
LEVELS = {
    1: {"advisory": "Exercise normal precautions", "phrase": "exercise normal precautions",
        "rating": "SAFE for tourists", "dot": "🟢", "meter": "🟩 🟩 🟩 🟩 🟩"},
    2: {"advisory": "Exercise increased caution", "phrase": "exercise increased caution",
        "rating": "INCREASED CAUTION", "dot": "🟡", "meter": "🟩 🟩 🟩 🟨 🟨"},
    3: {"advisory": "Reconsider travel", "phrase": "reconsider travel",
        "rating": "RECONSIDER TRAVEL", "dot": "🟠", "meter": "🟨 🟨 🟧 🟧 🟧"},
    4: {"advisory": "Do Not Travel", "phrase": "do not travel",
        "rating": "DO NOT TRAVEL", "dot": "🔴", "meter": "🟥 🟥 🟥 🟥 🟥"},
}

# ---------------------------------------------------------------------------
# Name -> ISO 3166-1 alpha-2. Frozen from ISO_codes.csv (official ISO names) shipped
# in the snapshot mirror — 249 entries, local at runtime (zero network for lookup).
# ---------------------------------------------------------------------------
NAME_TO_ISO = {
    "Afghanistan": "AF", "Albania": "AL", "Algeria": "DZ", "American Samoa": "AS",
    "Andorra": "AD", "Angola": "AO", "Anguilla": "AI", "Antarctica": "AQ",
    "Antigua and Barbuda": "AG", "Argentina": "AR", "Armenia": "AM", "Aruba": "AW",
    "Australia": "AU", "Austria": "AT", "Azerbaijan": "AZ", "Bahamas": "BS",
    "Bahrain": "BH", "Bangladesh": "BD", "Barbados": "BB", "Belarus": "BY",
    "Belgium": "BE", "Belize": "BZ", "Benin": "BJ", "Bermuda": "BM",
    "Bhutan": "BT", "Bolivia, Plurinational State of": "BO", "Bonaire, Sint Eustatius and Saba": "BQ", "Bosnia and Herzegovina": "BA",
    "Botswana": "BW", "Bouvet Island": "BV", "Brazil": "BR", "British Indian Ocean Territory": "IO",
    "Brunei Darussalam": "BN", "Bulgaria": "BG", "Burkina Faso": "BF", "Burundi": "BI",
    "Cambodia": "KH", "Cameroon": "CM", "Canada": "CA", "Cape Verde": "CV",
    "Cayman Islands": "KY", "Central African Republic": "CF", "Chad": "TD", "Chile": "CL",
    "China": "CN", "Christmas Island": "CX", "Cocos (Keeling) Islands": "CC", "Colombia": "CO",
    "Comoros": "KM", "Congo": "CG", "Congo, the Democratic Republic of the": "CD", "Cook Islands": "CK",
    "Costa Rica": "CR", "Croatia": "HR", "Cuba": "CU", "Curaçao": "CW",
    "Cyprus": "CY", "Czech Republic": "CZ", "Côte d'Ivoire": "CI", "Denmark": "DK",
    "Djibouti": "DJ", "Dominica": "DM", "Dominican Republic": "DO", "Ecuador": "EC",
    "Egypt": "EG", "El Salvador": "SV", "Equatorial Guinea": "GQ", "Eritrea": "ER",
    "Estonia": "EE", "Eswatini": "SZ", "Ethiopia": "ET", "Falkland Islands (Malvinas)": "FK",
    "Faroe Islands": "FO", "Fiji": "FJ", "Finland": "FI", "France": "FR",
    "French Guiana": "GF", "French Polynesia": "PF", "French Southern Territories": "TF", "Gabon": "GA",
    "Gambia": "GM", "Georgia": "GE", "Germany": "DE", "Ghana": "GH",
    "Gibraltar": "GI", "Greece": "GR", "Greenland": "GL", "Grenada": "GD",
    "Guadeloupe": "GP", "Guam": "GU", "Guatemala": "GT", "Guernsey": "GG",
    "Guinea": "GN", "Guinea-Bissau": "GW", "Guyana": "GY", "Haiti": "HT",
    "Heard Island and McDonald Islands": "HM", "Holy See (Vatican City State)": "VA", "Honduras": "HN", "Hong Kong": "HK",
    "Hungary": "HU", "Iceland": "IS", "India": "IN", "Indonesia": "ID",
    "Iran, Islamic Republic of": "IR", "Iraq": "IQ", "Ireland": "IE", "Isle of Man": "IM",
    "Israel": "IL", "Italy": "IT", "Jamaica": "JM", "Japan": "JP",
    "Jersey": "JE", "Jordan": "JO", "Kazakhstan": "KZ", "Kenya": "KE",
    "Kiribati": "KI", "Korea, Democratic People's Republic of": "KP", "Korea, Republic of": "KR", "Kuwait": "KW",
    "Kyrgyzstan": "KG", "Lao People's Democratic Republic": "LA", "Latvia": "LV", "Lebanon": "LB",
    "Lesotho": "LS", "Liberia": "LR", "Libya": "LY", "Liechtenstein": "LI",
    "Lithuania": "LT", "Luxembourg": "LU", "Macao": "MO", "Macedonia, the Former Yugoslav Republic of": "MK",
    "Madagascar": "MG", "Malawi": "MW", "Malaysia": "MY", "Maldives": "MV",
    "Mali": "ML", "Malta": "MT", "Marshall Islands": "MH", "Martinique": "MQ",
    "Mauritania": "MR", "Mauritius": "MU", "Mayotte": "YT", "Mexico": "MX",
    "Micronesia, Federated States of": "FM", "Moldova, Republic of": "MD", "Monaco": "MC", "Mongolia": "MN",
    "Montenegro": "ME", "Montserrat": "MS", "Morocco": "MA", "Mozambique": "MZ",
    "Myanmar": "MM", "Namibia": "NA", "Nauru": "NR", "Nepal": "NP",
    "Netherlands": "NL", "New Caledonia": "NC", "New Zealand": "NZ", "Nicaragua": "NI",
    "Niger": "NE", "Nigeria": "NG", "Niue": "NU", "Norfolk Island": "NF",
    "Northern Mariana Islands": "MP", "Norway": "NO", "Oman": "OM", "Pakistan": "PK",
    "Palau": "PW", "Palestine, State of": "PS", "Panama": "PA", "Papua New Guinea": "PG",
    "Paraguay": "PY", "Peru": "PE", "Philippines": "PH", "Pitcairn": "PN",
    "Poland": "PL", "Portugal": "PT", "Puerto Rico": "PR", "Qatar": "QA",
    "Romania": "RO", "Russian Federation": "RU", "Rwanda": "RW", "Réunion": "RE",
    "Saint Barthélemy": "BL", "Saint Helena, Ascension and Tristan da Cunha": "SH", "Saint Kitts and Nevis": "KN", "Saint Lucia": "LC",
    "Saint Martin (French part)": "MF", "Saint Pierre and Miquelon": "PM", "Saint Vincent and the Grenadines": "VC", "Samoa": "WS",
    "San Marino": "SM", "Sao Tome and Principe": "ST", "Saudi Arabia": "SA", "Senegal": "SN",
    "Serbia": "RS", "Seychelles": "SC", "Sierra Leone": "SL", "Singapore": "SG",
    "Sint Maarten (Dutch part)": "SX", "Slovakia": "SK", "Slovenia": "SI", "Solomon Islands": "SB",
    "Somalia": "SO", "South Africa": "ZA", "South Georgia and the South Sandwich Islands": "GS", "South Sudan": "SS",
    "Spain": "ES", "Sri Lanka": "LK", "Sudan": "SD", "Suriname": "SR",
    "Svalbard and Jan Mayen": "SJ", "Sweden": "SE", "Switzerland": "CH", "Syrian Arab Republic": "SY",
    "Taiwan, Province of China": "TW", "Tajikistan": "TJ", "Tanzania, United Republic of": "TZ", "Thailand": "TH",
    "Timor-Leste": "TL", "Togo": "TG", "Tokelau": "TK", "Tonga": "TO",
    "Trinidad and Tobago": "TT", "Tunisia": "TN", "Turkey": "TR", "Turkmenistan": "TM",
    "Turks and Caicos Islands": "TC", "Tuvalu": "TV", "Uganda": "UG", "Ukraine": "UA",
    "United Arab Emirates": "AE", "United Kingdom": "GB", "United States": "US", "United States Minor Outlying Islands": "UM",
    "Uruguay": "UY", "Uzbekistan": "UZ", "Vanuatu": "VU", "Venezuela, Bolivarian Republic of": "VE",
    "Viet Nam": "VN", "Virgin Islands, British": "VG", "Virgin Islands, U.S.": "VI", "Wallis and Futuna": "WF",
    "Western Sahara": "EH", "Yemen": "YE", "Zambia": "ZM", "Zimbabwe": "ZW",
    "Åland Islands": "AX",
}
# Common travel names -> ISO code (world.json names that differ from the formal
# ISO spelling, plus common aliases).
NAME_TO_ISO.update({
    "Bolivia": "BO",
    "Venezuela": "VE",
    "Moldova": "MD",
    "North Macedonia": "MK",
    "Vatican City": "VA",
    "Congo (Republic)": "CG",
    "DR Congo": "CD",
    "Tanzania": "TZ",
    "Brunei": "BN",
    "Iran": "IR",
    "North Korea": "KP",
    "South Korea": "KR",
    "Laos": "LA",
    "Palestine": "PS",
    "Syria": "SY",
    "Vietnam": "VN",
    "Micronesia": "FM",
    "United States": "US", "USA": "US", "America": "US",
    "UK": "GB", "Great Britain": "GB",
    "Czechia": "CZ", "Czech Republic": "CZ",
    "Ivory Coast": "CI",
    "Cape Verde": "CV",
    "Central African Republic": "CF",
    "Dominican Republic": "DO",
    "East Timor": "TL",
    "Falkland Islands": "FK",
    "French Polynesia": "PF",
    "Guinea-Bissau": "GW",
    "Hong Kong": "HK",
    "Macao": "MO",
    "N. Korea": "KP", "S. Korea": "KR",
    "St. Lucia": "LC", "St. Vincent": "VC", "St. Kitts": "KN",
    "Virgin Islands, U.S.": "VI",
})


class SafetyError(RuntimeError):
    """No verifiable advisory (offline, unresolvable place, no source) — caller degrades honestly."""


# ---------------------------------------------------------------------------
# name -> country resolution (frozen ISO names + common aliases, geocode fallback)
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", (s or "").strip())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


_ISO_NORM_CACHE: Dict[str, str] = {}


def _norm_iso_map() -> Dict[str, str]:
    global _ISO_NORM_CACHE
    if not _ISO_NORM_CACHE:
        _ISO_NORM_CACHE = {_norm(k): v for k, v in NAME_TO_ISO.items()}
    return _ISO_NORM_CACHE


def resolve_country(place: str) -> Tuple[str, str]:
    """'George Town, Cayman Islands' -> ('Cayman Islands', 'KY').

    Order: world dataset exact match -> whole name -> comma parts (most specific
    first) -> longest known country name inside the place -> Nominatim geocode
    (display_name's last component). Raises SafetyError when unresolvable."""
    place = (place or "").strip()
    if not place:
        raise SafetyError("no place to look up")
    nmap = _norm_iso_map()
    # 1) world dataset exact match (197 countries: name/canonical/aliases)
    try:
        from . import world as _world
        world = _world._load_world() or {}
        for c in world.get("countries", []):
            for name in [c.get("name"), c.get("canonical"), *(c.get("aliases") or [])]:
                if name and _norm(name) == _norm(place):
                    iso = nmap.get(_norm(name))
                    if iso:
                        return c.get("name") or name, iso
    except Exception:  # noqa: BLE001 — world dataset is an enhancement, never a dependency
        pass
    # 2) whole string against the frozen map
    iso = nmap.get(_norm(place))
    if iso:
        for k, v in NAME_TO_ISO.items():
            if _norm(k) == _norm(place):
                return k, iso
    # 3) comma parts, most specific last-part first (e.g. 'X, Cayman Islands')
    parts = [p.strip() for p in place.split(",") if p.strip()]
    for part in reversed(parts):
        iso = nmap.get(_norm(part))
        if iso:
            return part, iso
    # 4) substring: the longest known country name inside the place
    best = ("", 0)
    for k in NAME_TO_ISO:
        nk = _norm(k)
        if len(nk) >= 6 and nk in _norm(place):
            if len(nk) > best[1]:
                best = (k, len(nk))
    if best[0]:
        return best[0], nmap[_norm(best[0])]
    # 4b) known ambiguous city -> this app's canonical country (see CITY_HINTS)
    iso = CITY_HINTS.get(place.lower().strip())
    if iso:
        for k, v in NAME_TO_ISO.items():
            if v == iso:
                return k, iso
        return place, iso
    # 5) geocode fallback: Nominatim display_name's last component is the country
    try:
        from . import osm as _osm
        geo = _osm.geocode(place, "")
        last = (geo.get("display_name") or "").split(",")[-1].strip()
        if last:
            iso = nmap.get(_norm(last))
            if iso:
                return last, iso
    except Exception:  # noqa: BLE001
        pass
    raise SafetyError(f"could not resolve {place!r} to a country with an official advisory")


# ---------------------------------------------------------------------------
# SNAPSHOT signal: the official-data mirror (change log -> latest per ISO code)
# ---------------------------------------------------------------------------
_last_fetch = 0.0
_fetch_lock = threading.Lock()


def _http_get(url: str, timeout: int = TIMEOUT) -> bytes:
    global _last_fetch
    with _fetch_lock:
        wait = _last_fetch + 1.1 - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_fetch = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv, */*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise SafetyError(f"HTTP {e.code} fetching snapshot from {url.split('/')[2]}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise SafetyError(f"snapshot unreachable: {url.split('/')[2]} ({getattr(e, 'reason', type(e).__name__)})") from None


def _parse_snapshot(raw: bytes) -> Dict[str, Tuple[int, str]]:
    """change log (pubDate|ISO_A2|Threat-Level|Threat-Num) -> {ISO: (level, date)}.

    Keeps the LATEST row per code; skips impossible future dates (the feed has a
    2220 typo) and duplicate rows. Level comes from Threat-Num when present,
    else parsed from the 'Level N' text."""
    text = raw.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter="|")
    rows = [row for row in reader if len(row) == 4]
    if not rows:
        raise SafetyError("snapshot has no rows (format change?)")
    today_year = str(time.gmtime().tm_year)
    best: Dict[str, Tuple[str, int]] = {}
    for pub, iso, lvl, num in rows:
        iso = iso.strip().upper()
        year = pub[:4]
        if not (iso and len(iso) == 2 and year.isdigit() and year <= today_year):
            continue  # future-dated typo / malformed
        level = None
        if num.strip():
            try:
                level = int(float(num))
            except ValueError:
                level = None
        if level not in LEVELS:
            m = re.match(r"Level\s+(\d)", lvl)
            level = int(m.group(1)) if m and int(m.group(1)) in LEVELS else None
        if level is None:
            continue
        cur = best.get(iso)
        if cur is None or pub > cur[0]:
            best[iso] = (pub[:10], level)
    out = {iso: (lvl, date) for iso, (date, lvl) in best.items()}
    if len(out) < 150:
        raise SafetyError(f"snapshot looks wrong (only {len(out)} countries parsed)")
    return out


def _cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, "safety_cache.json") if data_dir else ""


def snapshot_level(iso: str, data_dir: str = "") -> Tuple[Optional[int], Optional[str]]:
    """Current advisory level for one ISO code from the snapshot (or (None, None)).

    7-day disk cache (data/safety_cache.json) keeps the 3.9MB fetch to once a week."""
    path = _cache_path(data_dir)
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                store = json.load(f)
            if (store.get("kind") == "snapshot"
                    and time.time() - float(store.get("ts", 0)) < CACHE_TTL_DAYS * 86400):
                entry = (store.get("value") or {}).get(iso)
                if entry:
                    return int(entry[0]), str(entry[1])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass  # fall through to a fresh fetch
    raw = _http_get(SNAPSHOT_URL)
    parsed = _parse_snapshot(raw)
    if path:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"kind": "snapshot", "ts": time.time(), "url": SNAPSHOT_URL,
                           "value": {k: [v[0], v[1]] for k, v in parsed.items()}}, f, ensure_ascii=False)
        except OSError:
            pass
    iso_entry = parsed.get(iso)
    if iso_entry is None:
        return None, None
    return iso_entry[0], iso_entry[1]


# ---------------------------------------------------------------------------
# LIVE signal (primary): the OFFICIAL travel.state.gov RSS feed of advisories
# ---------------------------------------------------------------------------
def _parse_rss(text: str) -> "list[tuple[str, int, str]]":
    """RSS items -> [(country_name, level, url), ...].

    Item titles look like 'North Korea - Level 4: Do Not Travel' (Mexico's adds
    a ' Travel Advisory' suffix; group pages say ' - See Summaries' and are
    skipped). HTML entities are unescaped; level must be a valid 1-4."""
    import html as _html
    pairs = re.findall(r"<item>\s*<title>(.*?)</title>\s*<link>(.*?)</link>",
                       text, re.S)
    out: "list[tuple[str, int, str]]" = []
    for title, link in pairs:
        title = _html.unescape(title).strip()
        m = re.search(r"Level\s+(\d)", title)
        if not m or int(m.group(1)) not in LEVELS:
            continue
        name = title.split(" - Level")[0]
        name = re.sub(r"\s*[-–]\s*See Summaries.*$", "", name).strip()
        name = re.sub(r"\s+Travel Advisory$", "", name, flags=re.I).strip()
        if not name:
            continue
        out.append((name, int(m.group(1)), _html.unescape(link).strip()))
    if not out:
        raise SafetyError("RSS feed has no advisory items (format change?)")
    return out


def rss_level(country: str, iso: str, data_dir: str = "") -> Optional[Tuple[int, str]]:
    """(level, url) for a country from the official live RSS feed, or None.

    1-day disk cache (data/safety_rss_cache.json) — the feed is official and
    fresh, so a week-old cache is NOT acceptable, but neither is a fetch per
    run. Matching uses the frozen alias set + RSS_ALIASES; never a guess."""
    path = os.path.join(data_dir, "safety_rss_cache.json") if data_dir else ""
    items = None
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                store = json.load(f)
            if (store.get("kind") == "rss"
                    and time.time() - float(store.get("ts", 0)) < RSS_TTL_DAYS * 86400):
                items = [tuple(x) for x in store.get("items", [])]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            items = None
    if items is None:
        raw = _http_get(RSS_URL)
        items = _parse_rss(raw.decode("utf-8", errors="replace"))
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"kind": "rss", "ts": time.time(), "url": RSS_URL,
                               "items": [list(x) for x in items]}, f, ensure_ascii=False)
            except OSError:
                pass
    cands = {_norm(country)}
    for k, v in NAME_TO_ISO.items():
        if v == iso:
            cands.add(_norm(k))
    for k, v in RSS_ALIASES.items():
        if v == iso:
            cands.add(k.lower())
    for name, level, url in items:
        if _norm(name) in cands:
            return level, url
    return None


# ---------------------------------------------------------------------------
# LIVE signal (secondary): the official page via the project's DDG search
# ---------------------------------------------------------------------------
def live_level(country: str) -> Optional[Tuple[int, str]]:
    """(level, url) from the live official advisory page, or None.

    The official page opens with its level phrase ('Exercise normal precautions',
    'Exercise increased caution', 'Reconsider travel', 'Do not travel') — we match
    those exact phrases in the search result's title/snippet. Anything else is a
    no-hit, never a guess."""
    from . import web as _web
    hits = _web.search_duckduckgo(f'travel.state.gov "{country}" travel advisory', 3)
    for h in hits:
        url = h.get("url") or ""
        if "travel.state.gov" not in url:
            continue
        text = ((h.get("title") or "") + " " + (h.get("snippet") or "")).lower()
        for lvl, meta in LEVELS.items():
            if meta["phrase"] in text:
                return lvl, url
    return None


# ---------------------------------------------------------------------------
# reconciliation (safety-first)
# ---------------------------------------------------------------------------
def travel_safety(place: str, data_dir: str = "") -> Dict[str, Any]:
    """Official travel-safety rating for a place (green->yellow->red meter).

    Returns a dict with level/status/source fields, or raises SafetyError —
    the tool layer turns that into an honest 'no advisory found' answer."""
    country, iso = resolve_country(place)
    live: Optional[Tuple[int, str]] = None
    live_source: Optional[str] = None
    snap: Optional[Tuple[int, str]] = None
    # Primary live signal: the official RSS feed (direct from travel.state.gov).
    try:
        live = rss_level(country, iso, data_dir)
        live_source = "official live feed" if live else None
    except Exception:  # noqa: BLE001 — live signal is best-effort
        live = None
    # Secondary live signal: the official page via DDG (only if the feed failed).
    if live is None:
        try:
            live = live_level(country)
            live_source = "official page (search)" if live else None
        except Exception:  # noqa: BLE001
            live = None
    try:
        snap = snapshot_level(iso, data_dir)
    except SafetyError:
        snap = None

    if live and snap:
        if live[0] == snap[0]:
            status, level = "confirmed", live[0]
        else:
            status, level = "conflict", max(live[0], snap[0])  # safety-first
    elif live:
        status, level = "live", live[0]
    elif snap:
        status, level = "snapshot", snap[0]
    else:
        raise SafetyError(
            f"no verifiable advisory for {country!r} (the official live feed, the "
            "search fallback, and the data snapshot were all unreachable) — I won't "
            "invent a safety rating")

    meta = LEVELS[level]
    return {
        "found": True,
        "place": place,
        "country": country,
        "iso": iso,
        "level": level,
        "advisory": meta["advisory"],
        "rating": meta["rating"],
        "dot": meta["dot"],
        "meter": meta["meter"],
        "status": status,
        "live_url": (live or (None, ""))[1],
        "live_level": live[0] if live else None,
        "live_source": live_source,
        "snapshot_level": snap[0] if snap else None,
        "snapshot_date": snap[1] if snap else None,
    }

