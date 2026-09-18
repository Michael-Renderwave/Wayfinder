"""Internet access for the agent — stdlib only (urllib + html.parser), no new deps.

Three building blocks used by tools.py:

  web_search(query, k) -> [{title, url, snippet, provider}, ...]
      DuckDuckGo (HTML endpoint) merged with Wikipedia's search API.
      Both are key-free. Failures raise WebError; callers degrade gracefully.

  advanced_search(query, k_per_source) -> {results, errors, sources, query}
      Federated, source-aware discovery (the 'more places' agent): Wikipedia,
      Wikivoyage, Tripadvisor, Expedia, Reddit, Instagram — searched concurrently
      through their public, indexed pages (key-free DDG `site:` queries).
      A result is a *lead*, not a factual claim; Reddit/Instagram are marked
      fetchable=False (never fetched behind a login wall).

  fetch_page(url) -> {title, text, chars}
      GET + HTML->text extraction (scripts/nav stripped), capped in size.
      SSRF guard: http(s) only, private/loopback/link-local targets rejected.

The fetched text is then chunked into the web crawl store (a VectorStore over
data/web_index.json) by tools.py — i.e. the web becomes part of the RAG index
with the same provenance model as local documents (CP 3.1), just tiered 'web'.
"""

from __future__ import annotations

import html as _html
import ipaddress
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from typing import Dict, List, Optional

WEB_TIMEOUT = int(os.environ.get("WEB_TIMEOUT", "10"))
WEB_MAX_BYTES = 1_500_000          # cap downloaded HTML per page
WEB_MAX_CHARS = int(os.environ.get("WEB_MAX_CHARS", "9000"))  # cap extracted text

# Live-web research is ON by default: key-free DuckDuckGo + Wikipedia plus the
# federated source-aware discovery path. Set WAYFINDER_WEB=0 before starting the
# server to run fully offline (KB only) — the CP 6.1 guardrails (L2/L3) deny the
# web tools whenever the flag is off, so this one switch is the whole gate.
WEB_ENABLED = os.environ.get("WAYFINDER_WEB", "1") == "1"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 Wayfinder/1.0")


class WebError(Exception):
    pass


# ----------------------------------------------------------------------
# low-level HTTP
# ----------------------------------------------------------------------
def _safe_url(url: str) -> bool:
    """SSRF guard: only public http(s) targets are allowed to be fetched."""
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower().strip(".")
    if not host or host in {"localhost", "0.0.0.0", "ip6-localhost"}:
        return False
    if host.endswith((".local", ".internal", ".lan", ".home.arpa")):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    except ValueError:
        return True  # hostname (not a literal IP) — allowed


def http_get(url: str, timeout: int = WEB_TIMEOUT, max_bytes: int = WEB_MAX_BYTES,
             _retries: int = 1) -> bytes:
    if not _safe_url(url):
        raise WebError(f"refusing to fetch non-public URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept-Language": "en"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise WebError(f"HTTP {resp.status} for {url}")
            return resp.read(max_bytes + 1)[:max_bytes]
    except WebError:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code == 429 and _retries > 0:
            # Rate-limited (e.g. Wikipedia burst): one short backoff retry, then give up.
            time.sleep(1.5)
            return http_get(url, timeout=timeout, max_bytes=max_bytes, _retries=_retries - 1)
        raise WebError(f"HTTP {exc.code} for {url}") from exc
    except Exception as exc:  # noqa: BLE001 — any network failure -> WebError
        raise WebError(f"{type(exc).__name__}: {exc}") from exc


# ----------------------------------------------------------------------
# HTML -> text (stdlib parser; drops script/style/nav noise)
# ----------------------------------------------------------------------
_DROP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe",
              "nav", "footer", "aside", "form", "button", "header"}
_BLOCK_TAGS = {"p", "div", "li", "ul", "ol", "section", "article", "br",
               "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._drop = 0

    def handle_starttag(self, tag, attrs):
        if tag in _DROP_TAGS:
            self._drop += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _DROP_TAGS and self._drop:
            self._drop -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._drop:
            self.parts.append(data)


def html_to_text(html: str, max_chars: int = WEB_MAX_CHARS) -> str:
    ex = _TextExtractor()
    try:
        ex.feed(html)
    except Exception:  # noqa: BLE001 — malformed HTML: keep whatever we parsed
        pass
    text = "".join(ex.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_chars]


def _strip_tags(fragment: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


# ----------------------------------------------------------------------
# search providers
# ----------------------------------------------------------------------
def search_duckduckgo(query: str, k: int = 5) -> List[Dict[str, str]]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    raw = http_get(url, max_bytes=800_000).decode("utf-8", errors="replace")

    links = re.findall(r'class="result__a"\s+href="([^"]+)"[^>]*>(.*?)</a>', raw, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', raw, re.S)

    out: List[Dict[str, str]] = []
    for i, (href, title) in enumerate(links[: k * 2]):  # scan extra: ads get filtered out
        # Sponsored ads are Bing/DuckDuckGo click-redirects (ad_domain=..., aclick) —
        # not organic results; refuse them (they 403 and carry no provenance).
        if any(marker in href for marker in ("ad_domain=", "ad_provider=", "bing.com/aclick", "y.js?")):
            continue
        target = href
        if "uddg=" in href:
            m = re.search(r"uddg=([^&]+)", href)
            if m:
                target = urllib.parse.unquote(m.group(1))
        if not target.startswith("http") or "duckduckgo.com" in target:
            continue  # internal redirect, not a real result
        out.append({
            "title": _strip_tags(title) or target,
            "url": target,
            "snippet": _strip_tags(snippets[i]) if i < len(snippets) else "",
            "provider": "duckduckgo",
        })
        if len(out) >= k:
            break
    return out


def search_wikipedia(query: str, k: int = 3) -> List[Dict[str, str]]:
    api = ("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query,
         "srlimit": k, "format": "json", "origin": "*"}))
    raw = http_get(api, max_bytes=400_000).decode("utf-8", errors="replace")
    data = json.loads(raw)
    out: List[Dict[str, str]] = []
    for hit in data.get("query", {}).get("search", []):
        title = hit.get("title", "")
        if not title:
            continue
        out.append({
            "title": title,
            "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
            "snippet": _strip_tags(hit.get("snippet", "")),
            "provider": "wikipedia",
            "date": (hit.get("timestamp") or "")[:10],
        })
    return out


_WIKI_INTRO_CACHE: Dict[str, Optional[str]] = {}


def wikipedia_intro(title: str, max_sentences: int = 2) -> Optional[str]:
    """A short grounded intro (Wikipedia, key-free) for a place — the first up-to-
    `max_sentences` sentences of the article intro. Returns None when offline,
    rate-limited, or no article is found. Cached per process. Provenance: the
    en.wikipedia.org article for `title` (the caller should keep that in mind when
    displaying it). Reuses the SSRF-guarded http_get (429 retry + UA)."""
    t = (title or "").strip()
    if not t or not WEB_ENABLED:
        return None
    key = t.lower()
    if key in _WIKI_INTRO_CACHE:
        return _WIKI_INTRO_CACHE[key]
    out: Optional[str] = None
    try:
        api = ("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": t,
             "srlimit": 3, "format": "json", "origin": "*"}))
        data = json.loads(http_get(api, max_bytes=400_000).decode("utf-8", errors="replace"))
        titles = [h.get("title", "") for h in data.get("query", {}).get("search", [])
                  if h.get("title")][:1]
        if titles:
            api2 = ("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                {"action": "query", "prop": "extracts", "explaintext": 1, "exintro": 1,
                 "redirects": 1, "titles": titles[0], "format": "json", "origin": "*"}))
            data2 = json.loads(http_get(api2, max_bytes=400_000).decode("utf-8", errors="replace"))
            pages = data2.get("query", {}).get("pages", {}) or {}
            page = next(iter(pages.values()), {})
            ex = (page.get("extract") or "").strip()
            if ex:
                sents = re.split(r"(?<=[.!?])\s+", ex)
                out = " ".join(sents[:max_sentences]).strip()
    except (WebError, OSError, ValueError, json.JSONDecodeError):
        out = None
    _WIKI_INTRO_CACHE[key] = out
    return out


def web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """Merged, de-duplicated search across providers. Raises WebError if all fail."""
    results: List[Dict[str, str]] = []
    errors: List[str] = []
    # Wikipedia first in the de-dup pass: the canonical article ('California') wins over
    # the same URL's organic copy ('California - Wikipedia') — otherwise the state
    # article lost to 'California City, California' once (CP 3.1: reliability).
    for provider, provider_k in ((search_wikipedia, min(3, k)), (search_duckduckgo, k)):
        try:
            results.extend(provider(query, provider_k))
        except WebError as exc:
            errors.append(f"{provider.__name__}: {exc}")
    seen = set()
    unique: List[Dict[str, str]] = []
    for r in results:
        key = r["url"].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    if not unique and errors:
        raise WebError("; ".join(errors))
    # Order: structured, reliable sources (Wikipedia) first — the agent prefers to
    # fetch a verifiable source over top organic hits (CP 3.1 reliability) — then
    # organic results for breadth. All stay key-free.
    wiki = [r for r in unique if r.get("provider") == "wikipedia"]
    rest = [r for r in unique if r.get("provider") != "wikipedia"]
    return (wiki + rest)[: max(k, 5)]


# Discovery sources are deliberately searched through their public, indexed pages instead
# of pretending to have private APIs or bypassing login / robots controls.  A result is a
# *lead*, not a factual claim: tools.py keeps the snippet and URL as provenance, and the
# agent labels these items as unverified until a public page is fetched and indexed.
RESEARCH_SOURCES = {
    "reddit": {"domain": "reddit.com", "label": "Reddit", "fetchable": False},
    "instagram": {"domain": "instagram.com", "label": "Instagram", "fetchable": False},
    "tripadvisor": {"domain": "tripadvisor.com", "label": "Tripadvisor", "fetchable": True},
    "expedia": {"domain": "expedia.com", "label": "Expedia", "fetchable": True},
    "wikivoyage": {"domain": "wikivoyage.org", "label": "Wikivoyage", "fetchable": True},
}


def _source_search(source: str, query: str, k: int) -> List[Dict[str, str]]:
    """Find public indexed pages for one configured source."""
    spec = RESEARCH_SOURCES[source]
    hits = search_duckduckgo(f"{query} site:{spec['domain']}", k)
    for hit in hits:
        hit["provider"] = source
        hit["source_type"] = source
        hit["source_label"] = spec["label"]
        hit["fetchable"] = spec["fetchable"]
    return hits


def advanced_search(query: str, k_per_source: int = 3) -> Dict[str, object]:
    """Federated, source-aware discovery for travel research.

    Searches Reddit, Instagram, Tripadvisor, Expedia, Wikivoyage and Wikipedia.
    Calls run concurrently so one slow/blocked site does not stop the research pass.
    Returned records have a stable ``source_type`` for UI/data consumers.
    """
    results: List[Dict[str, str]] = []
    errors: Dict[str, str] = {}

    def wiki() -> List[Dict[str, str]]:
        hits = search_wikipedia(query, k_per_source)
        for hit in hits:
            hit.update(source_type="wikipedia", source_label="Wikipedia", fetchable=True)
        return hits

    jobs = {"wikipedia": wiki}
    jobs.update({name: (lambda name=name: _source_search(name, query, k_per_source))
                 for name in RESEARCH_SOURCES})
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results.extend(future.result())
            except Exception as exc:  # individual source failure is expected on the open web
                errors[name] = str(exc)

    # Keep deterministic source order while removing duplicate URLs.
    order = {"wikipedia": 0, "wikivoyage": 1, "tripadvisor": 2, "expedia": 3,
             "reddit": 4, "instagram": 5}
    unique: List[Dict[str, str]] = []
    seen = set()
    for hit in sorted(results, key=lambda r: (order.get(r.get("source_type", ""), 99), r.get("title", ""))):
        key = hit.get("url", "").rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return {"results": unique, "errors": errors,
            "sources": list(jobs), "query": query}


def geocode_place(query: str) -> Optional[Dict[str, object]]:
    """Key-free two-step geocode: Wikipedia search -> page title -> coordinates.

    `list=geosearch` requires gscoord/gspage/gsbbox (a bare `gsrsearch` fails with
    `missingparam`), so we do the reliable two-step instead:
      1. action=query&list=search  -> best page title for the place
      2. action=query&prop=coordinates&titles=<title> -> lat/lon
    Returns {'lat', 'lon', 'title'} or None (offline, WAYFINDER_WEB=0, or the
    place has no Wikipedia coordinates) — callers degrade to 'no estimate' and
    never invent a position (CP 1.1 honesty)."""
    if not WEB_ENABLED:
        return None
    key = " ".join((query or "").lower().split())
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]
    try:
        geo = _geocode_place_uncached(key)
        # GeoNames recognizes many smaller towns and alternate spellings that do
        # not have an unambiguous English Wikipedia page. It is optional because
        # its free service requires a user-owned username.
        if not geo:
            from .geonames import geocode as geonames_geocode
            geo = geonames_geocode(query)
        _GEOCODE_CACHE[key] = geo  # caches None too: one failed place isn't retried every call
        return geo
    except Exception:  # noqa: BLE001 — geocode must never raise into the agent loop
        return None


def _geocode_place_uncached(query: str) -> Optional[Dict[str, object]]:
    try:
        api = ("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": query,
             "srlimit": 3, "format": "json", "origin": "*"}))
        data = json.loads(http_get(api, max_bytes=400_000).decode("utf-8", errors="replace"))
        titles = [h.get("title", "") for h in data.get("query", {}).get("search", [])]
        titles = [t for t in titles if t][:3]
        if not titles:
            return None
        api2 = ("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "prop": "coordinates", "titles": "|".join(titles),
             "format": "json", "origin": "*"}))
        data2 = json.loads(http_get(api2, max_bytes=400_000).decode("utf-8", errors="replace"))
        pages = data2.get("query", {}).get("pages", {})
        by_title = {p.get("title"): p for p in pages.values()}
        for t in titles:
            coords = (by_title.get(t) or {}).get("coordinates") or []
            if coords:
                c = coords[0]
                lat, lon = float(c.get("lat", 0.0)), float(c.get("lon", 0.0))
                if abs(lat) > 1e-6 or abs(lon) > 1e-6:  # (0,0) = "no coordinates" marker
                    return {"lat": lat, "lon": lon, "title": t}
        return None
    except WebError:
        return None  # offline / blocked -> no geocode, never a fake position


# In-process geocode cache (CP 2.1 'retrieved again with a cache'): hub cities are
# constant across runs, and re-querying Wikipedia on every flight_info call is what
# trips the 429 rate limit. Process-lifetime; the agent is single-process.
_GEOCODE_CACHE: Dict[str, Optional[Dict[str, object]]] = {}


# ----------------------------------------------------------------------
# flight_info live layer — IATA codes + real route data (flightconnections.com)
# ----------------------------------------------------------------------
# Built-in IATA table (works OFFLINE — no network needed). Normalized city
# name (lowercase) -> primary airport code. The Wikipedia 2-step fallback
# (resolve_iata_code) covers places not listed here.
_IATA_TABLE: Dict[str, str] = {
    # USA (major + common destinations)
    "miami": "MIA", "miami beach": "MIA", "fort lauderdale": "FLL",
    "west palm beach": "PBI", "tampa": "TPA", "orlando": "MCO",
    "sarasota": "SRQ", "naples": "APF", "key west": "EYW",
    "atlanta": "ATL", "chicago": "ORD", "dallas": "DFW", "fort worth": "DFW",
    "houston": "IAH", "denver": "DEN", "seattle": "SEA", "san francisco": "SFO",
    "san diego": "SAN", "boston": "BOS", "detroit": "DTW", "charlotte": "CLT",
    "philadelphia": "PHL", "newark": "EWR", "washington": "IAD", "washington dc": "IAD",
    "dc": "IAD", "new orleans": "MSY", "minneapolis": "MSP", "st louis": "STL",
    "phoenix": "PHX", "las vegas": "LAS", "nashville": "BNA", "memphis": "MEM",
    "cincinnati": "CVG", "kansas city": "MCI", "baltimore": "BWI", "tallahassee": "LEF",
    "honolulu": "HNL", "pittsburgh": "PIT", "salt lake city": "SLC", "boise": "BOI",
    "portland": "PDX", "anchorage": "ANC", "albuquerque": "ABQ", "oklahoma city": "OKC",
    "louisville": "SDF", "birmingham": "BHM", "gulfport": "GPT", "little rock": "LIT",
    "milwaukee": "MKE", "des moines": "DSM", "omaha": "OMA", "buffalo": "BUF",
    "hartford": "BDR", "rochester": "ROC", "sacramento": "SMF", "raleigh": "RDU",
    "richmond": "RIC", "burlington": "BTV", "bellingham": "BLI", "spokane": "GEG",
    # New York area -> JFK (primary international gateway)
    "new york": "JFK", "new york city": "JFK", "nyc": "JFK",
    "manhattan": "JFK", "brooklyn": "JFK", "queens": "JFK", "staten island": "JFK",
    "jersey city": "EWR", "new jersey": "EWR",
    # California / West
    "los angeles": "LAX", "la": "LAX", "california": "LAX", "orange county": "SNA",
    "long beach": "LGB", "san jose": "SJC", "palo alto": "SFO",
    # Canada
    "toronto": "YYZ", "montreal": "YUL", "quebec city": "YQB", "vancouver": "YVR",
    "calgary": "YYC", "ottawa": "YOW", "winnipeg": "YWG", "halifax": "YHZ",
    "edmonton": "YEG", "quebec": "YUL", "canada": "YYZ",
    # Mexico / Central America
    "cancun": "CUN", "mexico city": "MEX", "mexico": "MEX", "guadalajara": "GDL",
    "puebla": "PBC", "costa rica": "SJO", "belize": "BZE", "panama city": "PTY",
    # Cayman / Caribbean
    "cayman": "GCM", "cayman islands": "GCM", "cayman island": "GCM",
    "george town": "GCM", "grand cayman": "GCM", "nassau": "FAS",
    "bahamas": "FAS", "san juan": "SJU", "puerto rico": "SJU", "santo domingo": "STI",
    "dominican republic": "STI", "montego bay": "MBJ", "jamaica": "MBJ",
    "barbados": "BGI", "aruba": "AUA", "curacao": "CUR", "antigua": "ANU",
    "st martin": "SFG", "st. martin": "SFG", "st bartelemy": "SBH", "st. barthelemy": "SBH",
    # Europe
    "london": "LHR", "england": "LHR", "uk": "LHR", "united kingdom": "LHR",
    "paris": "CDG", "france": "CDG", "nice": "NCE", "lille": "LIL",
    "berlin": "BER", "frankfurt": "FRA", "munich": "MUC", "munchen": "MUC",
    "cologne": "CGN", "hamburg": "HAM", "dusseldorf": "DUS", "germany": "FRA",
    "rome": "FCO", "italy": "FCO", "milan": "MXP", "florence": "FLR", "venice": "VCE",
    "naples": "NAP", "amsterdam": "AMS", "netherlands": "AMS", "rotterdam": "RTM",
    "madrid": "MAD", "spain": "MAD", "barcelona": "BCN", "seville": "SVQ", "valencia": "VLC",
    "dublin": "DUB", "ireland": "DUB", "cork": "ORK", "galway": "GWY",
    "lisbon": "LIS", "portugal": "LIS", "porto": "OPO", "brussels": "BRU", "belgium": "BRU",
    "vienna": "VIE", "vienna city": "VIE", "austria": "VIE", "salzburg": "SZG",
    "zurich": "ZRH", "switzerland": "ZRH", "geneva": "GVA", "lucerne": "LUC",
    "stockholm": "ARN", "sweden": "ARN", "oslo": "OSL", "norway": "OSL",
    "copenhagen": "CPH", "denmark": "CPH", "helsinki": "HEL", "finland": "HEL",
    "reykjavik": "KEF", "iceland": "KEF", "athens": "ATH", "greece": "ATH",
    "budapest": "BUD", "hungary": "BUD", "prague": "PRG", "czech republic": "PRG",
    "warsaw": "WAW", "poland": "WAW", "london heathrow": "LHR", "london gatwick": "LGW",
    "london city": "LCY", "london stansted": "STN", "luton": "LTN",
    "paris orly": "ORY", "paris charles de gaulle": "CDG",
    "tokyo narita": "NRT", "tokyo haneda": "HND",
    # Middle East
    "dubai": "DXB", "abudhabi": "AUH", "abu dhabi": "AUH", "sharjah": "SHJ",
    "tel aviv": "TLV", "israel": "TLV", "jerusalem": "TLV",
    "amman": "AMM", "jordan": "AMM", "tehran": "IKA", "iran": "IKA",
    "riyadh": "RUH", "dammam": "DMM", "saudi arabia": "RUH", "muscat": "MCT",
    "oman": "MCT", "doha": "DOH", "qatar": "DOH", "bahrain": "BAH",
    "istanbul": "IST", "turkey": "IST", "turkiye": "IST", "izmir": "ADB",
    "cairo": "CAI", "egypt": "CAI", "luxor": "LXR", "ashkelon": "HFA",
    "beirut": "BEY", "lebanon": "BEY", "baghdad": "BGW", "iraq": "BGW",
    # Africa
    "johannesburg": "JNB", "south africa": "JNB", "cape town": "CPT",
    "durban": "DUR", "marrakesh": "RAK", "marrakech": "RAK", "morocco": "RBA",
    "rabat": "RBA", "casablanca": "CAS", "lagos": "LOS", "nigeria": "LOS",
    "accra": "ACC", "ghana": "ACC", "addis ababa": "ADD", "ethiopia": "ADD",
    "nairobi": "NBO", "kenya": "NBO", "dar es salaam": "DAR", "tanzania": "DAR",
    "kigali": "KGL", "rwanda": "KGL", "mombasa": "MBA", "zanzibar": "ZNZ",
    # Asia
    "tokyo": "NRT", "japan": "NRT", "osaka": "KIX", "kyoto": "UKY",
    "singapore": "SIN", "dubai city": "DXB",
    "bangkok": "BKK", "thailand": "BKK", "chiang mai": "CNX", "phuket": "HKT",
    "seoul": "ICN", "south korea": "ICN", "korea": "ICN", "busan": "PUS",
    "shanghai": "PVG", "beijing": "PEK", "china": "PEK", "guangzhou": "CAN",
    "shenzhen": "SZX", "hong kong": "HKG", "taipei": "TPE", "taiwan": "TPE",
    "kuala lumpur": "KUL", "malaysia": "KUL", "penang": "PEN", "langkawi": "LGK",
    "jakarta": "CGK", "indonesia": "CGK", "bali": "DPS", "medan": "KNO",
    "manila": "MNL", "philippines": "MNL", "cebu": "CEB",
    "hanoi": "HAN", "vietnam": "SGN", "ho chi minh city": "SGN", "ho chi minh": "SGN",
    "siem reap": "REU", "cambodia": "REU", "phnom penh": "PNH",
    "mumbai": "BOM", "bombay": "BOM", "delhi": "DEL", "india": "DEL",
    "bengaluru": "BLR", "bangalore": "BLR", "chennai": "MAA", "kolkata": "CCU",
    "hyderabad": "HYD", "goa": "GOI", "kochi": "COK", "thiruvananthapuram": "TRV",
    "kathmandu": "KTM", "nepal": "KTM", "dubai international": "DXB",
    # Oceania
    "sydney": "SYD", "australia": "SYD", "melbourne": "MEL", "brisbane": "BNE",
    "perth": "PER", "adelaide": "ADL", "auckland": "AKL", "new zealand": "AKL",
    "queenstown": "ZQN", "fiji": "NAN", "suva": "NAN",
}

# Airline IATA code -> display name (flightconnections route pages use IATA codes).
_AIRLINE_NAMES: Dict[str, str] = {
    "AA": "American Airlines", "UA": "United Airlines", "AK": "Alaska Airlines", "AS": "Alaska Airlines",
    "B6": "JetBlue", "DL": "Delta", "F9": "Frontier Airlines", "HA": "Hawaiian Airlines",
    "NK": "Spirit Airlines", "WN": "Southwest Airlines", "9E": "Endeavor Air",
    "MQ": "Envoy Air", "NV": "Northwest Airlines", "OH": "PSA Airlines",
    "XE": "ExpressJet", "YX": "SkyWest Airlines", "5X": "Alaska Airlines",
    "AC": "Air Canada", "PD": "Air Canada Express", "WS": "WestJet", "FD": "Porter Airlines",
    "AF": "Air France", "KL": "KLM", "LH": "Lufthansa", "IB": "Iberia",
    "BA": "British Airways", "EI": "Aer Lingus", "FR": "Ryanair", "U2": "easyJet",
    "U6": "Wizz Air", "TP": "TAP Air Portugal", "VY": "Vueling", "LX": "SWISS",
    "OS": "Austrian Airlines", "AY": "Finnair", "FI": "Finnair", "LO": "LOT Polish",
    "SK": "Scandinavian Airlines", "AZ": "Alitalia", "SN": "Brussels Airlines",
    "EK": "Emirates", "EY": "Etihad Airways", "QR": "Qatar Airways", "TK": "Turkish Airlines",
    "ET": "Ethiopian Airlines", "GF": "Royal Air Maroc", "QH": "Kuwait Airways",
    "RQ": "Royal Jordanian", "AM": "El Al", "XY": "flydubai", "FZ": "flydubai",
    "G9": "Air Arabia", "D6": "Air Arabia Abu Dhabi", "GA": "Air Arabia",
    "KM": "Air Kenya", "KE": "Korean Air", "OZ": "Asiana Airlines",
    "JL": "Japan Airlines", "NH": "ANA (All Nippon Airways)", "ZG": "ZIPAIR",
    "CA": "Air China", "CZ": "China Southern", "MU": "China Eastern",
    "3U": "Sichuan Airlines", "HU": "Hainan Airlines", "CX": "Cathay Pacific",
    "CI": "China Airlines", "BR": "China Airlines", "TR": "EVA Air",
    "SQ": "Singapore Airlines", "MH": "Malaysia Airlines", "TG": "Thai Airways",
    "VN": "Vietnam Airlines", "PG": "Philippine Airlines", "3K": "Air India Express",
    "6E": "IndiGo", "AI": "Air India", "QF": "Qantas", "J9": "Jetstar",
    "NZ": "Air New Zealand", "VX": "Virgin Australia",
}


_IATA_CACHE: Dict[str, Optional[str]] = {}
_FLIGHTCONN_CACHE: Dict[str, Optional[Dict[str, object]]] = {}


def resolve_iata_code(place: str) -> Optional[str]:
    """Resolve a city name (or code) to a 3-letter IATA airport code.

    Built-in table first (works offline); a bare 3-letter code passes through;
    otherwise a key-free Wikipedia 2-step (search -> intro extract, IATA regex).
    Results (including None) are cached per process — Wikipedia 429s on bursts,
    and hub cities are constant across runs (CP 2.1 'retrieved again with a cache').
    """
    key = " ".join((place or "").strip().lower().split())
    if not key:
        return None
    if key in _IATA_CACHE:
        return _IATA_CACHE[key]
    code = _resolve_iata_code_uncached(key)
    _IATA_CACHE[key] = code
    return code


def _resolve_iata_code_uncached(key: str) -> Optional[str]:
    if key in _IATA_TABLE:
        return _IATA_TABLE[key]
    if re.fullmatch(r"[a-z]{3}", key):
        return key.upper()  # already a code (e.g. 'mia') — trust it, let the route fetch verify
    # Home-base dataset (data/airports.json — 47 countries, built offline):
    # covers cities the built-in table lacks ('hillingdon' -> LHR, 'boulder city'
    # -> BLD, 'rennes' -> RNS) with ZERO network round-trips. Dataset rows only,
    # hub-ranked — never a guess; wrapped so a missing/corrupt dataset can't break
    # the resolver (falls through to the Wikipedia fallback as before).
    try:
        from . import airports as _airports
        code = _airports.iata_for(key)
        if code:
            return code
    except Exception:
        pass
    if not WEB_ENABLED:
        return None
    # Wikipedia fallback: try the more specific phrasing first (bare "<city> airport"
    # can hit disambiguation pages with no IATA code in the intro).
    for query in (f"{key} international airport", f"{key} airport"):
        try:
            api = ("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                {"action": "query", "list": "search", "srsearch": query,
                 "srlimit": 5, "format": "json", "origin": "*"}))
            data = json.loads(http_get(api, max_bytes=400_000).decode("utf-8", errors="replace"))
            titles = [h.get("title", "") for h in data.get("query", {}).get("search", [])]
            titles = [t for t in titles if t][:5]
            if not titles:
                continue
            api2 = ("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                {"action": "query", "prop": "extracts", "explaintext": 1, "exintro": 1,
                 "redirects": 1, "titles": "|".join(titles),
                 "format": "json", "origin": "*"}))
            data2 = json.loads(http_get(api2, max_bytes=400_000).decode("utf-8", errors="replace"))
            # IMPORTANT: iterate in SEARCH-RELEVANCE order (the `titles` list), not the
            # pages dict order (arbitrary page-id order) — otherwise an irrelevant page
            # with an IATA code can win (e.g. 'Tulsa' resolving to Denver once).
            by_title = {p.get("title"): p for p in (data2.get("query", {}).get("pages", {}) or {}).values()}
            for t in titles:
                page = by_title.get(t) or {}
                m = re.search(r"\bIATA[:\s]*([A-Z]{3})\b", page.get("extract", "") or "")
                if m:
                    return m.group(1)
        except WebError:
            continue  # offline / rate-limited -> next phrasing, then give up
    return None


def _parse_duration_h(text: str) -> Optional[float]:
    """'16h 17' / '16h17' / '8h' -> hours as float (flightconnections duration format)."""
    m = re.match(r"^\s*(\d{1,2})h\s?(\d{2})?\s*$", text or "")
    if not m:
        return None
    hours = int(m.group(1))
    minutes = int(m.group(2)) if m.group(2) else 0
    return hours + minutes / 60.0


def fetch_route_schedule(origin_code: str, dest_code: str) -> Optional[Dict[str, object]]:
    """Best-effort key-free route lookup on flightconnections.com (REAL schedule data).

    Verified working (2025): static HTML route pages with direct-flight
    availability, per-stopover real durations, operating airlines and route
    distance. Returns a small dict (direct, airlines, routes, fastest_direct_h,
    distance_km, notification, url, fetched_at) or None when the pair is
    unknown (404) or the page is blocked — the caller then degrades to the
    labeled distance-model estimate (CP 1.1: never a silent gap, never fake data).
    """
    o = (origin_code or "").strip().upper()
    d = (dest_code or "").strip().upper()
    if not (re.fullmatch(r"[A-Z]{3}", o) and re.fullmatch(r"[A-Z]{3}", d)):
        return None
    key = f"{o}-{d}"
    if key in _FLIGHTCONN_CACHE:
        return _FLIGHTCONN_CACHE[key]
    url = f"https://www.flightconnections.com/flights-from-{o.lower()}-to-{d.lower()}"
    try:
        raw = http_get(url, max_bytes=1_400_000).decode("utf-8", errors="replace")
    except WebError:
        _FLIGHTCONN_CACHE[key] = None
        return None
    sched = _parse_route_page(raw)
    if not sched:
        _FLIGHTCONN_CACHE[key] = None
        return None
    sched["url"] = url
    sched["fetched_at"] = time.strftime("%Y-%m-%d")
    _FLIGHTCONN_CACHE[key] = sched
    return sched


def _parse_route_page(html: str) -> Optional[Dict[str, object]]:
    """Extract the schedule facts we use from a flightconnections route page.

    Two verified page shapes:
      * direct:  <h1>Direct flights from X to Y</h1> + 'fastest direct flight
                  ... takes 8 hours and 55 minutes' + airlines <ul>
      * stops:   'no direct flights from X to Y' + 'N routes with 1 stop found'
                  + <li class=flight-path-via> items carrying data-stops /
                  data-connections / data-airlines (IATA codes) + real durations
    """
    out: Dict[str, object] = {
        "direct": False, "notification": None, "airlines": [], "routes": [],
        "fastest_direct_h": None, "distance_km": None,
    }
    # Route distance: '7,432 miles (11,961 kilometers)' / '4,415 miles (or 7,105 km)'
    m = re.search(r"distance between [^.<]{3,60}? is [\d,]+ miles \((?:or\s+)?([\d,]+) ?(?:km|kilometers?)\)", html, re.I)
    if m:
        out["distance_km"] = int(m.group(1).replace(",", ""))
    if re.search(r"<h1[^>]*>\s*Direct flights from", html, re.I):
        out["direct"] = True
        m = re.search(r"fastest direct flight from [^.]*?takes\s+(\d+)\s+hours?\s+and\s+(\d+)\s+minutes?", html, re.I)
        if m:
            out["fastest_direct_h"] = int(m.group(1)) + int(m.group(2)) / 60.0
        else:
            m = re.search(r"fastest direct flight from [^.]*?takes\s+(\d+(?:\.\d+)?)\s+hours?", html, re.I)
            if m:
                out["fastest_direct_h"] = float(m.group(1))
        block = re.search(r'<ul class="route-page-info-text airlines">(.*?)</ul>', html, re.S)
        if block:
            names = re.findall(r'airlines_sq/[^\"]*"[^>]*title="([^"]+)"', block.group(1))
            if not names:
                names = re.findall(r"</div>\s*([^<]{3,40})</li>", block.group(1))
            seen: set = set()
            for n in names:
                n = n.strip()
                if n and n.lower() not in seen:
                    seen.add(n.lower())
                    out["airlines"].append(n)
    else:
        m = re.search(r"route-notification-label\">\s*([^<]+)", html)
        if m:
            out["notification"] = m.group(1).strip()
        for li in re.findall(r'<li class="flight-path-via btn"(.*?)</li>', html, re.S):
            stops_m = re.search(r'data-stops="(\d+)"', li)
            dur_m = re.search(r'via-duration\">.*?(\d{1,2}h\s?\d{0,2})\s*<', li, re.S)
            if not (stops_m and dur_m):
                continue
            conns_m = re.search(r'data-connections="([^"]*)"', li)
            al_m = re.search(r'data-airlines="([^"]*)"', li)
            via_m = re.search(r'via-destination\">\s*([^<]+)<', li)
            codes = [c.strip().upper() for c in (al_m.group(1).split(",") if al_m else []) if c.strip()]
            out["routes"].append({
                "stops": int(stops_m.group(1)),
                "via": via_m.group(1).strip() if via_m else None,
                "connections": conns_m.group(1).strip() if conns_m else "",
                "airline_codes": codes,
                "duration_text": dur_m.group(1).replace(" ", ""),
                "duration_h": round(_parse_duration_h(dur_m.group(1)) or 0.0, 1),
            })
    if not (out["direct"] or out["routes"] or out["airlines"]):
        return None
    return out


def airline_display(code: str) -> str:
    """IATA airline code -> friendly name (unknown codes shown as-is)."""
    return _AIRLINE_NAMES.get((code or "").strip().upper(), (code or "").strip().upper())


def fetch_page(url: str) -> Dict[str, object]:
    """Download + extract a page. Returns {title, text, chars, fetched_at}."""
    raw = http_get(url)
    html = raw.decode("utf-8", errors="replace")
    text = html_to_text(html)
    if len(text) < 200:
        raise WebError("page contained too little readable text (JS-heavy or blocked?)")
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    title = _strip_tags(m.group(1)) if m else url
    return {
        "title": title or url,
        "text": text,
        "chars": len(text),
        "fetched_at": time.strftime("%Y-%m-%d"),
    }
