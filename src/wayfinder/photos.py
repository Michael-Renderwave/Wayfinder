"""Key-free photo lookup for the answer (landmarks + dining/food scenes).

Source: the **Wikimedia Commons** search API — CC-licensed media with a caption and
artist attribution (provenance, CP 3.1). Zero API keys: the Commons `action=query`
search endpoint is public. Reuses ``web.http_get`` (SSRF-guarded, 429 retry, UA set).

Honesty model: any failure (offline, no match, rate-limit, image off) raises
``PhotoError``; the caller OMITS the photo rather than fabricating one. The agent
never invents an image or a caption.

  photos_for_landmark(name, dest) -> [{title,url,thumb_url,caption,artist,license}]
  photos_for_scene(scene, dest)   -> [...],  scene in {"dining","beaches","museums",
                              "nature","shopping","stays","nightlife","luxury"}
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
from typing import Dict, List, Optional

from .web import WEB_ENABLED, WebError, http_get
from .world import (_country_capitals, ascii_norm, city_canonicals, home_countries,
                    known_city_countries)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_IMG_WIDTH = int(os.environ.get("WAYFINDER_IMG_WIDTH", "900"))
_MIN_SIDE = 150  # skip icons / tiny fragments (original image side, px)


class PhotoError(WebError):
    pass


# ---------------------------------------------------------------------------
# foreign-city filter (CP 3.1 bias guard: anchor photos to the destination)
# ---------------------------------------------------------------------------
# A photo is a foreign-city LEAK when it names a dataset CITY whose country is not
# the destination's home country AND it names no local city — e.g. a
# "Brazilian food, Quebec city" file in a Brazil answer (Quebec City is a Canadian
# city, no Brazilian city named) is dropped. A genuine shared-border photo that
# names a local city ("Iguazu Falls, Foz do Iguacu, Brazil") is kept, and an
# unlocated dish photo ("Feijoada") is kept. Gated on home_countries(dest): when
# the destination's home country can't be determined (curated Kennywood / Cayman /
# George Town) we can't judge provenance, so NOTHING is dropped — the demo
# destinations keep working exactly as before.
_CITY_RE_CACHE: Dict[frozenset, tuple] = {}


def _city_regexes(allowed: set):
    """(foreign-city regex, local-city regex) for the destination's home country.
    Cached per allowed-set. Either may be None (no such cities to look for)."""
    key = frozenset(allowed)
    cached = _CITY_RE_CACHE.get(key)
    if cached is not None:
        return cached
    city_country = known_city_countries()

    def compile(names: List[str]):
        if not names:
            return None
        names = sorted(names, key=len, reverse=True)  # longer names first
        return re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in names) + r")(?!\w)")

    foreign = compile([n for n, c in city_country.items() if c not in key])
    local = compile([n for n, c in city_country.items() if c in key])
    _CITY_RE_CACHE[key] = (foreign, local)
    return foreign, local


def _filter_home_place(images: List[dict], dest: str) -> List[dict]:
    """Drop images that name a foreign city (a dataset city whose country is not the
    destination's home country) with no local city present. Keeps images that name
    only local cities, or no city at all. When the destination's home country is
    unknown, returns ``images`` unchanged (honesty: we'd rather show a plausible
    local photo than starve the block, and we never regress the demo destinations).
    """
    allowed = home_countries(dest)
    if not allowed:
        return images
    foreign_re, local_re = _city_regexes(allowed)
    if foreign_re is None:
        return images
    kept: List[dict] = []
    for im in images:
        text = ascii_norm(f"{im.get('title', '')} {im.get('caption', '')}").lower()
        if foreign_re.search(text) and not (local_re is not None and local_re.search(text)):
            continue  # names a foreign city, no local city -> leak -> drop
        kept.append(im)
    return kept


def _clean(value: str) -> str:
    """Strip HTML/wikimedia markup and collapse whitespace from a Commons metadata
    value (captions/artist often contain <a> links, <p>, &amp;, [[File:...|cap]])."""
    if not value:
        return ""
    v = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", value)   # [[File:x|Caption]] -> Caption
    v = re.sub(r"<[^>]+>", " ", v)                              # drop tags
    v = re.sub(r"&amp;", "&", v)
    v = re.sub(r"&lt;", "<", v)
    v = re.sub(r"&gt;", ">", v)
    v = re.sub(r"&#0*39;|&apos;|&rsquo;|&lsquo;", "'", v)
    v = re.sub(r"&quot;", '"', v)
    v = re.sub(r"&nbsp;|\s+", " ", v).strip()
    return v


def _commons_search(query: str, limit: int = 6) -> List[dict]:
    """Search the Commons file namespace for bitmap images matching ``query``.

    Returns a list of {title, url, thumb_url, caption, artist, license} in search
    relevance order. Raises PhotoError when the web is off, the call fails, or no
    usable image is found (callers degrade by omitting the photo)."""
    if not WEB_ENABLED:
        raise PhotoError("web research is off in this deployment")
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",          # File: namespace
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": str(_IMG_WIDTH),  # request a scaled thumbnail (<= this width)
    }
    url = COMMONS_API + "?" + urllib.parse.urlencode(params)
    data = json.loads(http_get(url).decode("utf-8"))
    pages = (data.get("query") or {}).get("pages") or {}

    out: List[dict] = []
    for page in pages.values():
        ii = (page.get("imageinfo") or [{}])[0]
        thumb = ii.get("thumburl")
        full = ii.get("url")
        if not (thumb or full):
            continue
        # skip icons / fragments: original image must be reasonably large
        w = int(ii.get("width") or 0)
        h = int(ii.get("height") or 0)
        if w and h and min(w, h) < _MIN_SIDE:
            continue
        em = ii.get("extmetadata") or {}
        meta = lambda k: _clean((em.get(k) or {}).get("value") or "")  # noqa: E731
        caption = meta("ImageDescription") or meta("ObjectName") or meta("Artist")
        out.append({
            "title": (page.get("title") or "").replace("File:", "", 1).replace("_", " "),
            "url": full,
            "thumb_url": thumb or full,
            "caption": caption[:220],
            "artist": meta("Artist")[:120],
            "license": meta("LicenseShortName")[:60],
        })
    if not out:
        raise PhotoError(f"no Commons image found for {query!r}")
    return out


def photos_for_landmark(name: str, dest: str = "") -> List[dict]:
    """Photos of a specific landmark (e.g. 'Kennywood', 'George Town').

    Honesty rule: when the name and the destination don't contain each other
    (a *generic* local name, e.g. 'Church Arcade' mapped inside Kennywood), ONLY
    the disambiguated `name + destination` search is trusted — a bare search would
    match unrelated same-named places elsewhere (a 'Church Arcade' in the Czech
    Republic is not Kennywood's). A bare search is used only when the name already
    names the destination. No-match raises PhotoError; the caller omits the photo
    rather than showing a misattributed one."""
    name = (name or "").strip()
    if not name:
        raise PhotoError("no landmark name given")
    if dest and dest.lower() not in name.lower() and name.lower() not in dest.lower():
        imgs = _commons_search(f"{name} {dest}", limit=8)
    else:
        imgs = _commons_search(name, limit=8)
    imgs = _filter_home_place(imgs, dest)
    if not imgs:
        raise PhotoError(f"no on-destination image for {name!r}")
    return imgs[:4]


# scene -> (Commons query with {dest}, fallback when the destination is unknown).
# Phrasing is tuned for Commons file-title reality: 'Paris food' has thousands of
# hits, 'Paris luxury hotel' has hundreds, exotic phrasing gets zero.
SCENE_QUERIES = {
    "dining":    ("{dest} food", "travel food dining"),
    "beaches":   ("{dest} beach", "tropical beach sea"),
    "museums":   ("{dest} museum", "museum interior"),
    "nature":    ("{dest} landscape nature", "mountain landscape"),
    "shopping":  ("{dest} shopping street market", "shopping street"),
    "stays":     ("{dest} hotel", "hotel exterior"),
    "nightlife": ("{dest} nightlife", "city night skyline"),
    "luxury":    ("{dest} luxury hotel", "luxury hotel resort"),
    "festivals": ("{dest} festival", "festival parade"),
    "wildlife":  ("{dest} wildlife animal", "wildlife animal"),
    "history":   ("{dest} historic", "historic building"),
    "art":       ("{dest} art", "street art"),
    "night views": ("{dest} night skyline", "city night skyline"),
}

# scene -> keywords used to RANK the pool (relevance guard): Commons' search order
# is loose, so a 'Greece beach' pool can surface boats, turtles and townscapes
# before actual beaches. Images whose title/caption name the scene beat generic
# ones ('Punaluu Beach' > 'sea turtle on a shore' > 'island from a boat').
# Zero-keyword images keep Commons' order among themselves (stable sort).
SCENE_KEYWORDS = {
    "dining":    ("food", "restaurant", "dish", "meal", "cafe", "coffee", "kitchen",
                  "bake", "bread", "pastry", "pasta", "pizza", "sushi", "dessert",
                  "dining", "eatery", "taverna", "bistro", "cuisine", "wine", "winery",
                  "vineyard", "grape"),
    "beaches":   ("beach", "coast", "bay", "cove", "shore", "sea", "reef", "island",
                  "sand", "surf", "snorkel", "swim", "lagoon", "harbor", "harbour"),
    "museums":   ("museum", "gallery", "exhibition", "artifact", "sculpture",
                  "painting", "mural", "artwork"),
    "nature":    ("landscape", "mountain", "valley", "waterfall", "lake", "forest",
                  "jungle", "canyon", "cliff", "hiking", "trail", "wildlife",
                  "nature", "scenery", "vista", "desert", "dune", "dunes", "snow",
                  "skiing", "glacier"),
    "shopping":  ("shop", "shopping", "market", "bazaar", "store", "boutique",
                  "souvenir", "mall", "stall", "shopfront"),
    "stays":     ("hotel", "resort", "inn", "motel", "guesthouse", "villa", "lobby",
                  "suite", "accommodation", "hostel"),
    "nightlife": ("nightlife", "club", "bar", "cocktail", "neon", "night", "pub",
                  "lounge"),
    "luxury":    ("luxury", "resort", "hotel", "penthouse", "spa", "pool", "yacht",
                  "suite", "upscale"),
    "festivals": ("festival", "carnival", "parade", "celebration", "lantern",
                  "firework", "fireworks", "cultural", "procession"),
    "wildlife":  ("wildlife", "animal", "animals", "bird", "birds", "whale", "dolphin",
                  "elephant", "lion", "tiger", "giraffe", "monkey", "penguin", "seal",
                  "shark", "safari", "zoo", "aquarium", "flamingo", "gorilla", "zebra"),
    "history":   ("historic", "historical", "ancient", "old town", "ruins", "ruin",
                  "castle", "monument", "temple", "church", "cathedral", "fortress",
                  "palace", "heritage", "amphitheatre", "colosseum", "acropolis"),
    "art":       ("art", "gallery", "galleries", "painting", "paintings", "sculpture",
                  "sculptures", "mural", "murals", "exhibition", "street art"),
    "night views": ("night", "skyline", "neon", "illuminated", "lights", "evening",
                    "dusk", "cityscape"),
}

_US_STATE_RE: Optional[re.Pattern] = None
# US 2-letter state codes, matched ONLY in a 'town, ST' pattern (a leading comma)
# so common words that share a code (in, or, me) can never false-match. The
# common-word codes (IN/OR/ME/ID) are dropped entirely for the same reason. This
# catches the real leak shapes seen on Commons: 'Greece, NY', 'Monticello, AR'.
_US_STATE_ABBR_RE = re.compile(
    r",\s*(?:al|ak|az|ar|ca|co|ct|de|dc|fl|ga|hi|il|ia|ks|ky|la|md|ma|mi|mn"
    r"|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|pa|ri|sc|sd|tn|tx|ut|vt|va|wa"
    r"|wv|wi|wy)\b")
# street/building-type nouns that, sitting next to the destination word, mark a
# LOCAL place NAMED after the destination rather than the destination itself — a
# Hong-Kong '佐敦道 / Jordan Road' district or a 'Jordan Mansion' building, not the
# Kingdom of Jordan. Deliberately EXCLUDES natural features and common descriptors
# (valley, river, beach, villa, garden, building, court, drive, way) so a genuine
# destination photo is never mistaken for a namesake.
_PLACE_SUFFIX = ("road|street|avenue|boulevard|plaza|square|mansion|junction|lane")
_SUFFIX_RE_CACHE: Dict[str, Optional[re.Pattern]] = {}


def _namesake_suffix_re(dest: str) -> Optional[re.Pattern]:
    """Regex matching the destination word adjacent to a street/building-type noun
    (either side), cached per destination. The tell for a namesake: the
    destination name used as a local street or building name, not the place."""
    d = ascii_norm(dest).strip().lower()
    if not d:
        return None
    cached = _SUFFIX_RE_CACHE.get(d)
    if cached is None:
        cached = re.compile(
            rf"(?<!\w){re.escape(d)}\s+(?:{_PLACE_SUFFIX})\b"
            rf"|(?<!\w)(?:{_PLACE_SUFFIX})\s+{re.escape(d)}\b")
        _SUFFIX_RE_CACHE[d] = cached
    return cached


def _names_us_state(text: str) -> bool:
    """True when ``text`` names a US state — full name anywhere, or a 2-letter
    USPS code in a 'town, ST' pattern (', NY', ', AR'). Used only as a RANKING
    dampener for non-US destinations — never a hard filter, since several state
    names are also real place names elsewhere (Maine, FR; Nevada, ES; Oregon, IT)."""
    global _US_STATE_RE
    if _US_STATE_RE is None:
        from .world import US_STATE_CAPITALS
        states = sorted((s for s in US_STATE_CAPITALS if s != "united states"),
                        key=len, reverse=True)
        _US_STATE_RE = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(s) for s in states) + r")(?!\w)")
    if _US_STATE_RE.search(text):
        return True
    return bool(_US_STATE_ABBR_RE.search(text))


def _names_namesake(text: str, dest: str) -> bool:
    """True when ``text`` looks like a SAME-NAMED but DIFFERENT place (a namesake
    leak), not the destination itself: a US 'Jordan'/'Greece, NY' town, or the
    destination word used as a local street/building ('佐敦道 Jordan Road'). Only
    consulted for non-US destinations and only as a ranking dampener — sort-only,
    never a drop (CP 1.1 no-starvation)."""
    if _names_us_state(text):
        return True
    pat = _namesake_suffix_re(dest)
    return bool(pat and pat.search(text))


def _dedupe(images: List[dict]) -> List[dict]:
    """Drop literal duplicates of the same file (Commons sometimes returns one
    place under several near-identical uploads — e.g. the '佐敦道 Jordan Road'
    restaurant 7×). First-seen order is kept; DISTINCT files are never dropped."""
    seen: set = set()
    out: List[dict] = []
    for im in images:
        key = im.get("url") or im.get("thumb_url") or im.get("title")
        if key in seen:
            continue
        seen.add(key)
        out.append(im)
    return out


# country (ASCII-normalized, lowercase) -> national adjective / demonym. Commons
# captions often say 'Jordanian food' / 'Greek cuisine' / 'French bistro' — the
# adjective is the positive tell that a photo is about the COUNTRY itself rather
# than a same-named US town or a person named 'Jordan'. Covers the major
# destinations; the home-city check (dataset cities) covers the rest.
_COUNTRY_ADJ = {
    "jordan": "jordanian", "greece": "greek", "france": "french",
    "italy": "italian", "spain": "spanish", "turkey": "turkish",
    "türkiye": "turkish", "egypt": "egyptian", "india": "indian",
    "china": "chinese", "japan": "japanese", "germany": "german",
    "portugal": "portuguese", "mexico": "mexican", "brazil": "brazilian",
    "thailand": "thai", "vietnam": "vietnamese", "morocco": "moroccan",
    "kenya": "kenyan", "australia": "australian", "switzerland": "swiss",
    "austria": "austrian", "ireland": "irish", "norway": "norwegian",
    "sweden": "swedish", "denmark": "danish", "finland": "finnish",
    "poland": "polish", "hungary": "hungarian", "russia": "russian",
    "ukraine": "ukrainian", "georgia": "georgian", "lebanon": "lebanese",
    "israel": "israeli", "syria": "syrian", "iran": "iranian",
    "iraq": "iraqi", "saudi arabia": "saudi", "nepal": "nepali",
    "indonesia": "indonesian", "malaysia": "malaysian", "korea": "korean",
    "cuba": "cuban", "chile": "chilean", "peru": "peruvian",
    "colombia": "colombian", "nigeria": "nigerian", "ghana": "ghanaian",
    "tanzania": "tanzanian", "argentina": "argentine",
    "netherlands": "dutch", "belgium": "belgian", "czech": "czech",
    "czechia": "czech", "singapore": "singapore", "myanmar": "myanma",
    "burma": "burmese",
}


def _home_anchored(text: str, dest: str) -> bool:
    """True when ``text`` references the destination as a COUNTRY/REGION — names
    a dataset city in its home country, or its national adjective ('Jordanian',
    'Greek', 'French'). The positive tell that a photo is about the destination
    itself rather than a same-named place elsewhere. ``text`` should be
    lowercase/ascii-normalized (see ``_text``)."""
    d = ascii_norm(dest).strip().lower()
    hc = home_countries(dest)
    if hc:
        for city, cy in known_city_countries().items():
            if cy in hc and city in text:
                return True
    adj = _COUNTRY_ADJ.get(d)
    if adj and adj in text:
        return True
    return False


def photos_for_scene(scene: str, dest: str = "") -> List[dict]:
    """Photos of a scene at the destination — dining/food, beaches, museums,
    stays, luxury, and more. 'dining'/'food'/'restaurant'/'eat' map to a
    `{destination} food` search; the known scenes above use their own tuned
    queries; anything else searches `{destination} {scene}`. Returns
    on-destination images only (foreign-city leak filter, CP 3.1)."""
    scene = (scene or "").strip().lower()
    dest = (dest or "").strip()
    if scene in ("food", "restaurant", "eat"):
        scene = "dining"
    q, fallback = SCENE_QUERIES.get(
        scene, (f"{dest} {scene}".strip() or "travel scene", "travel scene"))
    q = q.format(dest=dest) if dest else fallback
    # Request a wider pool (deduped), then drop any foreign-place leak before
    # taking the best on-destination images (so the scene group isn't starved to
    # nothing once the off-place hits are filtered out).
    raw = _dedupe(_commons_search(q, limit=12))
    allowed = home_countries(dest)
    us_foreign = bool(allowed) and "united states" not in allowed

    def _text(im: dict) -> str:
        return ascii_norm(f"{im.get('title', '')} {im.get('caption', '')}").lower()

    def _name_rate(pool: List[dict]) -> float:
        if not pool:
            return 1.0
        return sum(1 for im in pool if _names_namesake(_text(im), dest)) / len(pool)

    imgs = _filter_home_place(raw, dest)

    # Name-collision rescue: a COUNTRY destination (not a city) whose pool is
    # MORE THAN HALF namesakes — a same-named but different place ('Jordan' ->
    # the Hong-Kong 'Jordan Road' district or US 'Jordan' towns; 'Greece' ->
    # 'Greece, NY') — matched a namesake, not the country: retry anchored on its
    # capital ('Jordan dining' -> 'Amman dining'). Only when that pool is
    # genuinely cleaner (never make a good pool worse). Cities ('New York') and
    # US destinations skip this, as do destinations with no capital in the set.
    # Only rescue when the pool is namesake-dominated AND has NO home-anchored
    # photo (a photo naming a home country-city or its adjective, e.g. 'Jordanian'
    # / 'Athens' / 'Greek'): if a genuine on-destination photo is present, the
    # primary sort already surfaces it, so swapping to the capital would only
    # risk a worse result. This is what keeps 'Jordan food' on the real Jordanian
    # dishes while still rescuing 'Jordan historic' (all undetectable US/person
    # namesakes) to 'Amman historic'.
    if (us_foreign and dest and dest not in city_canonicals() and len(raw) >= 4
            and _name_rate(imgs) > 0.5
            and not any(not _names_namesake(_text(im), dest) and _home_anchored(_text(im), dest)
                        for im in imgs)):
        cap = (_country_capitals().get(ascii_norm(dest).strip().lower()) or "").strip()
        if cap and cap.lower() != dest.lower():
            try:
                q2 = q.replace(dest, cap, 1)
                raw2 = _dedupe(_commons_search(q2, limit=12))
                imgs2 = _filter_home_place(raw2, dest)
                if imgs2 and _name_rate(imgs2) < _name_rate(imgs):
                    q, raw, imgs = q2, raw2, imgs2
            except PhotoError:
                pass  # capital pool unavailable: keep the original; ranking picks best

    if not imgs:
        raise PhotoError(f"no on-destination image for {q!r}")
    # Relevance guard (sort only, never a drop): 1) a name-collision dampener —
    # for a destination OUTSIDE the United States, an image that names a
    # same-named different place (a US 'Jordan'/'Greece, NY' town, or the
    # destination word as a local street/building) ranks BELOW otherwise-equal
    # images; 2) among equals, images that NAME the scene beat generic ones
    # (stable sort keeps Commons order on ties). The collision flag is the
    # PRIMARY key, so a namesake can't outrank the real destination merely by
    # carrying more scene-keywords — '佐敦道 Jordan Road … Restaurant' has
    # 'food'+'restaurant' but is a Hong-Kong place, not the Kingdom.
    kws = SCENE_KEYWORDS.get(scene)

    def _rank(im):
        text = _text(im)
        kw = sum(text.count(k) for k in kws) if kws else 0
        collision = 1 if (us_foreign and _names_namesake(text, dest)) else 0
        return (collision, -kw)

    if kws or us_foreign:
        imgs.sort(key=_rank)
    return imgs[:4]
