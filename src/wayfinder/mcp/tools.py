"""MCP tool surface for Wayfinder.

Mirrors the agent's own capabilities (CP 1.1 tool calling + CP 2.1 ReAct):
a flagship `plan_trip` that runs the full ReAct loop, plus each individual
tool so a client (e.g. Claude Code) can drive them directly.

Honesty model (CP 1.1) applies everywhere: sample data is labeled, fares are
estimates, empty buckets are said plainly, web results are unverified leads.
All sources are key-free.
"""

from typing import Optional, Union

from mcp.server import MCPServer

from wayfinder.mcp.types import ToolSuccess, ToolError
from wayfinder.mcp.utils import tool_error, tool_success

# ---------------------------------------------------------------------------
# shared agent (lazy: built on first tool call, so server start stays instant)
# ---------------------------------------------------------------------------
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from wayfinder.app import get_agent

        _agent = get_agent()
    return _agent


def _obs_payload(result: dict, obs: str) -> dict:
    """(result_dict, observation_text) -> MCP data dict with both."""
    return {"result": result, "observation": obs}


def _run_plan_trip(query: str, home_country: str, home_region: str) -> dict:
    """Drive the full ReAct loop and collect a structured briefing."""
    agent = _get_agent()
    home = None
    if home_country:
        home = {"country": home_country}
        if home_region:
            home["region"] = home_region

    answer: dict = {}
    done: dict = {}
    trace: list = []
    for ev in agent.run(query, home=home):
        t = ev.get("type")
        if t == "thought":
            trace.append(f"thought: {ev.get('text', '').strip()}")
        elif t == "action":
            args = ", ".join(f"{k}={v}" for k, v in (ev.get("input") or {}).items())
            trace.append(f"act: {ev.get('tool', '')}({args})")
        elif t == "observation":
            trace.append(f"observe: {ev.get('observation', '').strip()}")
        elif t == "prune":
            pass  # beam-pruned candidates stay in the stats, not the trace
        elif t == "guardrail":
            trace.append(f"guardrail: {str(ev.get('checks', ev.get('note', '')))[:300]}")
        elif t == "answer":
            answer = ev
        elif t == "done":
            done = ev

    sections = answer.get("sections") or []
    briefing = (answer.get("intro") or "").strip()
    if sections:
        briefing = (briefing + "\n\n" if briefing else "") + "\n\n".join(
            s.strip() for s in sections
        )

    return {
        "briefing": briefing,
        "sources": [
            {"tier": s.get("tier"), "title": s.get("title"), "url": s.get("url")}
            for s in (answer.get("sources") or [])
        ],
        "confidence_pct": answer.get("confidence_pct"),
        "confidence_label": answer.get("confidence_label"),
        "confidence_detail": answer.get("confidence_detail"),
        "trace": trace,
        "stats": {
            "steps": done.get("steps"),
            "searches": done.get("searches"),
            "pruned": done.get("pruned"),
            "elapsed_ms": done.get("elapsed_ms"),
            "mode": done.get("mode"),
        },
    }


def register_wayfinder(app: MCPServer):
    from wayfinder.tools import (
        advanced_search_tool,
        best_time,
        book_ticket,
        local_places_tool,
        luxury_experiences_tool,
        search_flights,
        search_hotels,
        search_kb,
        travel_safety_tool,
        web_search_tool,
    )

    @app.tool(
        name="plan_trip",
        title="Plan a Trip",
        description=(
            "Run the full Wayfinder research loop for a trip: ReAct planning over a cited "
            "knowledge base plus key-free live tools (flights, hotels, real local places, "
            "official safety rating, web research). Returns a grounded briefing with "
            "sources, a confidence band, and the reasoning trace. Slowest tool (tens of "
            "seconds) but the most complete answer."
        ),
        structured_output=True,
    )
    def plan_trip(
        query: str,
        home_country: str = "",
        home_region: str = "",
    ) -> Union[ToolSuccess[dict], ToolError]:
        try:
            return tool_success(_run_plan_trip(query, home_country, home_region))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")

    @app.tool(
        name="search_flights",
        title="Search Flights",
        description=(
            "SearchFlight() against the OAG-style schedule (sample data — fares are "
            "labeled estimates). Filter by origin/destination/date/max_price."
        ),
        structured_output=True,
    )
    def search_flights_tool(
        origin: str = "",
        destination: str = "",
        date: str = "",
        max_price: Optional[int] = None,
    ) -> Union[ToolSuccess[dict], ToolError]:
        try:
            agent = _get_agent()
            result, obs = search_flights(
                agent.flights_db,
                origin=origin,
                destination=destination,
                date=date,
                max_price=max_price,
            )
            return tool_success(_obs_payload(result, obs))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")

    @app.tool(
        name="search_hotels",
        title="Search Hotels",
        description=(
            "Stay options near a destination, ranked by proximity to the airport and "
            "main destinations (sample data, labeled). Optional max_price cap and "
            "'near' anchor city."
        ),
        structured_output=True,
    )
    def search_hotels_tool(
        destination: str = "",
        max_price: Optional[int] = None,
        near: str = "",
    ) -> Union[ToolSuccess[dict], ToolError]:
        try:
            agent = _get_agent()
            result, obs = search_hotels(
                agent.hotels_db, destination=destination, max_price=max_price, near=near
            )
            return tool_success(_obs_payload(result, obs))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")

    @app.tool(
        name="search_kb",
        title="Knowledge Base Search",
        description=(
            "Hard RAG search over Wayfinder's cited knowledge base (400+ docs, tiered "
            "primary/secondary/tertiary + recency scoring, beam-pruned with reasons)."
        ),
        structured_output=True,
    )
    def search_kb_tool(query: str, k: int = 8) -> Union[ToolSuccess[dict], ToolError]:
        try:
            agent = _get_agent()
            result, obs = search_kb(agent.store, query, k=k)
            return tool_success(_obs_payload(result, obs))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")

    @app.tool(
        name="best_time",
        title="Best Time to Visit",
        description=(
            "Seasonal verdict for a destination (Open-Meteo climate normals + "
            "Wikipedia holidays + Wikivoyage climate notes, key-free). Optional "
            "specific month (1-12)."
        ),
        structured_output=True,
    )
    def best_time_tool(destination: str, month: Optional[int] = None) -> Union[ToolSuccess[dict], ToolError]:
        try:
            result, obs = best_time(destination, month=month)
            return tool_success(_obs_payload(result, obs))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")

    @app.tool(
        name="travel_safety",
        title="Travel Safety",
        description=(
            "Official travel-safety rating (green->yellow->red meter) from the U.S. "
            "Department of State advisory — the primary live signal is the official "
            "travel.state.gov RSS feed, cross-checked against an official snapshot. "
            "Safety-first reconciliation when sources disagree."
        ),
        structured_output=True,
    )
    def travel_safety_tool_fn(place: str) -> Union[ToolSuccess[dict], ToolError]:
        try:
            agent = _get_agent()
            result, obs = travel_safety_tool(place, data_dir=agent.data_dir)
            return tool_success(_obs_payload(result, obs))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")

    @app.tool(
        name="local_places",
        title="Real Local Places",
        description=(
            "REAL named places near a destination from OpenStreetMap (key-free "
            "Nominatim + Overpass): dining, attractions, hotels, or "
            "pharmacies/hospitals/police when asked about safety services. Never "
            "invented — if OSM has no match it says so."
        ),
        structured_output=True,
    )
    def local_places_tool_fn(
        place: str,
        focus: str = "all",
        k: int = 12,
    ) -> Union[ToolSuccess[dict], ToolError]:
        try:
            agent = _get_agent()
            result, obs = local_places_tool(
                place, focus=focus, k=k, data_dir=agent.data_dir
            )
            return tool_success(_obs_payload(result, obs))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")

    @app.tool(
        name="luxury_experiences",
        title="Luxury Experiences",
        description=(
            "Higher-end layer for a destination: 4-5 star stays (OSM star_rating), "
            "golf/spa/marina/winery experiences (OSM), and Michelin three-star "
            "fine dining (Wikipedia's official list). Empty buckets are reported "
            "plainly, never filled with guesses."
        ),
        structured_output=True,
    )
    def luxury_experiences_tool_fn(destination: str) -> Union[ToolSuccess[dict], ToolError]:
        try:
            agent = _get_agent()
            result, obs = luxury_experiences_tool(destination, data_dir=agent.data_dir)
            return tool_success(_obs_payload(result, obs))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")

    @app.tool(
        name="web_search",
        title="Web Search",
        description=(
            "Live-web search (key-free DuckDuckGo + Wikipedia). Results are "
            "UNVERIFIED leads with full provenance (url + provider) — double-check "
            "before booking."
        ),
        structured_output=True,
    )
    def web_search_tool_fn(query: str, k: int = 5) -> Union[ToolSuccess[dict], ToolError]:
        try:
            result, obs = web_search_tool(query, k=k)
            return tool_success(_obs_payload(result, obs))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")

    @app.tool(
        name="advanced_search",
        title="Source-Aware Discovery",
        description=(
            "Source-aware discovery across travel and community sources "
            "(Wikipedia, Tripadvisor, Expedia, Reddit, Instagram, Wikivoyage). "
            "Returns structured LEADS, not facts — unavailable sources are listed "
            "honestly."
        ),
        structured_output=True,
    )
    def advanced_search_tool_fn(query: str, k_per_source: int = 3) -> Union[ToolSuccess[dict], ToolError]:
        try:
            result, obs = advanced_search_tool(query, k_per_source=k_per_source)
            return tool_success(_obs_payload(result, obs))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")

    @app.tool(
        name="book_ticket",
        title="Book Ticket (mock)",
        description=(
            "BookTicket() — MOCK booking that stays PENDING until the traveler "
            "confirms (never auto-confirmed). Pass the flight dict returned by "
            "search_flights."
        ),
        structured_output=True,
    )
    def book_ticket_tool(
        flight: dict,
        passenger: str = "Traveler",
    ) -> Union[ToolSuccess[dict], ToolError]:
        try:
            agent = _get_agent()
            result, obs = book_ticket(agent.bookings, flight, passenger=passenger, confirmed=False)
            return tool_success(_obs_payload(result, obs))
        except Exception as e:
            return tool_error(f"{e}", "GENERAL_EXCEPTION")
