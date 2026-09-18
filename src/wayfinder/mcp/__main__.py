import sys

from mcp.server import MCPServer

from wayfinder.mcp.tools import register_wayfinder

app = MCPServer(
    name="wayfinder",
    instructions=(
        "Wayfinder: a travel research agent. Use plan_trip for a full grounded trip "
        "briefing (flights, stays, dining, safety, luxury — cited + confidence-banded), "
        "or the individual tools (search_flights, search_hotels, search_kb, best_time, "
        "travel_safety, local_places, luxury_experiences, web_search, advanced_search, "
        "book_ticket) for targeted lookups. All sources are key-free; sample data is "
        "labeled, fares are estimates, and empty buckets are reported plainly."
    ),
)

_ = register_wayfinder(app)


def main():
    try:
        app.run()
    except Exception as e:
        print(
            f"wayfinder MCP server error: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
