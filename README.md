# Wayfinder
Travel research agent for Emeritus Agentic Course

## Getting Started
1. Download repo
2. Install [uv](https://docs.astral.sh/uv/)
3. Run `uv sync`
4. Run `uv run python -m wayfinder` → http://localhost:8000

We want an agent to:
1. Get live travel data (flights, hotels, weather, safety, luxury) from reputable key-free sources
2. Store any of the data that we get so that we can use it for later
3. Utilize this stored data for future trip planning and recommendations
4. Add guardrails to prevent misuse — sample data is labeled, fares are estimates, and nothing is invented
5. Add reasoning traces (thought → action → observation) so every step can be analyzed

## Use with Claude Code (MCP)
The repo ships a `.mcp.json` that registers the `wayfinder` MCP server (11 tools: `plan_trip` plus flights, hotels, KB, seasons, safety, local places, luxury, web, and a mock `book_ticket`). Just open Claude Code in this folder and ask it to plan a trip — e.g. "use Wayfinder to plan a 5-day Cayman Islands trip from Miami." No extra setup beyond `uv sync`.

## If Something Goes Wrong
- Port 8000 already in use → run `PORT=8080 uv run python -m wayfinder` instead
- `uv sync` needs internet once (downloads ~5 MB of dependencies); live flight/hotel/weather/safety data also needs internet. Knowledge-base answers work offline
- No Python 3.13 on your machine? uv downloads and manages one automatically — you only need uv itself
