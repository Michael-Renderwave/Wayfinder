"""Run the Wayfinder server:  uv run python -m wayfinder  →  http://localhost:8000"""

import os

import uvicorn

from wayfinder.app import app, get_agent


def main():
    # pre-build the vector index on first run (rebuilds itself if missing)
    get_agent()
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),  # 0.0.0.0 when deployed
        port=int(os.environ.get("PORT", "8000")),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
