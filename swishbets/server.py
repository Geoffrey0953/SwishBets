"""SwishBets MCP Server — NBA sports betting analysis via SSE transport."""
from __future__ import annotations

import json
import logging
from datetime import date

import uvicorn
from mcp.server.fastmcp import FastMCP

from swishbets.cache.ttl_cache import RedisCache
from swishbets.config import settings
from swishbets.services.analysis_service import AnalysisService
from swishbets.services.odds_service import OddsService
from swishbets.services.stats_service import StatsService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="SwishBets",
    instructions=(
        "NBA sports betting analysis assistant. "
        "Provides odds, player props, injury reports, team stats, "
        "and statistical value-bet recommendations powered by live data."
    ),
)

# ---------------------------------------------------------------------------
# Shared service instances
# ---------------------------------------------------------------------------
cache = RedisCache(settings.redis_url)
odds_service = OddsService(cache)
stats_service = StatsService(cache)
analysis_service = AnalysisService(odds_service, stats_service)

# ---------------------------------------------------------------------------
# Register tools
# ---------------------------------------------------------------------------
from swishbets.tools import odds as odds_tools
from swishbets.tools import stats as stats_tools
from swishbets.tools import analysis as analysis_tools

odds_tools.register(mcp, cache, odds_service)
stats_tools.register(mcp, cache, stats_service)
analysis_tools.register(mcp, cache, analysis_service, odds_service)

# ---------------------------------------------------------------------------
# Register resources
# ---------------------------------------------------------------------------

@mcp.resource("nba://teams")
async def resource_teams() -> str:
    """All active NBA teams."""
    teams = await stats_service.get_all_teams()
    return json.dumps([t.model_dump() for t in teams], indent=2)


@mcp.resource("nba://players")
async def resource_players() -> str:
    """All active NBA players."""
    players = await stats_service.get_all_players()
    return json.dumps([p.model_dump() for p in players], indent=2)


@mcp.resource("nba://standings")
async def resource_standings() -> str:
    """Current NBA standings."""
    standings = await stats_service.get_standings()
    return json.dumps(standings, indent=2)


@mcp.resource("nba://schedule")
async def resource_schedule() -> str:
    """Today's NBA schedule."""
    games = await stats_service.get_schedule(date.today())
    return json.dumps([g.model_dump() for g in games], indent=2, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info(
        "Starting SwishBets MCP server on http://%s:%d/sse",
        settings.server_host,
        settings.server_port,
    )
    # FastMCP exposes a Starlette app via .sse_app() for SSE transport
    app = mcp.sse_app()
    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
