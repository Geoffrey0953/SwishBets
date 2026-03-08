from __future__ import annotations

import json
from datetime import date, datetime, timezone

from mcp.server.fastmcp import FastMCP

from swishbets.cache.ttl_cache import RedisCache
from swishbets.config import settings
from swishbets.services.odds_service import OddsService

# These are populated by server.py at startup
_cache: RedisCache
_odds_service: OddsService


def register(mcp: FastMCP, cache: RedisCache, odds_service: OddsService) -> None:
    global _cache, _odds_service
    _cache = cache
    _odds_service = odds_service

    @mcp.tool()
    async def get_tonight_games() -> str:
        """Return tonight's NBA games with tip-off times."""
        today = datetime.now(tz=timezone.utc).date()
        games = await _odds_service.get_games(game_date=today)
        if not games:
            return "No NBA games found for tonight."

        lines = ["**Tonight's NBA Games**\n"]
        for g in games:
            tip = g.commence_time.astimezone().strftime("%I:%M %p %Z")
            lines.append(f"- {g.away_team} @ {g.home_team}  |  {tip}  |  `{g.id}`")
        return "\n".join(lines)

    @mcp.tool()
    async def get_odds(game_id: str) -> str:
        """Return moneyline, spread, and total odds for a game across all available sportsbooks.

        Args:
            game_id: The game ID returned by get_tonight_games.
        """
        lines_data = await _odds_service.get_odds(game_id)
        if not lines_data:
            return f"No odds found for game `{game_id}`."

        out = [f"**Odds for game `{game_id}`**\n"]
        for line in lines_data:
            out.append(f"### {line.bookmaker}")
            if line.moneyline_home is not None:
                out.append(
                    f"  Moneyline: {line.home_team} **{line.moneyline_home:+d}** | "
                    f"{line.away_team} **{line.moneyline_away:+d}**"
                )
            if line.spread:
                out.append(
                    f"  Spread: {line.spread.team} {line.spread.point:+.1f} "
                    f"({line.spread.price:+d})"
                )
            if line.total:
                out.append(
                    f"  Total: O{line.total.point} ({line.total.over_price:+d}) | "
                    f"U{line.total.point} ({line.total.under_price:+d})"
                )
            out.append("")
        return "\n".join(out)

    @mcp.tool()
    async def get_player_props(game_id: str, player_name: str = "") -> str:
        """Return player prop lines (points/assists/rebounds/threes) for a game.

        Args:
            game_id: The game ID.
            player_name: Optional player name filter. Leave blank for all props.
        """
        props = await _odds_service.get_player_props(game_id)
        if not props:
            return f"No player props found for game `{game_id}`."

        if player_name:
            props = [p for p in props if player_name.lower() in p.player_name.lower()]

        if not props:
            return f"No props found for player '{player_name}'."

        rows = ["| Player | Market | Line | Over | Under | Book |",
                "|--------|--------|------|------|-------|------|"]
        for p in props:
            market = p.market.replace("player_", "")
            rows.append(
                f"| {p.player_name} | {market} | {p.line} | "
                f"{p.over_price:+d} | {p.under_price:+d} | {p.bookmaker} |"
            )
        return "\n".join(rows)

    @mcp.tool()
    async def compare_books(game_id: str, market: str = "h2h") -> str:
        """Compare a specific market line across all sportsbooks to find the best number.

        Args:
            game_id: The game ID.
            market: One of "h2h" (moneyline), "spreads", or "totals".
        """
        comparison = await _odds_service.compare_books(game_id, market)
        if not comparison:
            return f"No {market} data found for game `{game_id}`."

        out = [f"**{market.upper()} comparison for `{game_id}`**\n"]
        out.append(f"```\n{json.dumps(comparison, indent=2)}\n```")
        return "\n".join(out)

    @mcp.tool()
    async def get_line_movement(game_id: str) -> str:
        """Show how the spread line has shifted since it opened.

        Args:
            game_id: The game ID.
        """
        movement = await _odds_service.get_line_movement(game_id)
        direction_emoji = {"Up": "⬆", "Down": "⬇", "Flat": "➡"}.get(
            movement.direction, ""
        )
        return (
            f"**Line Movement — `{game_id}`**\n\n"
            f"- Market: {movement.market}\n"
            f"- Book: {movement.bookmaker}\n"
            f"- Opening: {movement.opening_line:+.1f}\n"
            f"- Current: {movement.current_line:+.1f}\n"
            f"- Delta: {movement.delta:+.1f} {direction_emoji}\n"
            f"- Direction: {movement.direction}"
        )
