from __future__ import annotations

import json
from datetime import date, datetime, timezone

from mcp.server.fastmcp import FastMCP

from cache.ttl_cache import RedisCache
from config import settings
from services.odds_service import OddsService

# These are populated by server.py at startup
_cache: RedisCache
_odds_service: OddsService


def register(mcp: FastMCP, cache: RedisCache, odds_service: OddsService) -> None:
    global _cache, _odds_service
    _cache = cache
    _odds_service = odds_service

    @mcp.tool()
    async def get_tonight_games(game_date: str = "") -> str:
        """Return NBA games with tip-off times for a given date.

        Args:
            game_date: Date in YYYY-MM-DD format. Defaults to today.
        """
        if game_date:
            try:
                parsed_date = date.fromisoformat(game_date)
            except ValueError:
                return f"Invalid date format: '{game_date}'. Use YYYY-MM-DD."
        else:
            parsed_date = date.today()

        games = await _odds_service.get_games(game_date=parsed_date)
        if not games:
            return f"No NBA games found for {parsed_date}."

        lines = [f"**NBA Games — {parsed_date}**\n"]
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
                    f"  Spread: {line.spread.home_team} {line.spread.home_point:+.1f} "
                    f"({line.spread.home_price:+d}) | "
                    f"{line.spread.away_team} {line.spread.away_point:+.1f} "
                    f"({line.spread.away_price:+d})"
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
    async def get_pinnacle_line(game_id: str) -> str:
        """Get Pinnacle's sharp line and no-vig fair odds probabilities for a game.

        Args:
            game_id: The game ID returned by get_tonight_games.
        """
        try:
            line = await _odds_service.get_pinnacle_odds(game_id)
        except ValueError as exc:
            return f"Could not fetch Pinnacle line for `{game_id}`: {exc}"

        if not line:
            return f"Pinnacle line not available for `{game_id}`."

        from services.analysis_service import AnalysisService
        from cache.ttl_cache import RedisCache
        from services.stats_service import StatsService

        # Use inline no-vig calc to avoid importing the full service
        def _imp(price: int) -> float:
            if price > 0:
                return 100 / (price + 100)
            return abs(price) / (abs(price) + 100)

        out = [f"**Pinnacle Sharp Line — `{game_id}`**\n"]
        out.append(f"{line.away_team} @ {line.home_team}\n")

        if line.moneyline_home is not None and line.moneyline_away is not None:
            imp_h = _imp(line.moneyline_home)
            imp_a = _imp(line.moneyline_away)
            total = imp_h + imp_a
            nv_h = imp_h / total
            nv_a = imp_a / total
            out.append("**Moneyline (no-vig fair odds)**")
            out.append(f"- {line.home_team}: {line.moneyline_home:+d} → {nv_h:.1%} fair probability")
            out.append(f"- {line.away_team}: {line.moneyline_away:+d} → {nv_a:.1%} fair probability")
            out.append(f"- Pinnacle vig: {(total - 1) * 100:.2f}%\n")

        if line.spread:
            out.append(
                f"**Spread:** {line.spread.home_team} {line.spread.home_point:+.1f} "
                f"({line.spread.home_price:+d}) | "
                f"{line.spread.away_team} {line.spread.away_point:+.1f} "
                f"({line.spread.away_price:+d})\n"
            )

        if line.total:
            out.append(f"**Total:** {line.total.point} — Over {line.total.over_price:+d} / Under {line.total.under_price:+d}")

        return "\n".join(out)

    @mcp.tool()
    async def get_line_movement(game_id: str) -> str:
        """Show how the spread line has shifted since it opened.

        Args:
            game_id: The game ID returned by get_tonight_games.
        """
        try:
            movement = await _odds_service.get_line_movement(game_id)
        except ValueError as exc:
            return f"Could not fetch line movement for `{game_id}`: {exc}\n\nTip: use `get_tonight_games` to get a valid game ID."
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
