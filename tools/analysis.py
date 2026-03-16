from __future__ import annotations

from datetime import date, datetime, timezone

from mcp.server.fastmcp import FastMCP

from cache.ttl_cache import RedisCache
from services.analysis_service import AnalysisService
from services.odds_service import OddsService

_cache: RedisCache
_analysis_service: AnalysisService
_odds_service: OddsService


def register(
    mcp: FastMCP,
    cache: RedisCache,
    analysis_service: AnalysisService,
    odds_service: OddsService,
) -> None:
    global _cache, _analysis_service, _odds_service
    _cache = cache
    _analysis_service = analysis_service
    _odds_service = odds_service

    @mcp.tool()
    async def find_value_bets(game_date: str = "") -> str:
        """Scan tonight's NBA games and return bets where edge exceeds 3%.

        Uses team win-rate baselines from last 10 games vs. implied probabilities
        from available sportsbooks. Kelly criterion fraction included for sizing.

        Args:
            game_date: Date in YYYY-MM-DD format. Defaults to today.
        """
        if game_date:
            try:
                parsed_date = date.fromisoformat(game_date)
            except ValueError:
                return f"Invalid date format: '{game_date}'. Use YYYY-MM-DD."
        else:
            parsed_date = datetime.now(tz=timezone.utc).date()

        bets = await _analysis_service.find_value_bets(game_date=parsed_date)

        if not bets:
            return f"No value bets found for {parsed_date} (edge threshold: 3%)."

        rows = [
            f"**Value Bets — {parsed_date}**\n",
            "| Selection | Book | Odds | Implied% | True% | Edge | Kelly | Confidence |",
            "|-----------|------|------|----------|-------|------|-------|------------|",
        ]
        for b in bets:
            rows.append(
                f"| {b.selection} | {b.bookmaker} | {b.american_odds:+d} | "
                f"{b.implied_probability:.1%} | {b.true_probability:.1%} | "
                f"{b.edge:.1%} | {b.kelly_fraction:.1%} | {b.confidence} |"
            )
        return "\n".join(rows)

    @mcp.tool()
    async def find_value_props(game_id: str, last_n_games: int = 10) -> str:
        """Scan all player props in a game for statistical edges.

        Compares each prop line against the player's last-N-games distribution.
        Returns over/under recommendations with edge size and direction.

        Args:
            game_id: The game ID returned by get_tonight_games.
            last_n_games: Number of recent games to build the distribution from.
        """
        bets = await _analysis_service.find_value_props(game_id, last_n=last_n_games)

        if not bets:
            return (
                f"No value props found for game `{game_id}` "
                f"(using last {last_n_games} games, edge threshold: 3%)."
            )

        rows = [
            f"**Value Props — `{game_id}` (last {last_n_games} games)**\n",
            "| Selection | Book | Odds | Implied% | True% | Edge | Kelly | Confidence |",
            "|-----------|------|------|----------|-------|------|-------|------------|",
        ]
        for b in bets:
            rows.append(
                f"| {b.selection} | {b.bookmaker} | {b.american_odds:+d} | "
                f"{b.implied_probability:.1%} | {b.true_probability:.1%} | "
                f"{b.edge:.1%} | {b.kelly_fraction:.1%} | {b.confidence} |"
            )
        return "\n".join(rows)

    @mcp.tool()
    async def get_weather_impact(game_id: str) -> str:
        """Retrieve weather context for the game venue city.

        NBA games are indoors, so this is most useful for travel impact, cold/fatigue
        context, or future expansion to outdoor sports.

        Args:
            game_id: The game ID returned by get_tonight_games.
        """
        games = await _odds_service.get_games()
        game = next((g for g in games if g.id == game_id), None)

        if not game:
            return f"Game `{game_id}` not found in tonight's schedule."

        # Use home team city as venue approximation
        venue_city = game.home_team.split()[-1] if game.home_team else "New York"
        game_date = game.commence_time.date()

        weather = await _analysis_service.get_weather_impact(venue_city, game_date)

        if "error" in weather:
            return f"Could not retrieve weather for {venue_city}: {weather['error']}"

        return (
            f"**Weather — {weather.get('city')} ({weather.get('date')})**\n\n"
            f"- Temperature: {weather.get('temperature_f')}°F "
            f"(feels like {weather.get('feels_like_f')}°F)\n"
            f"- Conditions: {weather.get('conditions')}\n"
            f"- Wind: {weather.get('wind_mph')} mph\n"
            f"- Precipitation chance: {weather.get('precipitation_chance', 0):.0%}\n\n"
            f"_{weather.get('note', '')}_"
        )
