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
            parsed_date = date.today()

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
    async def find_positive_ev(min_edge_pct: float = 1.0) -> str:
        """Scan all upcoming games for +EV bets using Pinnacle as the true probability.

        Covers moneyline, totals, and player props across all 24+ books.
        Uses Pinnacle's no-vig odds as the sharp reference — any book offering
        better than Pinnacle's true probability is flagged as +EV.

        Args:
            min_edge_pct: Minimum edge percentage to include (default 1.0 = 1%).
        """
        min_edge = min_edge_pct / 100.0
        bets = await _analysis_service.find_positive_ev(min_edge=min_edge)

        if not bets:
            return f"No +EV bets found (min edge: {min_edge_pct}%)."

        rows = [
            f"**+EV Bets — Pinnacle No-Vig Reference (min edge: {min_edge_pct}%)**\n",
            "| Game | Market | Selection | Book | Odds | Pinnacle True% | Implied% | Edge | Kelly | Conf |",
            "|------|--------|-----------|------|------|----------------|----------|------|-------|------|",
        ]
        for b in bets:
            # Resolve game name from event_id
            game_label = b.event_id[:8]
            rows.append(
                f"| {game_label} | {b.market} | {b.selection} | {b.bookmaker} | "
                f"{b.american_odds:+d} | {b.true_probability:.1%} | "
                f"{b.implied_probability:.1%} | {b.edge:.1%} | "
                f"{b.kelly_fraction:.1%} | {b.confidence} |"
            )
        rows.append(f"\n_{len(bets)} opportunities found across all games and markets._")
        return "\n".join(rows)

    @mcp.tool()
    async def find_arb(min_arb_pct: float = 0.1) -> str:
        """Scan all upcoming games for arbitrage opportunities across all books.

        Finds situations where backing both sides of a market (e.g. home ML on
        DraftKings + away ML on Bovada) guarantees profit regardless of outcome.
        Covers moneyline, totals, and player props.

        Args:
            min_arb_pct: Minimum guaranteed profit % to include (default 0.1 = 0.1%).
        """
        arbs = await _analysis_service.find_arb(min_arb_pct=min_arb_pct / 100.0)

        if not arbs:
            return f"No arb opportunities found (min: {min_arb_pct}% profit)."

        rows = [
            f"**Arb Opportunities (min {min_arb_pct}% guaranteed profit per $100)**\n",
            "| Game | Market | Side A | Book A | Odds A | Side B | Book B | Odds B | Arb% | Stake A | Stake B |",
            "|------|--------|--------|--------|--------|--------|--------|--------|------|---------|---------|",
        ]
        for a in arbs:
            game_label = f"{a.away_team[:3]} @ {a.home_team[:3]}"
            rows.append(
                f"| {game_label} | {a.market} | {a.side_a} | {a.side_a_book} | "
                f"{a.side_a_odds:+d} | {a.side_b} | {a.side_b_book} | "
                f"{a.side_b_odds:+d} | {a.arb_pct:.2%} | "
                f"${a.side_a_stake:.2f} | ${a.side_b_stake:.2f} |"
            )
        rows.append(f"\n_{len(arbs)} arb opportunities found. Stakes shown per $100 total._")
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
