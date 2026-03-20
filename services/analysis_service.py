from __future__ import annotations

import logging
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from config import settings
from models.schemas import ValueBet
from services.odds_service import OddsService
from services.stats_service import StatsService

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 0.03  # 3% minimum edge to flag a value bet


class AnalysisService:
    def __init__(self, odds_service: OddsService, stats_service: StatsService) -> None:
        self.odds = odds_service
        self.stats = stats_service

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    def implied_probability(self, american_odds: int) -> float:
        """Convert American odds to implied probability (0–1)."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        return abs(american_odds) / (abs(american_odds) + 100)

    def american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds > 0:
            return (american_odds / 100) + 1.0
        return (100 / abs(american_odds)) + 1.0

    def calculate_edge(self, true_prob: float, implied_prob: float) -> float:
        """Edge = true probability minus implied probability."""
        return true_prob - implied_prob

    def kelly_criterion(self, edge: float, decimal_odds: float) -> float:
        """Kelly fraction = (edge * decimal_odds) / (decimal_odds - 1).
        Capped at 25% and returned as a fraction (e.g. 0.05 = 5%).
        """
        if decimal_odds <= 1 or edge <= 0:
            return 0.0
        kelly = (edge * decimal_odds) / (decimal_odds - 1)
        return round(min(kelly, 0.25), 4)

    def _confidence(self, edge: float) -> str:
        if edge >= 0.08:
            return "High"
        if edge >= 0.04:
            return "Medium"
        return "Low"

    # ------------------------------------------------------------------
    # Value bet finders
    # ------------------------------------------------------------------

    async def find_value_bets(self, game_date: Optional[date] = None) -> list[ValueBet]:
        """Scan tonight's games for moneyline value versus team win-rate baseline."""
        if game_date is None:
            game_date = date.today()

        games = await self.odds.get_games(game_date=game_date)
        value_bets: list[ValueBet] = []

        for game in games:
            try:
                lines = await self.odds.get_odds(game.id, markets=["h2h"])
            except Exception as exc:
                logger.warning("Could not fetch odds for %s: %s", game.id, exc)
                continue

            # Collect best (lowest vig) line per team across all books
            home_odds_list: list[int] = []
            away_odds_list: list[int] = []
            best_bookmaker = "consensus"
            for line in lines:
                if line.moneyline_home is not None:
                    home_odds_list.append(line.moneyline_home)
                if line.moneyline_away is not None:
                    away_odds_list.append(line.moneyline_away)
                best_bookmaker = line.bookmaker

            if not home_odds_list or not away_odds_list:
                continue

            # Use the most favorable (highest payout) line available
            best_home = max(home_odds_list, key=lambda x: self.american_to_decimal(x))
            best_away = max(away_odds_list, key=lambda x: self.american_to_decimal(x))

            implied_home = self.implied_probability(best_home)
            implied_away = self.implied_probability(best_away)

            # Baseline: team win-rate from last 10 games
            # We use win_pct as a rough true probability
            home_team_data = self.stats.find_team_by_name(game.home_team)
            away_team_data = self.stats.find_team_by_name(game.away_team)

            if not home_team_data or not away_team_data:
                continue

            try:
                home_stats = await self.stats.get_team_stats(
                    home_team_data["id"], last_n=10
                )
                away_stats = await self.stats.get_team_stats(
                    away_team_data["id"], last_n=10
                )
            except Exception as exc:
                logger.warning("Could not fetch stats for game %s: %s", game.id, exc)
                continue

            # Simple Pythagorean-style normalisation
            home_true = home_stats.win_pct
            away_true = away_stats.win_pct
            total = home_true + away_true
            if total > 0:
                home_true = home_true / total
                away_true = away_true / total

            # Home-court adjustment (+3%)
            home_true = min(home_true + 0.03, 0.99)
            away_true = max(away_true - 0.03, 0.01)

            for team, true_prob, implied_prob, american_odds in [
                (game.home_team, home_true, implied_home, best_home),
                (game.away_team, away_true, implied_away, best_away),
            ]:
                edge = self.calculate_edge(true_prob, implied_prob)
                if edge >= EDGE_THRESHOLD:
                    decimal = self.american_to_decimal(american_odds)
                    value_bets.append(
                        ValueBet(
                            event_id=game.id,
                            market="h2h",
                            selection=team,
                            bookmaker=best_bookmaker,
                            american_odds=american_odds,
                            implied_probability=round(implied_prob, 4),
                            true_probability=round(true_prob, 4),
                            edge=round(edge, 4),
                            kelly_fraction=self.kelly_criterion(edge, decimal),
                            confidence=self._confidence(edge),
                        )
                    )

        value_bets.sort(key=lambda b: b.edge, reverse=True)
        return value_bets

    async def find_value_props(
        self, event_id: str, last_n: int = 10
    ) -> list[ValueBet]:
        """For each player prop in a game, compare the line vs player's last-N average."""
        try:
            props = await self.odds.get_player_props(event_id)
        except Exception as exc:
            logger.warning("Could not fetch props for %s: %s", event_id, exc)
            return []

        if not props:
            return []

        # Determine commence_time from the first prop's event odds
        commence_time: Optional[datetime] = None
        try:
            lines = await self.odds.get_odds(event_id, markets=["h2h"])
            if lines:
                commence_time = lines[0].commence_time
        except Exception:
            pass

        # Fetch opening prop lines for all markets at once
        markets_needed = list({p.market for p in props})
        opening_lines = await self._get_opening_prop_lines(event_id, commence_time, markets_needed)

        value_bets: list[ValueBet] = []

        for prop in props:
            try:
                player_data = await self._find_player(prop.player_name)
                if not player_data:
                    continue

                game_logs = await self.stats.get_player_stats(player_data["id"])
                if not game_logs:
                    continue

                stat_col = self._prop_market_to_stat_col(prop.market)
                if stat_col is None:
                    continue

                values = [
                    float(g[stat_col])
                    for g in game_logs[:last_n]
                    if stat_col in g and g[stat_col] is not None
                ]
                if len(values) < 3:
                    continue

                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 1.0

                # Z-score of prop line relative to historical distribution
                z = (mean - prop.line) / max(stdev, 0.5)

                # Convert z-score to probability using normal CDF approximation
                true_over_prob = self._normal_cdf(z)
                true_under_prob = 1.0 - true_over_prob

                implied_over = self.implied_probability(prop.over_price)
                implied_under = self.implied_probability(prop.under_price)

                # Opening line for this player+market (for line movement signal)
                opening_line = opening_lines.get(prop.player_name, {}).get(prop.market)

                for direction, true_prob, implied_prob, american_odds in [
                    ("Over", true_over_prob, implied_over, prop.over_price),
                    ("Under", true_under_prob, implied_under, prop.under_price),
                ]:
                    edge = self.calculate_edge(true_prob, implied_prob)

                    line_movement_label: Optional[str] = None
                    if opening_line is not None:
                        line_delta = prop.line - opening_line
                        if abs(line_delta) >= 0.25:
                            # Agrees: line moved up + Over, or moved down + Under
                            agrees = (line_delta > 0 and direction == "Over") or (
                                line_delta < 0 and direction == "Under"
                            )
                            if agrees:
                                edge += 0.015
                                arrow = "↑" if line_delta > 0 else "↓"
                                line_movement_label = f"{line_delta:+.1f} {arrow} (agrees)"
                            else:
                                edge -= 0.010
                                arrow = "↑" if line_delta > 0 else "↓"
                                line_movement_label = f"{line_delta:+.1f} {arrow} (disagrees)"

                    if edge >= EDGE_THRESHOLD:
                        decimal = self.american_to_decimal(american_odds)
                        label = f"{prop.player_name} {direction} {prop.line} {prop.market}"
                        value_bets.append(
                            ValueBet(
                                event_id=event_id,
                                market=prop.market,
                                selection=label,
                                bookmaker=prop.bookmaker,
                                american_odds=american_odds,
                                implied_probability=round(implied_prob, 4),
                                true_probability=round(true_prob, 4),
                                edge=round(edge, 4),
                                kelly_fraction=self.kelly_criterion(edge, decimal),
                                confidence=self._confidence(edge),
                                line_movement=line_movement_label,
                            )
                        )
            except Exception as exc:
                logger.debug("Prop analysis error for %s: %s", prop.player_name, exc)
                continue

        value_bets.sort(key=lambda b: b.edge, reverse=True)
        return value_bets

    async def get_weather_impact(self, venue_city: str, game_date: date) -> dict[str, Any]:
        """Call OpenWeatherMap for weather context at the venue city."""
        if not settings.openweather_api_key:
            return {"note": "OpenWeatherMap API key not configured"}

        try:
            params = {
                "q": venue_city,
                "appid": settings.openweather_api_key,
                "units": "imperial",
                "dt": int(
                    datetime_from_date(game_date).timestamp()
                ),
            }
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{settings.openweather_base_url}/forecast", params=params
                )
                resp.raise_for_status()
                data = resp.json()

            forecast = data.get("list", [{}])[0]
            return {
                "city": venue_city,
                "date": str(game_date),
                "temperature_f": forecast.get("main", {}).get("temp"),
                "feels_like_f": forecast.get("main", {}).get("feels_like"),
                "conditions": forecast.get("weather", [{}])[0].get("description"),
                "wind_mph": forecast.get("wind", {}).get("speed"),
                "precipitation_chance": forecast.get("pop", 0),
                "note": "NBA games are indoors — weather context for travel/player conditions only",
            }
        except Exception as exc:
            logger.warning("Weather API error for %s: %s", venue_city, exc)
            return {"city": venue_city, "error": str(exc)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_opening_prop_lines(
        self,
        event_id: str,
        commence_time: Optional[datetime],
        markets: list[str],
    ) -> dict[str, dict[str, float]]:
        """Return {player_name: {market: opening_line}} from 48h-before snapshot."""
        if commence_time is None or not markets:
            return {}

        opening_ts = (commence_time - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            hist = await self.odds.get_historical_event_odds(event_id, opening_ts, markets=markets)
        except Exception as exc:
            logger.debug("Could not fetch opening prop lines: %s", exc)
            return {}

        hist_data = hist.get("data") or {}
        result: dict[str, dict[str, float]] = {}

        for bookmaker in hist_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", outcome.get("name", ""))
                    if not player_name or "point" not in outcome:
                        continue
                    result.setdefault(player_name, {})
                    # Only set if not already seen (first bookmaker wins)
                    if market_key not in result[player_name]:
                        result[player_name][market_key] = float(outcome["point"])

        return result

    async def _find_player(self, name: str) -> Optional[dict[str, Any]]:
        try:
            from nba_api.stats.static import players as nba_players

            name_lower = name.lower()
            for p in nba_players.get_active_players():
                if name_lower in p["full_name"].lower():
                    return p
        except Exception:
            pass
        return None

    def _prop_market_to_stat_col(self, market: str) -> Optional[str]:
        mapping = {
            "player_points": "PTS",
            "player_rebounds": "REB",
            "player_assists": "AST",
            "player_threes": "FG3M",
            "player_blocks": "BLK",
            "player_steals": "STL",
            "player_turnovers": "TOV",
            "player_double_double": None,
        }
        return mapping.get(market)

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Approximation of the standard normal CDF."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def datetime_from_date(d: date):
    from datetime import datetime

    return datetime(d.year, d.month, d.day, 19, 0)  # assume 7 PM tip-off
