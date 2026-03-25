from __future__ import annotations

import logging
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from cache.ttl_cache import STATS
from config import settings
from models.schemas import ValueBet
from services.odds_service import OddsService
from services.stats_service import StatsService

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 0.03  # 3% minimum edge to flag a value bet


class AnalysisService:
    def __init__(
        self,
        odds_service: OddsService,
        stats_service: StatsService,
        defense_service: Optional[Any] = None,
    ) -> None:
        self.odds = odds_service
        self.stats = stats_service
        self.defense_service = defense_service

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

            best_home = max(home_odds_list, key=lambda x: self.american_to_decimal(x))
            best_away = max(away_odds_list, key=lambda x: self.american_to_decimal(x))

            implied_home = self.implied_probability(best_home)
            implied_away = self.implied_probability(best_away)

            home_team_data = self.stats.find_team_by_name(game.home_team)
            away_team_data = self.stats.find_team_by_name(game.away_team)

            if not home_team_data or not away_team_data:
                continue

            try:
                home_stats = await self.stats.get_team_stats(home_team_data["id"], last_n=10)
                away_stats = await self.stats.get_team_stats(away_team_data["id"], last_n=10)
            except Exception as exc:
                logger.warning("Could not fetch stats for game %s: %s", game.id, exc)
                continue

            home_true = home_stats.win_pct
            away_true = away_stats.win_pct
            total = home_true + away_true
            if total > 0:
                home_true = home_true / total
                away_true = away_true / total

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

        # Resolve game info: commence_time, home/away teams
        commence_time: Optional[datetime] = None
        game_date: Optional[date] = None
        game_home_team: Optional[str] = None
        game_away_team: Optional[str] = None
        try:
            lines = await self.odds.get_odds(event_id, markets=["h2h"])
            if lines:
                commence_time = lines[0].commence_time
                game_date = commence_time.date()
                game_home_team = lines[0].home_team
                game_away_team = lines[0].away_team
        except Exception:
            pass

        # Resolve team IDs and fetch team stats for pace + b2b
        home_team_id: Optional[int] = None
        away_team_id: Optional[int] = None
        home_team_stats = None
        away_team_stats = None

        for team_name, setter in [(game_home_team, "home"), (game_away_team, "away")]:
            if not team_name:
                continue
            try:
                team_data = self.stats.find_team_by_name(team_name)
                if team_data:
                    tid = team_data["id"]
                    tstats = await self.stats.get_team_stats(tid, last_n=10)
                    if setter == "home":
                        home_team_id = tid
                        home_team_stats = tstats
                    else:
                        away_team_id = tid
                        away_team_stats = tstats
            except Exception as exc:
                logger.warning("Could not resolve team %s for event %s: %s", team_name, event_id, exc)

        # Compute combined pace grade for this matchup (used per-prop)
        combined_pace: Optional[float] = None
        game_pace_grade: Optional[str] = None
        if (
            home_team_stats and away_team_stats
            and home_team_stats.pace is not None
            and away_team_stats.pace is not None
        ):
            combined_pace = (home_team_stats.pace + away_team_stats.pace) / 2
            if combined_pace >= 100:
                game_pace_grade = "fast"
            elif combined_pace >= 97:
                game_pace_grade = "average"
            else:
                game_pace_grade = "slow"

        # Fetch injury reports for usage adjustment
        home_injuries: list = []
        away_injuries: list = []
        if home_team_id:
            try:
                home_report = await self.stats.get_injury_report(home_team_id)
                home_injuries = home_report.players
            except Exception:
                pass
        if away_team_id:
            try:
                away_report = await self.stats.get_injury_report(away_team_id)
                away_injuries = away_report.players
            except Exception:
                pass

        # Fetch opening prop lines for line-movement signal (disabled — historical API unreliable)
        # markets_needed = list({p.market for p in props})
        # opening_lines = await self._get_opening_prop_lines(event_id, commence_time, markets_needed)
        opening_lines: dict = {}

        value_bets: list[ValueBet] = []

        for prop in props:
            try:
                player_data = await self._find_player(prop.player_name)
                if not player_data:
                    continue

                game_logs = await self.stats.get_player_stats(player_data["id"])
                if not game_logs:
                    continue

                # Determine player's team from most recent game log
                player_team_id: Optional[int] = None
                opponent_team_id: Optional[int] = None
                player_team_abbrev = game_logs[0].get("TEAM_ABBREVIATION") if game_logs else None
                if player_team_abbrev:
                    pt_data = self.stats.find_team_by_name(player_team_abbrev)
                    if pt_data:
                        player_team_id = pt_data["id"]
                        if player_team_id == home_team_id:
                            opponent_team_id = away_team_id
                        elif player_team_id == away_team_id:
                            opponent_team_id = home_team_id

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

                z = (mean - prop.line) / max(stdev, 0.5)
                true_over_prob = self._normal_cdf(z)
                true_under_prob = 1.0 - true_over_prob

                implied_over = self.implied_probability(prop.over_price)
                implied_under = self.implied_probability(prop.under_price)

                # Opening line for line-movement signal
                opening_line = opening_lines.get(prop.player_name, {}).get(prop.market)

                # --- Trend detection (last3 vs last10) ---
                trend: Optional[str] = None
                values_last3 = values[:3]
                values_last10 = values[:10]
                if len(values_last3) >= 3 and len(values_last10) >= 5:
                    mean3 = statistics.mean(values_last3)
                    mean10 = statistics.mean(values_last10)
                    z3 = (mean3 - prop.line) / max(stdev, 0.5)
                    z10 = (mean10 - prop.line) / max(stdev, 0.5)
                    trend_delta = self._normal_cdf(z3) - self._normal_cdf(z10)
                    if trend_delta > 0.20:
                        trend = "heating_up"
                    elif trend_delta < -0.20:
                        trend = "cooling_off"
                    else:
                        trend = "stable"

                # --- Minutes consistency ---
                minutes_grade: Optional[str] = None
                try:
                    minutes_data = await self.stats.get_player_minutes_consistency(
                        player_data["id"], last_n=last_n
                    )
                    minutes_grade = minutes_data.get("grade")
                except Exception:
                    pass

                # --- Back-to-back ---
                is_b2b = False
                if player_team_id and game_date:
                    try:
                        is_b2b = await self.stats.is_team_on_back_to_back(player_team_id, game_date)
                    except Exception:
                        pass

                # --- Usage adjustment ---
                player_injuries = (
                    home_injuries if player_team_id == home_team_id else away_injuries
                )
                out_players = [
                    p.player_name
                    for p in player_injuries
                    if p.status in ("Out", "Out For Season")
                ]
                usage_adj = await self._calculate_usage_adjustment(
                    player_data["id"], player_team_id, out_players
                )

                # --- Opponent defensive rank ---
                opp_def_rank: Optional[int] = None
                opp_def_grade: Optional[str] = None
                if opponent_team_id and self.defense_service:
                    try:
                        opp_def_result = await self.defense_service.get_opponent_def_rank(
                            opponent_team_id, stat_col
                        )
                        opp_def_rank = opp_def_result.get("rank")
                        opp_def_grade = opp_def_result.get("grade")
                    except Exception:
                        pass

                # --- Build ValueBet for Over and Under ---
                for direction, true_prob, implied_prob, american_odds in [
                    ("Over", true_over_prob, implied_over, prop.over_price),
                    ("Under", true_under_prob, implied_under, prop.under_price),
                ]:
                    edge = self.calculate_edge(true_prob, implied_prob)

                    # Line movement adjustment (existing)
                    line_movement_label: Optional[str] = None
                    if opening_line is not None:
                        line_delta = prop.line - opening_line
                        if abs(line_delta) >= 0.25:
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

                    # Trend adjustment
                    if trend == "heating_up":
                        edge += 0.02 if direction == "Over" else -0.01
                    elif trend == "cooling_off":
                        edge += -0.02 if direction == "Over" else 0.01

                    # Minutes consistency adjustment
                    if minutes_grade == "very_consistent":
                        edge += 0.015
                    elif minutes_grade == "consistent":
                        edge += 0.005
                    elif minutes_grade == "volatile":
                        edge -= 0.015

                    # Back-to-back fatigue adjustment
                    if is_b2b:
                        if prop.market in ("player_points", "player_assists"):
                            edge -= 0.01
                        elif prop.market in ("player_rebounds", "player_threes"):
                            edge -= 0.005

                    # Pace-of-play adjustment (points and assists only)
                    if combined_pace is not None and prop.market in (
                        "player_points", "player_assists"
                    ):
                        if combined_pace >= 100:
                            edge += 0.008
                        elif combined_pace < 97:
                            edge -= 0.008

                    # Usage adjustment
                    edge += usage_adj

                    # Opponent defensive rank adjustment
                    if opp_def_grade == "weak":
                        edge += 0.010
                    elif opp_def_grade == "elite":
                        edge -= 0.010

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
                                trend=trend,
                                minutes_grade=minutes_grade,
                                back_to_back=is_b2b if is_b2b else None,
                                pace_grade=game_pace_grade,
                                usage_boost=round(usage_adj, 4) if usage_adj != 0.0 else None,
                                opponent_def_rank=opp_def_rank,
                                opponent_def_grade=opp_def_grade,
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

    async def _calculate_usage_adjustment(
        self,
        player_id: int,
        team_id: Optional[int],
        injured_players: list[str],
    ) -> float:
        """Returns an edge delta based on usage absorbed from injured teammates."""
        if not injured_players or team_id is None:
            return 0.0

        try:
            cache_key = f"league_player_stats:{settings.nba_season}"
            player_stats_data = await self.stats.cache.get(cache_key)
            if not player_stats_data:
                print("[API CALL]   nba_api → LeagueDashPlayerStats (Advanced, PerGame)")
                from nba_api.stats.endpoints import leaguedashplayerstats

                result = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=settings.nba_season,
                    measure_type_detailed_defense="Advanced",
                    per_mode_detailed="PerGame",
                )
                df = result.league_dash_player_stats.get_data_frame()
                player_stats_data = df.to_dict(orient="records")
                await self.stats.cache.set(cache_key, player_stats_data, STATS)

            # Filter to team members
            team_players = [p for p in player_stats_data if p.get("TEAM_ID") == team_id]
            player_row = next(
                (p for p in player_stats_data if p.get("PLAYER_ID") == player_id), None
            )

            if not player_row or "USG_PCT" not in (player_row or {}):
                return 0.0

            player_usg = float(player_row["USG_PCT"] or 0)

            injured_lower = {n.lower() for n in injured_players}
            active_teammates = [
                p for p in team_players
                if p.get("PLAYER_ID") != player_id
                and (p.get("PLAYER_NAME") or "").lower() not in injured_lower
            ]
            out_team_players = [
                p for p in team_players
                if (p.get("PLAYER_NAME") or "").lower() in injured_lower
            ]

            total_out_usg = sum(float(p.get("USG_PCT") or 0) for p in out_team_players)
            if total_out_usg == 0:
                return 0.0

            active_usg_total = sum(float(p.get("USG_PCT") or 0) for p in active_teammates)
            if active_usg_total == 0:
                return 0.0

            player_share = player_usg / active_usg_total
            usage_boost = player_share * total_out_usg

            if usage_boost < -0.05:
                return -0.012
            if usage_boost > 0.05:
                return 0.012
            return 0.0

        except Exception as exc:
            logger.warning("_calculate_usage_adjustment failed: %s", exc)
            return 0.0

    async def _get_opening_prop_lines(
        self,
        event_id: str,
        commence_time: Optional[datetime],
        markets: list[str],
    ) -> dict[str, dict[str, float]]:
        """Return {player_name: {market: opening_line}} from 48h-before snapshot.
        NOTE: Disabled — historical API returns EVENT_NOT_FOUND for concluded games.
        """
        return {}

        # if commence_time is None or not markets:
        #     return {}

        # opening_ts = (commence_time - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        # try:
        #     hist = await self.odds.get_historical_event_odds(event_id, opening_ts, markets=markets)
        # except Exception as exc:
        #     logger.debug("Could not fetch opening prop lines: %s", exc)
        #     return {}

        # hist_data = hist.get("data") or {}
        # result: dict[str, dict[str, float]] = {}

        # for bookmaker in hist_data.get("bookmakers", []):
        #     for market in bookmaker.get("markets", []):
        #         market_key = market.get("key", "")
        #         for outcome in market.get("outcomes", []):
        #             player_name = outcome.get("description", outcome.get("name", ""))
        #             if not player_name or "point" not in outcome:
        #                 continue
        #             result.setdefault(player_name, {})
        #             if market_key not in result[player_name]:
        #                 result[player_name][market_key] = float(outcome["point"])

        # return result

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
