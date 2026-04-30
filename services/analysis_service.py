from __future__ import annotations

import logging
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from cache.ttl_cache import STATS
from config import settings
from models.schemas import ArbOpportunity, ValueBet
from services.odds_service import OddsService
from services.stats_service import StatsService

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 0.01  # 1% minimum edge to flag a value bet

# Prop odds filters: hard cutoff = data error (skip), soft = suspicious (flag with ⚠)
_MAX_PROP_ODDS = 1000   # skip entirely — no legitimate prop line is ever > +1000
_WARN_PROP_ODDS = 500   # flag with ⚠ — unusual but possible for deep longshots

# Books that share the same operator — arbing between them is impossible
_BOOK_GROUP: dict[str, str] = {
    "hardrockbet_fl": "hardrockbet",
    "hardrockbet_az": "hardrockbet",
    "lowvig": "betonlineag",
}


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

    def no_vig_prob(self, price_a: int, price_b: int) -> tuple[float, float]:
        """Strip vig from a two-sided market. Returns (prob_a, prob_b) summing to 1.0."""
        imp_a = self.implied_probability(price_a)
        imp_b = self.implied_probability(price_b)
        total = imp_a + imp_b
        return imp_a / total, imp_b / total

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

    def _canonical_book(self, book: str) -> str:
        """Normalize regional book variants to their parent operator."""
        return _BOOK_GROUP.get(book, book)

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
        """Scan upcoming games for moneyline value using Pinnacle no-vig as true probability."""
        # Pass game_date only when explicitly provided — None fetches all upcoming games
        games = await self.odds.get_games(game_date=game_date)
        value_bets: list[ValueBet] = []

        for game in games:
            try:
                lines = await self.odds.get_odds(game.id, markets=["h2h"])
            except Exception as exc:
                logger.warning("Could not fetch odds for %s: %s", game.id, exc)
                continue

            home_book_odds: list[tuple[str, int]] = []  # (bookmaker, price)
            away_book_odds: list[tuple[str, int]] = []
            for line in lines:
                if line.moneyline_home is not None:
                    home_book_odds.append((line.bookmaker, line.moneyline_home))
                if line.moneyline_away is not None:
                    away_book_odds.append((line.bookmaker, line.moneyline_away))

            if not home_book_odds or not away_book_odds:
                continue

            best_home_book, best_home = max(home_book_odds, key=lambda x: self.american_to_decimal(x[1]))
            best_away_book, best_away = max(away_book_odds, key=lambda x: self.american_to_decimal(x[1]))

            implied_home = self.implied_probability(best_home)
            implied_away = self.implied_probability(best_away)

            # --- True probability: Pinnacle no-vig (falls back to team win%) ---
            home_true: Optional[float] = None
            away_true: Optional[float] = None
            try:
                pinnacle_line = await self.odds.get_pinnacle_odds(game.id, markets=["h2h"])
                if (
                    pinnacle_line
                    and pinnacle_line.moneyline_home is not None
                    and pinnacle_line.moneyline_away is not None
                ):
                    home_true, away_true = self.no_vig_prob(
                        pinnacle_line.moneyline_home, pinnacle_line.moneyline_away
                    )
            except Exception as exc:
                logger.warning("Could not fetch Pinnacle line for %s: %s", game.id, exc)

            if home_true is None or away_true is None:
                # Fallback: normalize team win rates
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
                total = home_stats.win_pct + away_stats.win_pct
                if total > 0:
                    home_true = home_stats.win_pct / total
                    away_true = away_stats.win_pct / total
                else:
                    continue

            for team, true_prob, implied_prob, american_odds, bookmaker in [
                (game.home_team, home_true, implied_home, best_home, best_home_book),
                (game.away_team, away_true, implied_away, best_away, best_away_book),
            ]:
                edge = self.calculate_edge(true_prob, implied_prob)
                if edge >= EDGE_THRESHOLD:
                    decimal = self.american_to_decimal(american_odds)
                    value_bets.append(
                        ValueBet(
                            event_id=game.id,
                            market="h2h",
                            selection=team,
                            bookmaker=bookmaker,
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
    # Arb & Positive EV scanners
    # ------------------------------------------------------------------

    async def find_arb(self, min_arb_pct: float = 0.001) -> list[ArbOpportunity]:
        """Scan all upcoming games for arbitrage across all books and markets."""
        games = await self.odds.get_games()
        arbs: list[ArbOpportunity] = []
        for game in games:
            try:
                lines = await self.odds.get_odds(game.id)
                arbs.extend(self._scan_game_lines_for_arb(game, lines, min_arb_pct))
            except Exception as exc:
                logger.warning("Arb scan (game lines) failed for %s: %s", game.id, exc)
            try:
                props = await self.odds.get_player_props(game.id)
                arbs.extend(self._scan_props_for_arb(game, props, min_arb_pct))
            except Exception as exc:
                logger.warning("Arb scan (props) failed for %s: %s", game.id, exc)
        arbs.sort(key=lambda a: a.arb_pct, reverse=True)
        return arbs

    def _scan_game_lines_for_arb(
        self, game: Any, lines: list, min_arb_pct: float
    ) -> list[ArbOpportunity]:
        arbs: list[ArbOpportunity] = []

        # --- h2h ---
        best_home: dict = {}  # bookmaker -> price
        best_away: dict = {}
        for line in lines:
            if line.moneyline_home is not None:
                if not best_home or self.american_to_decimal(line.moneyline_home) > self.american_to_decimal(best_home["price"]):
                    best_home = {"book": line.bookmaker, "price": line.moneyline_home}
            if line.moneyline_away is not None:
                if not best_away or self.american_to_decimal(line.moneyline_away) > self.american_to_decimal(best_away["price"]):
                    best_away = {"book": line.bookmaker, "price": line.moneyline_away}
        if best_home and best_away and self._canonical_book(best_home["book"]) != self._canonical_book(best_away["book"]):
            arb = self._calc_arb(best_home["price"], best_away["price"])
            if arb and arb >= min_arb_pct:
                da = self.american_to_decimal(best_home["price"])
                db = self.american_to_decimal(best_away["price"])
                total_imp = 1/da + 1/db
                stake_a = round(100 / (da * total_imp), 2)
                arbs.append(ArbOpportunity(
                    event_id=game.id, market="h2h",
                    home_team=game.home_team, away_team=game.away_team,
                    side_a=game.home_team, side_a_book=best_home["book"], side_a_odds=best_home["price"],
                    side_b=game.away_team, side_b_book=best_away["book"], side_b_odds=best_away["price"],
                    arb_pct=round(arb, 4),
                    side_a_stake=stake_a, side_b_stake=round(100 - stake_a, 2),
                ))

        # --- spreads (group by both side points — only compare same-line prices) ---
        from collections import defaultdict
        spread_by_point: dict[tuple[float, float], list] = defaultdict(list)
        for line in lines:
            if line.spread is not None:
                spread_by_point[
                    (line.spread.home_point, line.spread.away_point)
                ].append(line)
        for (home_point, away_point), grp in spread_by_point.items():
            best_home_sp: dict = {}
            best_away_sp: dict = {}
            for line in grp:
                if not best_home_sp or self.american_to_decimal(line.spread.home_price) > self.american_to_decimal(best_home_sp["price"]):
                    best_home_sp = {"book": line.bookmaker, "price": line.spread.home_price}
                if not best_away_sp or self.american_to_decimal(line.spread.away_price) > self.american_to_decimal(best_away_sp["price"]):
                    best_away_sp = {"book": line.bookmaker, "price": line.spread.away_price}
            if best_home_sp and best_away_sp and self._canonical_book(best_home_sp["book"]) != self._canonical_book(best_away_sp["book"]):
                arb = self._calc_arb(best_home_sp["price"], best_away_sp["price"])
                if arb and arb >= min_arb_pct:
                    da = self.american_to_decimal(best_home_sp["price"])
                    db = self.american_to_decimal(best_away_sp["price"])
                    total_imp = 1/da + 1/db
                    stake_a = round(100 / (da * total_imp), 2)
                    arbs.append(ArbOpportunity(
                        event_id=game.id, market="spreads",
                        home_team=game.home_team, away_team=game.away_team,
                        side_a=f"{game.home_team} {home_point:+.1f}",
                        side_a_book=best_home_sp["book"], side_a_odds=best_home_sp["price"],
                        side_b=f"{game.away_team} {away_point:+.1f}",
                        side_b_book=best_away_sp["book"], side_b_odds=best_away_sp["price"],
                        arb_pct=round(arb, 4),
                        side_a_stake=stake_a, side_b_stake=round(100 - stake_a, 2),
                    ))

        # --- totals (group by total — only compare over/under at the same number) ---
        totals_by_point: dict[float, list] = defaultdict(list)
        for line in lines:
            if line.total is not None:
                totals_by_point[line.total.point].append(line)
        for point, grp in totals_by_point.items():
            best_over: dict = {}
            best_under: dict = {}
            for line in grp:
                if not best_over or self.american_to_decimal(line.total.over_price) > self.american_to_decimal(best_over["price"]):
                    best_over = {"book": line.bookmaker, "price": line.total.over_price}
                if not best_under or self.american_to_decimal(line.total.under_price) > self.american_to_decimal(best_under["price"]):
                    best_under = {"book": line.bookmaker, "price": line.total.under_price}
            if best_over and best_under and self._canonical_book(best_over["book"]) != self._canonical_book(best_under["book"]):
                arb = self._calc_arb(best_over["price"], best_under["price"])
                if arb and arb >= min_arb_pct:
                    da = self.american_to_decimal(best_over["price"])
                    db = self.american_to_decimal(best_under["price"])
                    total_imp = 1/da + 1/db
                    stake_a = round(100 / (da * total_imp), 2)
                    arbs.append(ArbOpportunity(
                        event_id=game.id, market="totals",
                        home_team=game.home_team, away_team=game.away_team,
                        side_a=f"Over {point}", side_a_book=best_over["book"], side_a_odds=best_over["price"],
                        side_b=f"Under {point}", side_b_book=best_under["book"], side_b_odds=best_under["price"],
                        arb_pct=round(arb, 4),
                        side_a_stake=stake_a, side_b_stake=round(100 - stake_a, 2),
                    ))
        return arbs

    def _scan_props_for_arb(
        self, game: Any, props: list, min_arb_pct: float
    ) -> list[ArbOpportunity]:
        """Group props by (player, market, line), find best over/under across books."""
        arbs: list[ArbOpportunity] = []
        # Group by (player_name, market, line)
        groups: dict[tuple, dict] = {}
        for p in props:
            if p.over_price > _MAX_PROP_ODDS or p.under_price > _MAX_PROP_ODDS:
                continue
            key = (p.player_name, p.market, p.line)
            groups.setdefault(key, {"best_over": None, "best_under": None})
            if groups[key]["best_over"] is None or self.american_to_decimal(p.over_price) > self.american_to_decimal(groups[key]["best_over"]["price"]):
                groups[key]["best_over"] = {"book": p.bookmaker, "price": p.over_price}
            if groups[key]["best_under"] is None or self.american_to_decimal(p.under_price) > self.american_to_decimal(groups[key]["best_under"]["price"]):
                groups[key]["best_under"] = {"book": p.bookmaker, "price": p.under_price}

        for (player, market, line), sides in groups.items():
            bo = sides["best_over"]
            bu = sides["best_under"]
            if not bo or not bu or self._canonical_book(bo["book"]) == self._canonical_book(bu["book"]):
                continue
            arb = self._calc_arb(bo["price"], bu["price"])
            if arb and arb >= min_arb_pct:
                da = self.american_to_decimal(bo["price"])
                db = self.american_to_decimal(bu["price"])
                total_imp = 1/da + 1/db
                stake_a = round(100 / (da * total_imp), 2)
                over_flag = " ⚠" if _WARN_PROP_ODDS < bo["price"] <= _MAX_PROP_ODDS else ""
                under_flag = " ⚠" if _WARN_PROP_ODDS < bu["price"] <= _MAX_PROP_ODDS else ""
                arbs.append(ArbOpportunity(
                    event_id=game.id, market=market,
                    home_team=game.home_team, away_team=game.away_team,
                    side_a=f"{player} Over {line}{over_flag}", side_a_book=bo["book"], side_a_odds=bo["price"],
                    side_b=f"{player} Under {line}{under_flag}", side_b_book=bu["book"], side_b_odds=bu["price"],
                    arb_pct=round(arb, 4),
                    side_a_stake=stake_a, side_b_stake=round(100 - stake_a, 2),
                ))
        return arbs

    def _calc_arb(self, price_a: int, price_b: int) -> Optional[float]:
        """Returns arb profit % if exists, else None."""
        da = self.american_to_decimal(price_a)
        db = self.american_to_decimal(price_b)
        total_implied = 1/da + 1/db
        if total_implied < 1.0:
            return 1.0 - total_implied
        return None

    async def find_positive_ev(self, min_edge: float = 0.01) -> list[ValueBet]:
        """Scan all upcoming games for +EV bets using Pinnacle no-vig as true probability.
        Covers game lines (h2h, spreads, totals) and player props.
        """
        games = await self.odds.get_games()
        ev_bets: list[ValueBet] = []

        for game in games:
            # --- Game lines ---
            try:
                pinnacle = await self.odds.get_pinnacle_odds(game.id)
                if pinnacle:
                    lines = await self.odds.get_odds(game.id)
                    ev_bets.extend(self._scan_game_ev(game, lines, pinnacle, min_edge))
            except Exception as exc:
                logger.warning("+EV scan (game lines) failed for %s: %s", game.id, exc)

            # --- Player props ---
            try:
                pinnacle_props = await self.odds.get_pinnacle_props(game.id)
                if pinnacle_props:
                    props = await self.odds.get_player_props(game.id)
                    ev_bets.extend(self._scan_props_ev(game, props, pinnacle_props, min_edge))
            except Exception as exc:
                logger.warning("+EV scan (props) failed for %s: %s", game.id, exc)

        ev_bets.sort(key=lambda b: b.edge, reverse=True)
        return ev_bets

    def _scan_game_ev(
        self, game: Any, lines: list, pinnacle: Any, min_edge: float
    ) -> list[ValueBet]:
        bets: list[ValueBet] = []

        # h2h
        if pinnacle.moneyline_home and pinnacle.moneyline_away:
            true_home, true_away = self.no_vig_prob(pinnacle.moneyline_home, pinnacle.moneyline_away)
            for line in lines:
                if line.bookmaker == "pinnacle":
                    continue
                for team, true_prob, odds in [
                    (game.home_team, true_home, line.moneyline_home),
                    (game.away_team, true_away, line.moneyline_away),
                ]:
                    if odds is None:
                        continue
                    imp = self.implied_probability(odds)
                    edge = true_prob - imp
                    if edge >= min_edge:
                        dec = self.american_to_decimal(odds)
                        bets.append(ValueBet(
                            event_id=game.id, market="h2h", selection=team,
                            bookmaker=line.bookmaker, american_odds=odds,
                            implied_probability=round(imp, 4),
                            true_probability=round(true_prob, 4),
                            edge=round(edge, 4),
                            kelly_fraction=self.kelly_criterion(edge, dec),
                            confidence=self._confidence(edge),
                        ))

        # spreads
        if pinnacle.spread:
            pinn_home_p, pinn_away_p = self.no_vig_prob(pinnacle.spread.home_price, pinnacle.spread.away_price)
            for line in lines:
                if line.bookmaker == "pinnacle" or not line.spread:
                    continue
                if (
                    line.spread.home_point != pinnacle.spread.home_point
                    or line.spread.away_point != pinnacle.spread.away_point
                ):
                    continue
                for team, true_prob, odds, point in [
                    (game.home_team, pinn_home_p, line.spread.home_price, line.spread.home_point),
                    (game.away_team, pinn_away_p, line.spread.away_price, line.spread.away_point),
                ]:
                    imp = self.implied_probability(odds)
                    edge = true_prob - imp
                    if edge >= min_edge:
                        dec = self.american_to_decimal(odds)
                        bets.append(ValueBet(
                            event_id=game.id, market="spreads",
                            selection=f"{team} {point:+.1f}",
                            bookmaker=line.bookmaker, american_odds=odds,
                            implied_probability=round(imp, 4),
                            true_probability=round(true_prob, 4),
                            edge=round(edge, 4),
                            kelly_fraction=self.kelly_criterion(edge, dec),
                            confidence=self._confidence(edge),
                        ))

        # totals
        if pinnacle.total:
            pinn_over_p, pinn_under_p = self.no_vig_prob(pinnacle.total.over_price, pinnacle.total.under_price)
            for line in lines:
                if line.bookmaker == "pinnacle" or not line.total:
                    continue
                if line.total.point != pinnacle.total.point:
                    continue
                for side, true_prob, odds in [
                    ("Over", pinn_over_p, line.total.over_price),
                    ("Under", pinn_under_p, line.total.under_price),
                ]:
                    imp = self.implied_probability(odds)
                    edge = true_prob - imp
                    if edge >= min_edge:
                        dec = self.american_to_decimal(odds)
                        bets.append(ValueBet(
                            event_id=game.id, market="totals",
                            selection=f"{side} {pinnacle.total.point}",
                            bookmaker=line.bookmaker, american_odds=odds,
                            implied_probability=round(imp, 4),
                            true_probability=round(true_prob, 4),
                            edge=round(edge, 4),
                            kelly_fraction=self.kelly_criterion(edge, dec),
                            confidence=self._confidence(edge),
                        ))
        return bets

    def _scan_props_ev(
        self, game: Any, props: list, pinnacle_props: list, min_edge: float
    ) -> list[ValueBet]:
        # Build Pinnacle no-vig map keyed by (player_name, market, line)
        pinn_map: dict[tuple, tuple[float, float]] = {}
        for pp in pinnacle_props:
            if pp.over_price > _MAX_PROP_ODDS or pp.under_price > _MAX_PROP_ODDS:
                continue
            true_over, true_under = self.no_vig_prob(pp.over_price, pp.under_price)
            pinn_map[(pp.player_name, pp.market, pp.line)] = (true_over, true_under)

        bets: list[ValueBet] = []
        for p in props:
            if p.bookmaker == "pinnacle":
                continue
            if p.over_price > _MAX_PROP_ODDS or p.under_price > _MAX_PROP_ODDS:
                continue
            key = (p.player_name, p.market, p.line)
            if key not in pinn_map:
                continue
            true_over, true_under = pinn_map[key]
            for side, true_prob, odds in [
                ("Over", true_over, p.over_price),
                ("Under", true_under, p.under_price),
            ]:
                imp = self.implied_probability(odds)
                edge = true_prob - imp
                if edge >= min_edge:
                    dec = self.american_to_decimal(odds)
                    flag = " ⚠" if _WARN_PROP_ODDS < odds <= _MAX_PROP_ODDS else ""
                    bets.append(ValueBet(
                        event_id=game.id,
                        market=p.market,
                        selection=f"{p.player_name} {side} {p.line}{flag}",
                        bookmaker=p.bookmaker,
                        american_odds=odds,
                        implied_probability=round(imp, 4),
                        true_probability=round(true_prob, 4),
                        edge=round(edge, 4),
                        kelly_fraction=self.kelly_criterion(edge, dec),
                        confidence=self._confidence(edge),
                    ))
        return bets

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

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Approximation of the standard normal CDF."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def datetime_from_date(d: date):
    from datetime import datetime

    return datetime(d.year, d.month, d.day, 19, 0)  # assume 7 PM tip-off
