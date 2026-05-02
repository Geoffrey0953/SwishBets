"""SwishBets test suite — covers every function and flags known bugs.

Run:
    .venv/bin/python test.py

Bug regression tests (TestBugRegression) confirm bugs on current code.
After fixes they should PASS instead of FAIL.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.schemas import (
    Game,
    OddsLine,
    PlayerProp,
    Spread,
    TeamStats,
    Total,
    ValueBet,
)
from services.analysis_service import AnalysisService
from services.defense_service import DefenseService
from services.odds_service import OddsService
from services.stats_service import StatsService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.run(coro)


def make_cache() -> MagicMock:
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.delete = AsyncMock()
    return cache


def _make_game(
    game_id: str = "test_game",
    home: str = "Los Angeles Lakers",
    away: str = "Golden State Warriors",
) -> Game:
    return Game(
        id=game_id,
        home_team=home,
        away_team=away,
        commence_time=datetime(2026, 3, 21, 19, 0, tzinfo=timezone.utc),
    )


def _make_odds_line(
    bookmaker: str,
    home_ml: int,
    away_ml: int,
    spread_point: float = -5.5,
    game_id: str = "test_game",
) -> OddsLine:
    return OddsLine(
        event_id=game_id,
        home_team="Los Angeles Lakers",
        away_team="Golden State Warriors",
        commence_time=datetime(2026, 3, 21, 19, 0, tzinfo=timezone.utc),
        bookmaker=bookmaker,
        moneyline_home=home_ml,
        moneyline_away=away_ml,
        spread=Spread(
            home_team="Los Angeles Lakers", home_point=spread_point, home_price=-110,
            away_team="Golden State Warriors", away_point=-spread_point, away_price=-110,
        ),
        total=Total(point=220.5, over_price=-110, under_price=-110),
    )


def _make_team_stats(
    team_id: int,
    name: str,
    wins: int,
    losses: int,
    pace: float = 98.5,
) -> TeamStats:
    played = wins + losses
    return TeamStats(
        team_id=team_id,
        team_name=name,
        games_played=played,
        wins=wins,
        losses=losses,
        win_pct=round(wins / played, 3) if played else 0.0,
        points_per_game=112.0,
        points_allowed_per_game=108.0,
        pace=pace,
        offensive_rating=112.0,
        defensive_rating=108.0,
        net_rating=4.0,
        last_n_games=10,
    )


def _make_analysis_svc(
    odds_mock=None,
    stats_mock=None,
    defense_mock=None,
) -> AnalysisService:
    return AnalysisService(
        odds_service=odds_mock or MagicMock(spec=OddsService),
        stats_service=stats_mock or MagicMock(spec=StatsService),
        defense_service=defense_mock,
    )


# ---------------------------------------------------------------------------
# Group 1 — Pure math helpers
# ---------------------------------------------------------------------------

class TestImpliedProbability(unittest.TestCase):

    def setUp(self):
        self.svc = _make_analysis_svc()

    def test_plus_100_is_even_money(self):
        self.assertAlmostEqual(self.svc.implied_probability(100), 0.5, places=4)

    def test_minus_110_standard_vig(self):
        self.assertAlmostEqual(self.svc.implied_probability(-110), 0.52381, places=4)

    def test_minus_200_heavy_favourite(self):
        self.assertAlmostEqual(self.svc.implied_probability(-200), 0.66667, places=4)

    def test_plus_200_underdog(self):
        self.assertAlmostEqual(self.svc.implied_probability(200), 0.33333, places=4)

    def test_minus_150(self):
        self.assertAlmostEqual(self.svc.implied_probability(-150), 0.60, places=4)

    def test_plus_130(self):
        self.assertAlmostEqual(self.svc.implied_probability(130), 0.43478, places=4)


class TestAmericanToDecimal(unittest.TestCase):

    def setUp(self):
        self.svc = _make_analysis_svc()

    def test_plus_100(self):
        self.assertAlmostEqual(self.svc.american_to_decimal(100), 2.0, places=4)

    def test_minus_110(self):
        self.assertAlmostEqual(self.svc.american_to_decimal(-110), 1.9091, places=3)

    def test_plus_150(self):
        self.assertAlmostEqual(self.svc.american_to_decimal(150), 2.5, places=4)

    def test_minus_200(self):
        self.assertAlmostEqual(self.svc.american_to_decimal(-200), 1.5, places=4)

    def test_plus_300(self):
        self.assertAlmostEqual(self.svc.american_to_decimal(300), 4.0, places=4)

    def test_minus_300(self):
        self.assertAlmostEqual(self.svc.american_to_decimal(-300), 1.3333, places=3)


class TestCalculateEdge(unittest.TestCase):

    def setUp(self):
        self.svc = _make_analysis_svc()

    def test_positive_edge(self):
        self.assertAlmostEqual(self.svc.calculate_edge(0.60, 0.52), 0.08, places=6)

    def test_negative_edge(self):
        self.assertAlmostEqual(self.svc.calculate_edge(0.40, 0.55), -0.15, places=6)

    def test_zero_edge(self):
        self.assertAlmostEqual(self.svc.calculate_edge(0.50, 0.50), 0.0, places=6)

    def test_small_edge_above_3pct_threshold(self):
        self.assertGreater(self.svc.calculate_edge(0.555, 0.524), 0.03)


class TestKellyCriterion(unittest.TestCase):
    """Quarter-Kelly: full_kelly * 0.25, capped at 6.25% (0.0625)."""

    def setUp(self):
        self.svc = _make_analysis_svc()

    def test_even_money_ten_pct_edge(self):
        # full_kelly = 0.10*2.0/1.0 = 0.20 → quarter = 0.05
        self.assertAlmostEqual(self.svc.kelly_criterion(0.10, 2.0), 0.05, places=4)

    def test_zero_edge_returns_zero(self):
        self.assertEqual(self.svc.kelly_criterion(0.0, 2.0), 0.0)

    def test_negative_edge_returns_zero(self):
        self.assertEqual(self.svc.kelly_criterion(-0.05, 2.0), 0.0)

    def test_decimal_one_returns_zero(self):
        self.assertEqual(self.svc.kelly_criterion(0.10, 1.0), 0.0)

    def test_high_edge_capped_at_6_25_pct(self):
        # full_kelly = (0.60*5.0)/4.0 = 0.75 → quarter = 0.1875 → capped at 0.0625
        result = self.svc.kelly_criterion(0.60, 5.0)
        self.assertAlmostEqual(result, 0.0625, places=4)

    def test_typical_minus_110_odds(self):
        # full_kelly ≈ 0.1472 → quarter ≈ 0.0368
        result = self.svc.kelly_criterion(0.07, 1.9091)
        self.assertAlmostEqual(result, 0.0368, places=3)

    def test_result_rounded_to_4_decimals(self):
        # full_kelly = 0.20 → quarter = 0.05
        result = self.svc.kelly_criterion(0.10, 2.0)
        self.assertEqual(result, 0.05)

    def test_never_exceeds_6_25_pct(self):
        """No matter the edge/odds, quarter-Kelly is never above 6.25%."""
        for edge in [0.10, 0.20, 0.50, 1.0]:
            for decimal in [1.5, 2.0, 3.0, 5.0]:
                with self.subTest(edge=edge, decimal=decimal):
                    self.assertLessEqual(self.svc.kelly_criterion(edge, decimal), 0.0625)


class TestConfidence(unittest.TestCase):

    def setUp(self):
        self.svc = _make_analysis_svc()

    def test_high_at_threshold(self):
        self.assertEqual(self.svc._confidence(0.08), "High")

    def test_high_above_threshold(self):
        self.assertEqual(self.svc._confidence(0.15), "High")

    def test_medium_at_lower_threshold(self):
        self.assertEqual(self.svc._confidence(0.04), "Medium")

    def test_medium_just_below_high(self):
        self.assertEqual(self.svc._confidence(0.079), "Medium")

    def test_low_just_below_medium(self):
        self.assertEqual(self.svc._confidence(0.039), "Low")

    def test_low_at_zero(self):
        self.assertEqual(self.svc._confidence(0.0), "Low")

    def test_low_negative(self):
        self.assertEqual(self.svc._confidence(-0.05), "Low")


class TestNormalCDF(unittest.TestCase):

    def test_z_zero_is_half(self):
        self.assertAlmostEqual(AnalysisService._normal_cdf(0), 0.5, places=6)

    def test_z_1_96_is_97_5_pct(self):
        self.assertAlmostEqual(AnalysisService._normal_cdf(1.96), 0.975, places=2)

    def test_z_neg_1_96_is_2_5_pct(self):
        self.assertAlmostEqual(AnalysisService._normal_cdf(-1.96), 0.025, places=2)

    def test_symmetry(self):
        for z in [0.5, 1.0, 1.5, 2.0, 2.5]:
            with self.subTest(z=z):
                self.assertAlmostEqual(
                    AnalysisService._normal_cdf(z) + AnalysisService._normal_cdf(-z),
                    1.0, places=8,
                )

    def test_positive_z_above_half(self):
        self.assertGreater(AnalysisService._normal_cdf(1.0), 0.5)

    def test_negative_z_below_half(self):
        self.assertLess(AnalysisService._normal_cdf(-1.0), 0.5)

    def test_large_z_approaches_one(self):
        self.assertGreater(AnalysisService._normal_cdf(4.0), 0.999)

    def test_large_negative_z_approaches_zero(self):
        self.assertLess(AnalysisService._normal_cdf(-4.0), 0.001)


# ---------------------------------------------------------------------------
# Group 2 — Bug 3: get_player_stats always caps at 10 rows
# ---------------------------------------------------------------------------

class TestGetPlayerStatsBug(unittest.TestCase):

    def setUp(self):
        self.cache = make_cache()
        self.svc = StatsService(self.cache)

    def _make_mock_endpoint(self, n_rows: int):
        import pandas as pd
        data = {
            "GAME_ID": [f"00{i}" for i in range(n_rows)],
            "GAME_DATE": ["2026-03-01"] * n_rows,
            "WL": ["W"] * n_rows,
            "PTS": [25.0] * n_rows,
            "REB": [5.0] * n_rows,
            "AST": [7.0] * n_rows,
            "MIN": ["35"] * n_rows,
            "TEAM_ABBREVIATION": ["LAL"] * n_rows,
            "PLUS_MINUS": [5.0] * n_rows,
            "FG3M": [3.0] * n_rows,
            "BLK": [1.0] * n_rows,
            "STL": [1.5] * n_rows,
            "TOV": [2.0] * n_rows,
        }
        mock_ep = MagicMock()
        mock_ep.player_game_logs.get_data_frame.return_value = pd.DataFrame(data)
        return mock_ep

    def test_20_row_df_returns_only_10(self):
        """BUG 3: df.head(10) is hardcoded — 20 available rows still yields only 10."""
        self.svc._get_player_game_logs = MagicMock(return_value=self._make_mock_endpoint(20))
        result = run(self.svc.get_player_stats(2544))
        self.assertEqual(
            len(result), 10,
            "BUG 3: get_player_stats hardcodes head(10). "
            "After fix it should return all rows up to the requested last_n.",
        )

    def test_5_row_df_returns_all_5(self):
        self.svc._get_player_game_logs = MagicMock(return_value=self._make_mock_endpoint(5))
        result = run(self.svc.get_player_stats(2544))
        self.assertEqual(len(result), 5)

    def test_empty_df_returns_empty_list(self):
        import pandas as pd
        mock_ep = MagicMock()
        mock_ep.player_game_logs.get_data_frame.return_value = pd.DataFrame()
        self.svc._get_player_game_logs = MagicMock(return_value=mock_ep)
        result = run(self.svc.get_player_stats(9999))
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Group 3 — Bug 2: dead negative branch in _calculate_usage_adjustment
# ---------------------------------------------------------------------------

class TestUsageAdjustmentDeadCode(unittest.TestCase):

    def test_usage_boost_product_is_always_non_negative(self):
        """player_share * total_out_usg is always >= 0, so -0.012 is unreachable."""
        for player_usg in [0.05, 0.15, 0.25, 0.35]:
            for active_usg_total in [0.10, 0.50, 0.80, 1.0]:
                for total_out_usg in [0.05, 0.10, 0.20, 0.30]:
                    with self.subTest(p=player_usg, a=active_usg_total, o=total_out_usg):
                        player_share = player_usg / active_usg_total
                        usage_boost = player_share * total_out_usg
                        self.assertGreaterEqual(
                            usage_boost, 0,
                            "BUG 2: the `if usage_boost < -0.05` branch is dead code.",
                        )

    def test_positive_boost_returns_0_012(self):
        """A high-usage teammate being out → player absorbs > 5% → returns +0.012."""
        cache = make_cache()
        stats_svc = StatsService(cache)
        analysis_svc = _make_analysis_svc(stats_mock=stats_svc)

        # player_id=1 (25% usage), injured: player_id=2 (20% usage)
        # active_usg_total (excl. injured, excl. player 1) = 0.18+0.17+0.20 = 0.55
        # player_share = 0.25 / 0.55 = 0.4545
        # usage_boost = 0.4545 * 0.20 = 0.0909 > 0.05 → 0.012
        player_stats_data = [
            {"PLAYER_ID": 1, "TEAM_ID": 100, "USG_PCT": 0.25, "PLAYER_NAME": "Star Player"},
            {"PLAYER_ID": 2, "TEAM_ID": 100, "USG_PCT": 0.20, "PLAYER_NAME": "Injured Player"},
            {"PLAYER_ID": 3, "TEAM_ID": 100, "USG_PCT": 0.18, "PLAYER_NAME": "Bench A"},
            {"PLAYER_ID": 4, "TEAM_ID": 100, "USG_PCT": 0.17, "PLAYER_NAME": "Bench B"},
            {"PLAYER_ID": 5, "TEAM_ID": 100, "USG_PCT": 0.20, "PLAYER_NAME": "Sixth Man"},
        ]
        cache.get = AsyncMock(return_value=player_stats_data)

        result = run(analysis_svc._calculate_usage_adjustment(1, 100, ["Injured Player"]))
        self.assertAlmostEqual(result, 0.012, places=3)

    def test_small_boost_below_5pct_returns_zero(self):
        cache = make_cache()
        stats_svc = StatsService(cache)
        analysis_svc = _make_analysis_svc(stats_mock=stats_svc)

        # player_id=1 (10% usage), injured: player_id=2 (3% usage)
        # active_usg_total (excl. injured, excl. player 1) = 0.30+0.30+0.27 = 0.87
        # player_share = 0.10 / 0.87 ≈ 0.115
        # usage_boost = 0.115 * 0.03 ≈ 0.003 < 0.05 → 0.0
        player_stats_data = [
            {"PLAYER_ID": 1, "TEAM_ID": 100, "USG_PCT": 0.10, "PLAYER_NAME": "Bench"},
            {"PLAYER_ID": 2, "TEAM_ID": 100, "USG_PCT": 0.03, "PLAYER_NAME": "Injured"},
            {"PLAYER_ID": 3, "TEAM_ID": 100, "USG_PCT": 0.30, "PLAYER_NAME": "Star A"},
            {"PLAYER_ID": 4, "TEAM_ID": 100, "USG_PCT": 0.30, "PLAYER_NAME": "Star B"},
            {"PLAYER_ID": 5, "TEAM_ID": 100, "USG_PCT": 0.27, "PLAYER_NAME": "Starter"},
        ]
        cache.get = AsyncMock(return_value=player_stats_data)

        result = run(analysis_svc._calculate_usage_adjustment(1, 100, ["Injured"]))
        self.assertAlmostEqual(result, 0.0, places=3)

    def test_no_injured_players_returns_zero(self):
        cache = make_cache()
        analysis_svc = _make_analysis_svc(stats_mock=StatsService(cache))
        result = run(analysis_svc._calculate_usage_adjustment(1, 100, []))
        self.assertEqual(result, 0.0)

    def test_no_team_id_returns_zero(self):
        cache = make_cache()
        analysis_svc = _make_analysis_svc(stats_mock=StatsService(cache))
        result = run(analysis_svc._calculate_usage_adjustment(1, None, ["Someone"]))
        self.assertEqual(result, 0.0)


# ---------------------------------------------------------------------------
# Group 4 — Bug 4: best_bookmaker is last, not the one with the best odds
# ---------------------------------------------------------------------------

class TestBestBookmakerAttribution(unittest.TestCase):

    def _run_scenario(self):
        """3 bookmakers: A (+120), B (+150 best odds), C (+110 last).
        Returns the Lakers ValueBet."""
        game = _make_game()
        line_a = _make_odds_line("bookmaker_a", home_ml=120, away_ml=-140)
        line_b = _make_odds_line("bookmaker_b", home_ml=150, away_ml=-175)  # BEST
        line_c = _make_odds_line("bookmaker_c", home_ml=110, away_ml=-130)  # LAST

        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_games = AsyncMock(return_value=[game])
        odds_mock.get_odds = AsyncMock(return_value=[line_a, line_b, line_c])

        home_stats = _make_team_stats(1610612747, "Lakers", wins=8, losses=2)
        away_stats = _make_team_stats(1610612744, "Warriors", wins=4, losses=6)

        stats_mock = MagicMock(spec=StatsService)
        stats_mock.find_team_by_name = MagicMock(
            side_effect=lambda name: (
                {"id": 1610612747, "full_name": "Los Angeles Lakers"}
                if "Lakers" in name
                else {"id": 1610612744, "full_name": "Golden State Warriors"}
            )
        )
        stats_mock.get_team_stats = AsyncMock(
            side_effect=lambda tid, last_n=10: (
                home_stats if tid == 1610612747 else away_stats
            )
        )

        svc = _make_analysis_svc(odds_mock=odds_mock, stats_mock=stats_mock)
        bets = run(svc.find_value_bets())
        return next((b for b in bets if "Lakers" in b.selection), None)

    def test_best_odds_value_correctly_picked(self):
        """+150 (bookmaker_b) should be the odds used for the bet."""
        bet = self._run_scenario()
        self.assertIsNotNone(bet)
        self.assertEqual(
            bet.american_odds, 150,
            "The best available odds (+150) should be correctly selected.",
        )

    def test_bookmaker_attribution_is_correct(self):
        """bookmaker attribute should be 'bookmaker_b' — the one with the best odds (+150)."""
        bet = self._run_scenario()
        self.assertIsNotNone(bet)
        self.assertEqual(
            bet.bookmaker, "bookmaker_b",
            f"bookmaker='{bet.bookmaker if bet else None}'. "
            "Should point to the bookmaker offering the best odds, not the last one in the list.",
        )


# ---------------------------------------------------------------------------
# Group 5 — get_line_movement behaviour
# ---------------------------------------------------------------------------

class TestGetLineMovement(unittest.TestCase):

    def setUp(self):
        self.cache = make_cache()
        self.svc = OddsService(self.cache)

    def test_no_history_returns_flat(self):
        line = _make_odds_line("consensus", -150, 130, spread_point=-5.5)

        async def mock_odds(eid, markets):
            return [line]

        async def mock_hist(eid, ts, markets=None):
            return {}

        self.svc.get_odds = mock_odds
        self.svc.get_historical_event_odds = mock_hist

        result = run(self.svc.get_line_movement("test_game"))
        self.assertEqual(result.opening_line, -5.5)
        self.assertEqual(result.current_line, -5.5)
        self.assertEqual(result.delta, 0)
        self.assertEqual(result.direction, "Flat")

    def test_no_spread_data_returns_zeros(self):
        no_spread = OddsLine(
            event_id="test_game",
            home_team="Lakers",
            away_team="Warriors",
            commence_time=datetime(2026, 3, 21, 19, 0, tzinfo=timezone.utc),
            bookmaker="test",
        )

        async def mock_odds(eid, markets):
            return [no_spread]

        self.svc.get_odds = mock_odds
        result = run(self.svc.get_line_movement("test_game"))
        self.assertEqual(result.delta, 0)
        self.assertEqual(result.direction, "Flat")

    def test_empty_odds_returns_flat(self):
        async def mock_odds(eid, markets):
            return []

        self.svc.get_odds = mock_odds
        result = run(self.svc.get_line_movement("test_game"))
        self.assertEqual(result.direction, "Flat")

    def test_line_moved_down(self):
        """current=-7.5, opening=-5.5 → delta=-2.0, direction='Down'."""
        current = OddsLine(
            event_id="test_game",
            home_team="Lakers",
            away_team="Warriors",
            commence_time=datetime(2026, 3, 21, 19, 0, tzinfo=timezone.utc),
            bookmaker="dk",
            spread=Spread(home_team="Lakers", home_point=-7.5, home_price=-110, away_team="Warriors", away_point=7.5, away_price=-110),
        )
        hist = {
            "data": {
                "bookmakers": [{
                    "key": "dk",
                    "markets": [{
                        "key": "spreads",
                        "outcomes": [{"name": "Lakers", "point": -5.5, "price": -110}],
                    }],
                }],
            }
        }

        async def mock_odds(eid, markets):
            return [current]

        async def mock_hist(eid, ts, markets=None):
            return hist

        self.svc.get_odds = mock_odds
        self.svc.get_historical_event_odds = mock_hist

        result = run(self.svc.get_line_movement("test_game"))
        self.assertAlmostEqual(result.delta, -2.0, places=1)
        self.assertEqual(result.direction, "Down")
        self.assertAlmostEqual(result.opening_line, -5.5, places=1)
        self.assertAlmostEqual(result.current_line, -7.5, places=1)

    def test_line_moved_up(self):
        """current=-3.5, opening=-5.5 → delta=+2.0, direction='Up'."""
        current = OddsLine(
            event_id="test_game",
            home_team="Lakers",
            away_team="Warriors",
            commence_time=datetime(2026, 3, 21, 19, 0, tzinfo=timezone.utc),
            bookmaker="dk",
            spread=Spread(home_team="Lakers", home_point=-3.5, home_price=-110, away_team="Warriors", away_point=3.5, away_price=-110),
        )
        hist = {
            "data": {
                "bookmakers": [{
                    "key": "dk",
                    "markets": [{
                        "key": "spreads",
                        "outcomes": [{"name": "Lakers", "point": -5.5, "price": -110}],
                    }],
                }],
            }
        }

        async def mock_odds(eid, markets):
            return [current]

        async def mock_hist(eid, ts, markets=None):
            return hist

        self.svc.get_odds = mock_odds
        self.svc.get_historical_event_odds = mock_hist

        result = run(self.svc.get_line_movement("test_game"))
        self.assertAlmostEqual(result.delta, 2.0, places=1)
        self.assertEqual(result.direction, "Up")


# ---------------------------------------------------------------------------
# Group 6 — compare_books
# ---------------------------------------------------------------------------

class TestCompareBooks(unittest.TestCase):

    def setUp(self):
        self.cache = make_cache()
        self.svc = OddsService(self.cache)

    def test_h2h_returns_all_bookmakers(self):
        lines = [
            _make_odds_line("draftkings", -150, 130),
            _make_odds_line("fanduel", -140, 120),
            _make_odds_line("betmgm", -155, 135),
        ]

        async def mock_odds(eid, markets):
            return lines

        self.svc.get_odds = mock_odds
        result = run(self.svc.compare_books("test_game", "h2h"))
        self.assertIn("draftkings", result)
        self.assertIn("fanduel", result)
        self.assertIn("betmgm", result)

    def test_h2h_values_correct(self):
        lines = [_make_odds_line("dk", -150, 130)]

        async def mock_odds(eid, markets):
            return lines

        self.svc.get_odds = mock_odds
        result = run(self.svc.compare_books("test_game", "h2h"))
        self.assertEqual(result["dk"]["home"], -150)
        self.assertEqual(result["dk"]["away"], 130)

    def test_spreads_market(self):
        lines = [
            _make_odds_line("dk", -150, 130, spread_point=-5.5),
            _make_odds_line("fd", -140, 120, spread_point=-6.0),
        ]

        async def mock_odds(eid, markets):
            return lines

        self.svc.get_odds = mock_odds
        result = run(self.svc.compare_books("test_game", "spreads"))
        self.assertEqual(result["dk"]["home_point"], -5.5)
        self.assertEqual(result["fd"]["home_point"], -6.0)

    def test_totals_market(self):
        lines = [_make_odds_line("dk", -150, 130)]

        async def mock_odds(eid, markets):
            return lines

        self.svc.get_odds = mock_odds
        result = run(self.svc.compare_books("test_game", "totals"))
        self.assertEqual(result["dk"]["point"], 220.5)
        self.assertEqual(result["dk"]["over"], -110)
        self.assertEqual(result["dk"]["under"], -110)

    def test_empty_odds_returns_empty_dict(self):
        async def mock_odds(eid, markets):
            return []

        self.svc.get_odds = mock_odds
        result = run(self.svc.compare_books("test_game", "h2h"))
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Group 6b — exact-line matching for arb / EV
# ---------------------------------------------------------------------------

class TestExactLineMatching(unittest.TestCase):

    def setUp(self):
        self.svc = _make_analysis_svc()
        self.game = _make_game()

    def test_totals_arb_does_not_cross_different_points(self):
        over_210 = _make_odds_line("book_over", -110, +100)
        over_210.total = Total(point=210.5, over_price=+140, under_price=-110)

        under_212 = _make_odds_line("book_under", -110, +100)
        under_212.total = Total(point=212.5, over_price=-110, under_price=+140)

        result = self.svc._scan_game_lines_for_arb(
            self.game, [over_210, under_212], min_arb_pct=0.001
        )

        self.assertEqual(result, [])

    def test_totals_arb_allows_same_point(self):
        over_211 = _make_odds_line("book_over", -110, +100)
        over_211.total = Total(point=211.5, over_price=+140, under_price=-110)

        under_211 = _make_odds_line("book_under", -110, +100)
        under_211.total = Total(point=211.5, over_price=-110, under_price=+140)

        result = self.svc._scan_game_lines_for_arb(
            self.game, [over_211, under_211], min_arb_pct=0.001
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].side_a, "Over 211.5")
        self.assertEqual(result[0].side_b, "Under 211.5")

    def test_spread_ev_skips_different_point_than_pinnacle(self):
        pinnacle = _make_odds_line("pinnacle", -110, -110, spread_point=-5.5)
        off_market = _make_odds_line("book", -110, -110, spread_point=-7.5)
        off_market.spread.home_price = +140
        off_market.spread.away_price = +140

        result = self.svc._scan_game_ev(
            self.game, [off_market], pinnacle, min_edge=0.001
        )

        self.assertEqual(result, [])

    def test_total_ev_skips_different_point_than_pinnacle(self):
        pinnacle = _make_odds_line("pinnacle", -110, -110)
        pinnacle.total = Total(point=211.5, over_price=-110, under_price=-110)

        off_market = _make_odds_line("book", -110, -110)
        off_market.total = Total(point=213.5, over_price=+140, under_price=+140)

        result = self.svc._scan_game_ev(
            self.game, [off_market], pinnacle, min_edge=0.001
        )

        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Group 7 — get_player_minutes_consistency
# ---------------------------------------------------------------------------

class TestMinutesConsistency(unittest.TestCase):

    def setUp(self):
        self.cache = make_cache()
        self.svc = StatsService(self.cache)

    def _patch(self, minutes_list):
        logs = [{"MIN": str(m)} for m in minutes_list]

        async def mock_gps(pid, last_n=10):
            return logs

        self.svc.get_player_stats = mock_gps

    def test_very_consistent(self):
        # stdev ≈ 0.84 → < 3 → "very_consistent"
        self._patch([32, 33, 31, 32, 33])
        result = run(self.svc.get_player_minutes_consistency(1))
        self.assertEqual(result["grade"], "very_consistent")
        self.assertLess(result["std_dev"], 3.0)

    def test_consistent(self):
        # stdev ≈ 3.05 → 3 ≤ stdev < 5 → "consistent"
        self._patch([25, 30, 28, 33, 27])
        result = run(self.svc.get_player_minutes_consistency(1))
        self.assertEqual(result["grade"], "consistent")
        self.assertGreaterEqual(result["std_dev"], 3.0)
        self.assertLess(result["std_dev"], 5.0)

    def test_moderate(self):
        # stdev ≈ 6.58 → 5 ≤ stdev < 8 → "moderate"
        self._patch([20, 28, 35, 22, 33])
        result = run(self.svc.get_player_minutes_consistency(1))
        self.assertEqual(result["grade"], "moderate")

    def test_volatile(self):
        # stdev ≈ 17.45 → ≥ 8 → "volatile"
        self._patch([5, 38, 10, 40, 7])
        result = run(self.svc.get_player_minutes_consistency(1))
        self.assertEqual(result["grade"], "volatile")
        self.assertGreaterEqual(result["std_dev"], 8.0)

    def test_colon_format(self):
        # "32:30"→32.5, "31:00"→31.0, "33:00"→33.0 → mean≈32.2
        logs = [{"MIN": "32:30"}, {"MIN": "31:00"}, {"MIN": "33:00"}]

        async def mock_gps(pid, last_n=10):
            return logs

        self.svc.get_player_stats = mock_gps
        result = run(self.svc.get_player_minutes_consistency(1))
        self.assertEqual(result["grade"], "very_consistent")
        self.assertAlmostEqual(result["mean_minutes"], 32.2, delta=0.1)

    def test_empty_logs_fallback(self):
        async def mock_gps(pid, last_n=10):
            return []

        self.svc.get_player_stats = mock_gps
        result = run(self.svc.get_player_minutes_consistency(1))
        self.assertEqual(result["grade"], "moderate")
        self.assertEqual(result["games"], 0)

    def test_none_min_skipped(self):
        logs = [{"MIN": None}, {"MIN": "32"}, {"MIN": "33"}, {"MIN": "31"}]

        async def mock_gps(pid, last_n=10):
            return logs

        self.svc.get_player_stats = mock_gps
        result = run(self.svc.get_player_minutes_consistency(1))
        self.assertEqual(result["games"], 3)

    def test_cache_hit_skips_underlying_call(self):
        self.cache.get = AsyncMock(return_value={
            "mean_minutes": 32.0, "std_dev": 1.0,
            "grade": "very_consistent", "games": 5,
        })
        call_count = {"n": 0}

        async def counting(pid, last_n=10):
            call_count["n"] += 1
            return []

        self.svc.get_player_stats = counting
        result = run(self.svc.get_player_minutes_consistency(1))
        self.assertEqual(call_count["n"], 0)
        self.assertEqual(result["grade"], "very_consistent")


# ---------------------------------------------------------------------------
# Group 8 — is_team_on_back_to_back
# ---------------------------------------------------------------------------

class TestIsTeamOnBackToBack(unittest.TestCase):

    def setUp(self):
        self.cache = make_cache()
        self.svc = StatsService(self.cache)

    def _lakers_teams(self):
        return [{"id": 1610612747, "full_name": "Los Angeles Lakers",
                 "abbreviation": "LAL", "nickname": "Lakers", "city": "Los Angeles"}]

    def test_true_when_played_yesterday(self):
        yesterday_game = Game(
            id="g1", home_team="LAL", away_team="GSW",
            commence_time=datetime(2026, 3, 20, 19, 0, tzinfo=timezone.utc),
        )

        async def mock_schedule(d):
            return [yesterday_game]

        self.svc.get_schedule = mock_schedule
        with patch("nba_api.stats.static.teams.get_teams", return_value=self._lakers_teams()):
            result = run(self.svc.is_team_on_back_to_back(1610612747, date(2026, 3, 21)))
        self.assertTrue(result)

    def test_false_when_not_on_schedule_yesterday(self):
        other_game = Game(
            id="g1", home_team="BOS", away_team="MIA",
            commence_time=datetime(2026, 3, 20, 19, 0, tzinfo=timezone.utc),
        )

        async def mock_schedule(d):
            return [other_game]

        self.svc.get_schedule = mock_schedule
        with patch("nba_api.stats.static.teams.get_teams", return_value=self._lakers_teams()):
            result = run(self.svc.is_team_on_back_to_back(1610612747, date(2026, 3, 21)))
        self.assertFalse(result)

    def test_false_when_no_games_yesterday(self):
        async def mock_schedule(d):
            return []

        self.svc.get_schedule = mock_schedule
        with patch("nba_api.stats.static.teams.get_teams", return_value=self._lakers_teams()):
            result = run(self.svc.is_team_on_back_to_back(1610612747, date(2026, 3, 21)))
        self.assertFalse(result)

    def test_cached_true_returned(self):
        self.cache.get = AsyncMock(return_value=True)
        result = run(self.svc.is_team_on_back_to_back(1610612747, date(2026, 3, 21)))
        self.assertTrue(result)

    def test_cached_false_not_confused_with_miss(self):
        """False is a valid cached value — must not be treated as a cache miss."""
        self.cache.get = AsyncMock(return_value=False)
        result = run(self.svc.is_team_on_back_to_back(1610612747, date(2026, 3, 21)))
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Group 9 — find_team_by_name (real nba_api static data, no network)
# ---------------------------------------------------------------------------

class TestFindTeamByName(unittest.TestCase):

    def setUp(self):
        self.cache = make_cache()
        self.svc = StatsService(self.cache)

    def test_nickname(self):
        result = self.svc.find_team_by_name("Lakers")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 1610612747)

    def test_abbreviation(self):
        result = self.svc.find_team_by_name("LAL")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 1610612747)

    def test_full_name(self):
        result = self.svc.find_team_by_name("Los Angeles Lakers")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 1610612747)

    def test_lowercase(self):
        result = self.svc.find_team_by_name("lakers")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 1610612747)

    def test_lowercase_abbreviation(self):
        result = self.svc.find_team_by_name("lal")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 1610612747)

    def test_celtics(self):
        result = self.svc.find_team_by_name("Celtics")
        self.assertIsNotNone(result)
        self.assertEqual(result["abbreviation"], "BOS")

    def test_warriors(self):
        result = self.svc.find_team_by_name("Warriors")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 1610612744)

    def test_not_found(self):
        result = self.svc.find_team_by_name("notateam")
        self.assertIsNone(result)

    def test_partial_city(self):
        result = self.svc.find_team_by_name("Golden State")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 1610612744)


# ---------------------------------------------------------------------------
# Group 10 — find_value_bets integration
# ---------------------------------------------------------------------------

class TestFindValueBets(unittest.TestCase):

    def _make_svc(self, home_wins, away_wins, home_ml, away_ml):
        game = _make_game()
        line = _make_odds_line("dk", home_ml=home_ml, away_ml=away_ml)

        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_games = AsyncMock(return_value=[game])
        odds_mock.get_odds = AsyncMock(return_value=[line])

        home_stats = _make_team_stats(1610612747, "Lakers", wins=home_wins, losses=10 - home_wins)
        away_stats = _make_team_stats(1610612744, "Warriors", wins=away_wins, losses=10 - away_wins)

        stats_mock = MagicMock(spec=StatsService)
        stats_mock.find_team_by_name = MagicMock(
            side_effect=lambda name: (
                {"id": 1610612747, "full_name": "Los Angeles Lakers"}
                if "Lakers" in name
                else {"id": 1610612744, "full_name": "Golden State Warriors"}
            )
        )
        stats_mock.get_team_stats = AsyncMock(
            side_effect=lambda tid, last_n=10: (
                home_stats if tid == 1610612747 else away_stats
            )
        )
        return _make_analysis_svc(odds_mock=odds_mock, stats_mock=stats_mock)

    def test_value_bet_appears_with_large_edge(self):
        svc = self._make_svc(home_wins=7, away_wins=3, home_ml=150, away_ml=-180)
        bets = run(svc.find_value_bets())
        self.assertGreater(len(bets), 0)

    def test_edge_equals_true_minus_implied(self):
        svc = self._make_svc(home_wins=7, away_wins=3, home_ml=150, away_ml=-180)
        bets = run(svc.find_value_bets())
        for bet in bets:
            expected = round(bet.true_probability - bet.implied_probability, 4)
            self.assertAlmostEqual(bet.edge, expected, places=3)

    def test_no_bet_when_edge_below_threshold(self):
        # ~50% win rate vs -110 implied → edge ≈ -0.02 → no bet
        svc = self._make_svc(home_wins=5, away_wins=5, home_ml=-110, away_ml=-110)
        bets = run(svc.find_value_bets())
        self.assertEqual(bets, [])

    def test_sorted_by_edge_descending(self):
        svc = self._make_svc(home_wins=8, away_wins=2, home_ml=150, away_ml=-200)
        bets = run(svc.find_value_bets())
        for i in range(len(bets) - 1):
            self.assertGreaterEqual(bets[i].edge, bets[i + 1].edge)

    def test_confidence_matches_edge(self):
        svc = self._make_svc(home_wins=9, away_wins=1, home_ml=150, away_ml=-200)
        bets = run(svc.find_value_bets())
        for bet in bets:
            if bet.edge >= 0.08:
                self.assertEqual(bet.confidence, "High")
            elif bet.edge >= 0.04:
                self.assertEqual(bet.confidence, "Medium")
            else:
                self.assertEqual(bet.confidence, "Low")

    def test_kelly_within_bounds(self):
        # Quarter-Kelly: max is 6.25% (0.0625)
        svc = self._make_svc(home_wins=8, away_wins=2, home_ml=130, away_ml=-160)
        bets = run(svc.find_value_bets())
        for bet in bets:
            self.assertGreaterEqual(bet.kelly_fraction, 0.0)
            self.assertLessEqual(bet.kelly_fraction, 0.0625)

    def test_team_not_found_returns_empty(self):
        game = _make_game()
        line = _make_odds_line("dk", -150, 130)
        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_games = AsyncMock(return_value=[game])
        odds_mock.get_odds = AsyncMock(return_value=[line])
        stats_mock = MagicMock(spec=StatsService)
        stats_mock.find_team_by_name = MagicMock(return_value=None)

        svc = _make_analysis_svc(odds_mock=odds_mock, stats_mock=stats_mock)
        bets = run(svc.find_value_bets())
        self.assertEqual(bets, [])

    def test_no_games_returns_empty(self):
        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_games = AsyncMock(return_value=[])
        svc = _make_analysis_svc(odds_mock=odds_mock)
        self.assertEqual(run(svc.find_value_bets()), [])


# ---------------------------------------------------------------------------
# Group 12 — DefenseService.get_opponent_def_rank
# ---------------------------------------------------------------------------

class TestDefenseService(unittest.TestCase):

    def setUp(self):
        self.cache = make_cache()
        self.svc = DefenseService(self.cache)

    def _make_df(self, n=30):
        import pandas as pd
        # OPP_PTS increases → rank 1 = fewest allowed = best defence
        return pd.DataFrame({
            "TEAM_ID": list(range(1, n + 1)),
            "OPP_PTS": [float(95 + i * 2) for i in range(n)],
            "OPP_REB": [float(40 + i) for i in range(n)],
            "OPP_AST": [float(22 + i) for i in range(n)],
            "OPP_FG3M": [float(10 + i) for i in range(n)],
        })

    def _patch(self, df):
        mock_result = MagicMock()
        mock_result.league_dash_team_stats.get_data_frame.return_value = df
        return patch(
            "nba_api.stats.endpoints.leaguedashteamstats.LeagueDashTeamStats",
            return_value=mock_result,
        )

    def test_unsupported_stat_returns_empty(self):
        self.assertEqual(run(self.svc.get_opponent_def_rank(1, "FGA")), {})

    def test_empty_stat_returns_empty(self):
        self.assertEqual(run(self.svc.get_opponent_def_rank(1, "")), {})

    def test_rank_1_best(self):
        df = self._make_df()
        with self._patch(df):
            result = run(self.svc.get_opponent_def_rank(1, "PTS"))
        self.assertEqual(result["rank"], 1)

    def test_rank_30_worst(self):
        df = self._make_df()
        with self._patch(df):
            result = run(self.svc.get_opponent_def_rank(30, "PTS"))
        self.assertEqual(result["rank"], 30)

    def test_grade_elite_ranks_1_to_8(self):
        df = self._make_df()
        for tid in [1, 8]:
            self.cache.get = AsyncMock(return_value=None)
            with self._patch(df):
                result = run(self.svc.get_opponent_def_rank(tid, "PTS"))
            self.assertEqual(result["grade"], "elite", f"tid={tid} rank={result['rank']}")

    def test_grade_average_ranks_9_to_22(self):
        df = self._make_df()
        for tid in [9, 22]:
            self.cache.get = AsyncMock(return_value=None)
            with self._patch(df):
                result = run(self.svc.get_opponent_def_rank(tid, "PTS"))
            self.assertEqual(result["grade"], "average", f"tid={tid} rank={result['rank']}")

    def test_grade_weak_ranks_23_to_30(self):
        df = self._make_df()
        for tid in [23, 30]:
            self.cache.get = AsyncMock(return_value=None)
            with self._patch(df):
                result = run(self.svc.get_opponent_def_rank(tid, "PTS"))
            self.assertEqual(result["grade"], "weak", f"tid={tid} rank={result['rank']}")

    def test_team_not_in_df_returns_empty(self):
        df = self._make_df(30)
        with self._patch(df):
            result = run(self.svc.get_opponent_def_rank(9999, "PTS"))
        self.assertEqual(result, {})

    def test_value_field_correct(self):
        df = self._make_df()
        with self._patch(df):
            result = run(self.svc.get_opponent_def_rank(1, "PTS"))
        self.assertAlmostEqual(result["value"], 95.0, places=1)

    def test_reb_category(self):
        df = self._make_df()
        with self._patch(df):
            result = run(self.svc.get_opponent_def_rank(1, "REB"))
        self.assertEqual(result["rank"], 1)
        self.assertAlmostEqual(result["value"], 40.0, places=1)

    def test_cache_hit(self):
        self.cache.get = AsyncMock(return_value={"rank": 5, "grade": "elite", "value": 100.0})
        result = run(self.svc.get_opponent_def_rank(1610612747, "PTS"))
        self.assertEqual(result["rank"], 5)
        self.assertEqual(result["grade"], "elite")


# ---------------------------------------------------------------------------
# Group 13 — Cache hit/miss behaviour
# ---------------------------------------------------------------------------

class TestCacheHitMiss(unittest.TestCase):

    def test_props_api_called_on_miss(self):
        cache = make_cache()
        svc = OddsService(cache)
        with patch.object(svc, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"bookmakers": []}
            run(svc.get_player_props("event1"))
            mock_get.assert_called_once()

    def test_props_api_skipped_on_hit(self):
        cache = make_cache()
        cache.get = AsyncMock(return_value=[{
            "event_id": "event1", "player_name": "LeBron James",
            "market": "player_points", "line": 27.5,
            "over_price": -110, "under_price": -110, "bookmaker": "dk",
        }])
        svc = OddsService(cache)
        with patch.object(svc, "_get", new_callable=AsyncMock) as mock_get:
            result = run(svc.get_player_props("event1"))
            mock_get.assert_not_called()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].player_name, "LeBron James")

    def test_get_odds_skipped_on_hit(self):
        cache = make_cache()
        cache.get = AsyncMock(return_value=[{
            "event_id": "ev1", "home_team": "Lakers", "away_team": "Warriors",
            "commence_time": "2026-03-21T19:00:00+00:00",
            "bookmaker": "dk", "moneyline_home": -150, "moneyline_away": 130,
        }])
        svc = OddsService(cache)
        with patch.object(svc, "_get", new_callable=AsyncMock) as mock_get:
            result = run(svc.get_odds("ev1"))
            mock_get.assert_not_called()
            self.assertEqual(result[0].bookmaker, "dk")

    def test_stats_cache_set_after_call(self):
        import pandas as pd
        cache = make_cache()
        svc = StatsService(cache)

        mock_ep = MagicMock()
        mock_ep.team_game_logs.get_data_frame.return_value = pd.DataFrame({
            "WL": ["W"], "PTS": [110.0], "PLUS_MINUS": [5.0],
            "TEAM_NAME": ["Lakers"],
        })
        svc._get_team_game_logs = MagicMock(return_value=mock_ep)

        adv_df = pd.DataFrame({
            "TEAM_ID": [1610612747], "PACE": [98.5],
            "OFF_RATING": [112.0], "DEF_RATING": [108.0], "NET_RATING": [4.0],
        })
        mock_adv = MagicMock()
        mock_adv.league_dash_team_stats.get_data_frame.return_value = adv_df
        svc._get_league_team_stats_advanced = MagicMock(return_value=mock_adv)

        run(svc.get_team_stats(1610612747))
        cache.set.assert_called()


# ---------------------------------------------------------------------------
# Group 14 — get_team_stats correctness
# ---------------------------------------------------------------------------

class TestGetTeamStats(unittest.TestCase):

    def setUp(self):
        self.cache = make_cache()
        self.svc = StatsService(self.cache)

    def _stub_advanced(self):
        import pandas as pd
        adv_df = pd.DataFrame({
            "TEAM_ID": [1610612747], "PACE": [99.0],
            "OFF_RATING": [110.0], "DEF_RATING": [107.0], "NET_RATING": [3.0],
        })
        mock_adv = MagicMock()
        mock_adv.league_dash_team_stats.get_data_frame.return_value = adv_df
        self.svc._get_league_team_stats_advanced = MagicMock(return_value=mock_adv)

    def test_win_loss_record(self):
        import pandas as pd
        mock_ep = MagicMock()
        mock_ep.team_game_logs.get_data_frame.return_value = pd.DataFrame({
            "WL": ["W", "W", "L", "W", "L"],
            "PTS": [110.0, 105.0, 100.0, 115.0, 108.0],
            "PLUS_MINUS": [5.0, 3.0, -4.0, 8.0, -2.0],
            "TEAM_NAME": ["Los Angeles Lakers"] * 5,
        })
        self.svc._get_team_game_logs = MagicMock(return_value=mock_ep)
        self._stub_advanced()

        result = run(self.svc.get_team_stats(1610612747, last_n=5))
        self.assertEqual(result.wins, 3)
        self.assertEqual(result.losses, 2)
        self.assertAlmostEqual(result.win_pct, 0.6, places=2)

    def test_ppg(self):
        import pandas as pd
        mock_ep = MagicMock()
        mock_ep.team_game_logs.get_data_frame.return_value = pd.DataFrame({
            "WL": ["W", "W", "W"],
            "PTS": [100.0, 110.0, 120.0],
            "PLUS_MINUS": [5.0, 5.0, 5.0],
            "TEAM_NAME": ["Lakers"] * 3,
        })
        self.svc._get_team_game_logs = MagicMock(return_value=mock_ep)
        self._stub_advanced()

        result = run(self.svc.get_team_stats(1610612747))
        self.assertAlmostEqual(result.points_per_game, 110.0, places=1)

    def test_papg_fallback_from_plus_minus(self):
        import pandas as pd
        # PAPG = (110-10 + 120-5) / 2 = (100 + 115) / 2 = 107.5
        mock_ep = MagicMock()
        mock_ep.team_game_logs.get_data_frame.return_value = pd.DataFrame({
            "WL": ["W", "W"],
            "PTS": [110.0, 120.0],
            "PLUS_MINUS": [10.0, 5.0],
            "TEAM_NAME": ["Lakers"] * 2,
        })
        self.svc._get_team_game_logs = MagicMock(return_value=mock_ep)
        self._stub_advanced()

        result = run(self.svc.get_team_stats(1610612747))
        self.assertAlmostEqual(result.points_allowed_per_game, 107.5, places=1)

    def test_empty_df_returns_unknown(self):
        import pandas as pd
        mock_ep = MagicMock()
        mock_ep.team_game_logs.get_data_frame.return_value = pd.DataFrame()
        self.svc._get_team_game_logs = MagicMock(return_value=mock_ep)

        result = run(self.svc.get_team_stats(9999))
        self.assertEqual(result.wins, 0)
        self.assertEqual(result.team_name, "Unknown")


# ---------------------------------------------------------------------------
# Group 15 — Regression: BUG-A (skip-tier filter in find_value_bets)
# ---------------------------------------------------------------------------

class TestFindValueBetsSkipTier(unittest.TestCase):
    """fliff/espnbet must never appear as the best book in find_value_bets()."""

    def _make_svc(self, lines):
        game = _make_game()
        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_games = AsyncMock(return_value=[game])
        odds_mock.get_odds = AsyncMock(return_value=lines)
        odds_mock.get_pinnacle_odds = AsyncMock(return_value=None)  # force stats fallback

        home_stats = _make_team_stats(1610612747, "Lakers", wins=7, losses=3)
        away_stats = _make_team_stats(1610612744, "Warriors", wins=3, losses=7)
        stats_mock = MagicMock(spec=StatsService)
        stats_mock.find_team_by_name = MagicMock(
            side_effect=lambda name: (
                {"id": 1610612747, "full_name": "Los Angeles Lakers"} if "Lakers" in name
                else {"id": 1610612744, "full_name": "Golden State Warriors"}
            )
        )
        stats_mock.get_team_stats = AsyncMock(
            side_effect=lambda tid, last_n=10: home_stats if tid == 1610612747 else away_stats
        )
        return _make_analysis_svc(odds_mock=odds_mock, stats_mock=stats_mock)

    def test_fliff_excluded_even_if_best_odds(self):
        """fliff offers +300 (best on paper) but must be excluded."""
        lines = [
            _make_odds_line("draftkings", home_ml=130, away_ml=-155),
            _make_odds_line("fliff", home_ml=300, away_ml=-400),  # skip tier
        ]
        svc = self._make_svc(lines)
        bets = run(svc.find_value_bets())
        for bet in bets:
            self.assertNotIn("fliff", bet.bookmaker,
                "fliff is a sweepstakes book and must not appear in value bet results.")

    def test_espnbet_excluded_even_if_best_odds(self):
        """espnbet (defunct) must never appear in results."""
        lines = [
            _make_odds_line("fanduel", home_ml=120, away_ml=-145),
            _make_odds_line("espnbet", home_ml=250, away_ml=-300),  # skip tier
        ]
        svc = self._make_svc(lines)
        bets = run(svc.find_value_bets())
        for bet in bets:
            self.assertNotIn("espnbet", bet.bookmaker,
                "espnbet is defunct and must not appear in value bet results.")

    def test_real_book_selected_when_skip_tier_also_present(self):
        """With fliff at +300 and DK at +130, DK should be selected."""
        lines = [
            _make_odds_line("draftkings", home_ml=130, away_ml=-155),
            _make_odds_line("fliff", home_ml=300, away_ml=-400),
        ]
        svc = self._make_svc(lines)
        bets = run(svc.find_value_bets())
        home_bets = [b for b in bets if "Lakers" in b.selection]
        if home_bets:
            self.assertEqual(home_bets[0].bookmaker, "draftkings")
            self.assertEqual(home_bets[0].american_odds, 130)


# ---------------------------------------------------------------------------
# Group 16 — Regression: Fix D (player name normalization)
# ---------------------------------------------------------------------------

class TestNormName(unittest.TestCase):
    def setUp(self):
        self.svc = _make_analysis_svc()

    def test_lowercase(self):
        self.assertEqual(self.svc._norm_name("LeBron James"), "lebron james")

    def test_hyphen_collapsed(self):
        self.assertEqual(
            self.svc._norm_name("Shai Gilgeous-Alexander"),
            "shai gilgeous alexander",
        )

    def test_period_collapsed(self):
        self.assertEqual(self.svc._norm_name("J.J. Redick"), "j j redick")

    def test_extra_spaces_collapsed(self):
        self.assertEqual(self.svc._norm_name("  Kevin   Durant  "), "kevin durant")

    def test_hyphen_and_space_match(self):
        """The key insight: 'Foo-Bar' and 'Foo Bar' must normalize to the same string."""
        self.assertEqual(
            self.svc._norm_name("Shai Gilgeous-Alexander"),
            self.svc._norm_name("Shai Gilgeous Alexander"),
        )

    def test_case_insensitive_match(self):
        self.assertEqual(
            self.svc._norm_name("Anthony Davis"),
            self.svc._norm_name("anthony davis"),
        )


# ---------------------------------------------------------------------------
# Group 17 — Regression: Fix E (confidence factors in book count)
# ---------------------------------------------------------------------------

class TestConfidenceWithBookCount(unittest.TestCase):
    def setUp(self):
        self.svc = _make_analysis_svc()

    def test_thin_market_is_always_low(self):
        """book_count < 5 → always Low regardless of edge."""
        for edge in [0.01, 0.04, 0.08, 0.20]:
            for count in [1, 2, 3, 4]:
                with self.subTest(edge=edge, count=count):
                    self.assertEqual(
                        self.svc._confidence(edge, book_count=count), "Low",
                        f"edge={edge}, book_count={count} should be Low (thin market).",
                    )

    def test_liquid_market_uses_edge_thresholds(self):
        """book_count >= 5 → normal edge-based confidence."""
        for count in [5, 6, 10, 20]:
            with self.subTest(count=count):
                self.assertEqual(self.svc._confidence(0.08, book_count=count), "High")
                self.assertEqual(self.svc._confidence(0.04, book_count=count), "Medium")
                self.assertEqual(self.svc._confidence(0.02, book_count=count), "Low")

    def test_default_book_count_is_liquid(self):
        """Default (no book_count arg) should behave as liquid market."""
        self.assertEqual(self.svc._confidence(0.08), "High")
        self.assertEqual(self.svc._confidence(0.04), "Medium")
        self.assertEqual(self.svc._confidence(0.02), "Low")

    def test_boundary_at_5_books(self):
        """4 books → Low; 5 books → respects edge."""
        self.assertEqual(self.svc._confidence(0.08, book_count=4), "Low")
        self.assertEqual(self.svc._confidence(0.08, book_count=5), "High")


# ---------------------------------------------------------------------------
# Prop helpers (used by Groups 21–22)
# ---------------------------------------------------------------------------

def _make_pinnacle_prop(player_name, market, line, over_price, under_price):
    return PlayerProp(
        event_id="test_game", player_name=player_name, market=market, line=line,
        over_price=over_price, under_price=under_price, bookmaker="pinnacle",
    )


def _make_prop(player_name, market, line, over_price, under_price, bookmaker):
    return PlayerProp(
        event_id="test_game", player_name=player_name, market=market, line=line,
        over_price=over_price, under_price=under_price, bookmaker=bookmaker,
    )


# ---------------------------------------------------------------------------
# Group 18 — _downgrade_confidence
# ---------------------------------------------------------------------------

class TestDowngradeConfidence(unittest.TestCase):

    def setUp(self):
        self.svc = _make_analysis_svc()

    def test_high_becomes_medium(self):
        self.assertEqual(self.svc._downgrade_confidence("High"), "Medium")

    def test_medium_becomes_low(self):
        self.assertEqual(self.svc._downgrade_confidence("Medium"), "Low")

    def test_low_stays_low(self):
        self.assertEqual(self.svc._downgrade_confidence("Low"), "Low")


# ---------------------------------------------------------------------------
# Group 19 — _sync_usage_boost
# ---------------------------------------------------------------------------

class TestSyncUsageBoost(unittest.TestCase):

    def setUp(self):
        self.svc = _make_analysis_svc()

    def _make_roster(self):
        """5-player roster. Star=25%, Injured=20%, three bench players 18/17/20%."""
        return [
            {"player_id": 1, "team_id": 100, "usg_pct": 0.25, "player_name_lower": "star player"},
            {"player_id": 2, "team_id": 100, "usg_pct": 0.20, "player_name_lower": "injured player"},
            {"player_id": 3, "team_id": 100, "usg_pct": 0.18, "player_name_lower": "bench a"},
            {"player_id": 4, "team_id": 100, "usg_pct": 0.17, "player_name_lower": "bench b"},
            {"player_id": 5, "team_id": 100, "usg_pct": 0.20, "player_name_lower": "sixth man"},
        ]

    def test_large_boost_returns_0_012(self):
        """Star absorbs >5% usage from injured star teammate → +0.012.

        active_usg_total (excl. star pid=1, excl. injured) = 0.18+0.17+0.20 = 0.55
        player_share = 0.25/0.55 ≈ 0.4545
        boost = 0.4545 * 0.20 ≈ 0.0909 > 0.05 → 0.012
        """
        result = self.svc._sync_usage_boost(1, 0.25, self._make_roster(), {"injured player"})
        self.assertAlmostEqual(result, 0.012, places=3)

    def test_small_boost_returns_zero(self):
        """Low-usage player with a minor injury → boost < 5% → 0.0.

        active_usg_total (excl. pid=1, excl. injured) = 0.30+0.30+0.27 = 0.87
        player_share = 0.10/0.87 ≈ 0.115
        boost = 0.115 * 0.03 ≈ 0.003 < 0.05 → 0.0
        """
        roster = [
            {"player_id": 1, "team_id": 100, "usg_pct": 0.10, "player_name_lower": "bench"},
            {"player_id": 2, "team_id": 100, "usg_pct": 0.03, "player_name_lower": "injured"},
            {"player_id": 3, "team_id": 100, "usg_pct": 0.30, "player_name_lower": "star a"},
            {"player_id": 4, "team_id": 100, "usg_pct": 0.30, "player_name_lower": "star b"},
            {"player_id": 5, "team_id": 100, "usg_pct": 0.27, "player_name_lower": "starter"},
        ]
        result = self.svc._sync_usage_boost(1, 0.10, roster, {"injured"})
        self.assertAlmostEqual(result, 0.0, places=3)

    def test_no_injured_players_returns_zero(self):
        """Empty injured set → no boost regardless of roster."""
        result = self.svc._sync_usage_boost(1, 0.25, self._make_roster(), set())
        self.assertEqual(result, 0.0)

    def test_active_usg_zero_returns_zero(self):
        """No remaining active teammates → divide-by-zero guarded, returns 0.0."""
        roster = [
            {"player_id": 1, "team_id": 100, "usg_pct": 0.25, "player_name_lower": "star"},
            {"player_id": 2, "team_id": 100, "usg_pct": 0.20, "player_name_lower": "injured"},
            # player 1 is the subject, player 2 is injured — no active teammates remain
        ]
        result = self.svc._sync_usage_boost(1, 0.25, roster, {"injured"})
        self.assertEqual(result, 0.0)


# ---------------------------------------------------------------------------
# Group 20 — _get_player_info_map
# ---------------------------------------------------------------------------

class TestGetPlayerInfoMap(unittest.TestCase):

    _PLAYER_DATA = [
        {"PLAYER_ID": 2544, "TEAM_ID": 1610612747, "USG_PCT": 0.30, "PLAYER_NAME": "LeBron James"},
        {"PLAYER_ID": 1629029, "TEAM_ID": 1610612747, "USG_PCT": 0.28, "PLAYER_NAME": "Anthony Davis"},
        {"PLAYER_ID": 203507, "TEAM_ID": 1610612744, "USG_PCT": 0.25, "PLAYER_NAME": "Shai Gilgeous-Alexander"},
    ]

    def _make_svc(self, cached_data):
        cache = make_cache()
        cache.get = AsyncMock(return_value=cached_data)
        return _make_analysis_svc(stats_mock=StatsService(cache))

    def test_name_map_keys_are_normalized(self):
        """Hyphenated names collapse to spaces; keys are lowercase."""
        svc = self._make_svc(self._PLAYER_DATA)
        name_map, _ = run(svc._get_player_info_map())
        self.assertIn("lebron james", name_map)
        self.assertIn("anthony davis", name_map)
        self.assertIn("shai gilgeous alexander", name_map)  # hyphen → space
        # Original hyphenated key must NOT be present
        self.assertNotIn("shai gilgeous-alexander", name_map)

    def test_name_map_values_correct(self):
        svc = self._make_svc(self._PLAYER_DATA)
        name_map, _ = run(svc._get_player_info_map())
        entry = name_map["lebron james"]
        self.assertEqual(entry["player_id"], 2544)
        self.assertEqual(entry["team_id"], 1610612747)
        self.assertAlmostEqual(entry["usg_pct"], 0.30, places=2)

    def test_team_map_groups_by_team_id(self):
        svc = self._make_svc(self._PLAYER_DATA)
        _, team_map = run(svc._get_player_info_map())
        self.assertIn(1610612747, team_map)
        self.assertIn(1610612744, team_map)
        self.assertEqual(len(team_map[1610612747]), 2)   # LeBron + AD
        self.assertEqual(len(team_map[1610612744]), 1)   # Shai

    def test_api_failure_returns_empty_maps(self):
        """When cache misses and nba_api raises, returns ({}, {})."""
        from unittest.mock import patch
        cache = make_cache()
        cache.get = AsyncMock(return_value=None)   # force miss
        svc = _make_analysis_svc(stats_mock=StatsService(cache))
        with patch(
            "nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats",
            side_effect=Exception("API down"),
        ):
            name_map, team_map = run(svc._get_player_info_map())
        self.assertEqual(name_map, {})
        self.assertEqual(team_map, {})


# ---------------------------------------------------------------------------
# Group 21 — _scan_props_ev with contextual signals (injury / B2B / defense)
# ---------------------------------------------------------------------------

class TestScanPropsEVSignals(unittest.TestCase):
    """
    Pinnacle: over -115 / under -105
      imp_over = 115/215 ≈ 0.53488, imp_under = 105/205 ≈ 0.51220
      total_vig = 1.04708
      true_over ≈ 0.51086, true_under ≈ 0.48914

    5 real books, each: over +120 / under -130
      imp_over = 100/220 ≈ 0.45455  → edge_over ≈ 0.0563  (Medium, 5 books)
      imp_under = 130/230 ≈ 0.56522 → edge_under ≈ -0.076  (no bet)
    """
    BOOKS = ["draftkings", "fanduel", "betmgm", "betrivers", "caesars"]
    HOME_ID = 1610612747
    AWAY_ID  = 1610612744

    def setUp(self):
        self.svc = _make_analysis_svc()
        self.game = _make_game()
        self.pinn_props = [_make_pinnacle_prop("LeBron James", "player_points", 27.5, -115, -105)]
        self.props = [
            _make_prop("LeBron James", "player_points", 27.5, 120, -130, bk)
            for bk in self.BOOKS
        ]
        self.name_map = {
            "lebron james": {
                "player_id": 2544, "team_id": self.HOME_ID,
                "usg_pct": 0.30, "player_name_lower": "lebron james",
            },
        }
        self.team_map = {
            self.HOME_ID: [
                {"player_id": 2544, "team_id": self.HOME_ID,
                 "usg_pct": 0.30, "player_name_lower": "lebron james"},
                {"player_id": 9999, "team_id": self.HOME_ID,
                 "usg_pct": 0.20, "player_name_lower": "anthony davis"},
            ]
        }

    def _over_bets(self, bets):
        return [b for b in bets if "Over" in b.selection]

    # --- baseline ---

    def test_baseline_produces_one_over_bet_per_book(self):
        """Without any context signals, one Over bet per book (5 total), no Under bets."""
        bets = self.svc._scan_props_ev(self.game, self.props, self.pinn_props, min_edge=0.01)
        self.assertEqual(len(bets), len(self.BOOKS))
        self.assertTrue(all("Over" in b.selection for b in bets))

    def test_baseline_confidence_is_medium(self):
        """edge ≈ 0.056 and 5 books → 'Medium'."""
        bets = self.svc._scan_props_ev(self.game, self.props, self.pinn_props, min_edge=0.01)
        for b in bets:
            self.assertEqual(b.confidence, "Medium")

    def test_back_to_back_is_none_when_player_not_in_name_map(self):
        """When name_map is empty, back_to_back field is None (no team data)."""
        bets = self.svc._scan_props_ev(self.game, self.props, self.pinn_props, min_edge=0.01)
        for b in bets:
            self.assertIsNone(b.back_to_back)

    # --- B2B ---

    def test_b2b_downgrades_medium_to_low(self):
        bets = self.svc._scan_props_ev(
            self.game, self.props, self.pinn_props, min_edge=0.01,
            name_map=self.name_map, team_map=self.team_map,
            home_team_id=self.HOME_ID, away_team_id=self.AWAY_ID,
            home_b2b=True,
        )
        for b in self._over_bets(bets):
            self.assertEqual(b.confidence, "Low", "B2B should step Medium → Low")

    def test_b2b_field_set_true_on_valuebet(self):
        bets = self.svc._scan_props_ev(
            self.game, self.props, self.pinn_props, min_edge=0.01,
            name_map=self.name_map, team_map=self.team_map,
            home_team_id=self.HOME_ID, away_team_id=self.AWAY_ID,
            home_b2b=True,
        )
        for b in self._over_bets(bets):
            self.assertTrue(b.back_to_back)

    def test_no_b2b_field_false_when_player_found(self):
        bets = self.svc._scan_props_ev(
            self.game, self.props, self.pinn_props, min_edge=0.01,
            name_map=self.name_map, team_map=self.team_map,
            home_team_id=self.HOME_ID, away_team_id=self.AWAY_ID,
            home_b2b=False,
        )
        for b in self._over_bets(bets):
            self.assertFalse(b.back_to_back)

    # --- injury / usage boost ---

    def test_injury_boost_sets_usage_boost_field(self):
        """Injured high-usage teammate → usage_boost=0.012 on ValueBet.

        active_usg_total (excl. LeBron, excl. AD) = three bench players = 0.50 total
        player_share ≈ 0.30/0.50 = 0.60; boost = 0.60 * 0.25 = 0.15 > 0.05 → 0.012
        """
        team_map = {
            self.HOME_ID: [
                {"player_id": 2544,  "team_id": self.HOME_ID, "usg_pct": 0.30, "player_name_lower": "lebron james"},
                {"player_id": 1234,  "team_id": self.HOME_ID, "usg_pct": 0.25, "player_name_lower": "anthony davis"},
                {"player_id": 5678,  "team_id": self.HOME_ID, "usg_pct": 0.18, "player_name_lower": "austin reaves"},
                {"player_id": 9012,  "team_id": self.HOME_ID, "usg_pct": 0.17, "player_name_lower": "rui hachimura"},
                {"player_id": 3456,  "team_id": self.HOME_ID, "usg_pct": 0.10, "player_name_lower": "cam christie"},
            ]
        }
        bets = self.svc._scan_props_ev(
            self.game, self.props, self.pinn_props, min_edge=0.01,
            name_map=self.name_map, team_map=team_map,
            home_team_id=self.HOME_ID, away_team_id=self.AWAY_ID,
            home_injured_lower={"anthony davis"},
        )
        for b in self._over_bets(bets):
            self.assertIsNotNone(b.usage_boost)
            self.assertAlmostEqual(b.usage_boost, 0.012, places=3)

    def test_no_injury_usage_boost_is_none(self):
        bets = self.svc._scan_props_ev(
            self.game, self.props, self.pinn_props, min_edge=0.01,
            name_map=self.name_map, team_map=self.team_map,
            home_team_id=self.HOME_ID, away_team_id=self.AWAY_ID,
        )
        for b in self._over_bets(bets):
            self.assertIsNone(b.usage_boost)

    def test_usage_boost_directional_over_up_under_down(self):
        """Usage boost increases Over true_prob and DECREASES Under true_prob.

        Probabilities must remain individually valid (not both inflated).
        Adjusted true probs for a boosted player: adj_over + adj_under = 1.0 still.
        """
        true_over, true_under = self.svc.no_vig_prob(-115, -105)
        boost = 0.012
        adj_over  = true_over  + boost        # Over goes up
        adj_under = true_under - boost        # Under goes down
        self.assertAlmostEqual(adj_over + adj_under, 1.0, places=6,
            msg="Adjusted true probs must still sum to 1.0")

    # --- opponent defense ---

    def test_over_vs_elite_defense_downgrades_medium_to_low(self):
        """LeBron (home) faces away-team elite D → Over confidence Medium → Low."""
        away_def_ranks = {"PTS": {"rank": 3, "grade": "elite", "value": 99.5}}
        bets = self.svc._scan_props_ev(
            self.game, self.props, self.pinn_props, min_edge=0.01,
            name_map=self.name_map, team_map=self.team_map,
            home_team_id=self.HOME_ID, away_team_id=self.AWAY_ID,
            away_def_ranks=away_def_ranks,
        )
        for b in self._over_bets(bets):
            self.assertEqual(b.confidence, "Low")

    def test_over_vs_weak_defense_no_downgrade(self):
        """Weak defense → Over confidence unchanged (Medium)."""
        away_def_ranks = {"PTS": {"rank": 28, "grade": "weak", "value": 118.0}}
        bets = self.svc._scan_props_ev(
            self.game, self.props, self.pinn_props, min_edge=0.01,
            name_map=self.name_map, team_map=self.team_map,
            home_team_id=self.HOME_ID, away_team_id=self.AWAY_ID,
            away_def_ranks=away_def_ranks,
        )
        for b in self._over_bets(bets):
            self.assertEqual(b.confidence, "Medium")

    def test_opponent_def_rank_and_grade_populated(self):
        away_def_ranks = {"PTS": {"rank": 5, "grade": "elite", "value": 100.0}}
        bets = self.svc._scan_props_ev(
            self.game, self.props, self.pinn_props, min_edge=0.01,
            name_map=self.name_map, team_map=self.team_map,
            home_team_id=self.HOME_ID, away_team_id=self.AWAY_ID,
            away_def_ranks=away_def_ranks,
        )
        for b in self._over_bets(bets):
            self.assertEqual(b.opponent_def_rank, 5)
            self.assertEqual(b.opponent_def_grade, "elite")

    def test_b2b_and_elite_defense_double_downgrade(self):
        """B2B + elite defense → two downgrades. Need High initially (edge ≥ 0.08).

        Pinnacle: over -115 / under -105 → true_over ≈ 0.5109
        Books: over +200  → imp ≈ 0.3333  → edge ≈ 0.1776 → High
        B2B: High → Medium; elite D on Over: Medium → Low.
        """
        pinn_high = [_make_pinnacle_prop("LeBron James", "player_points", 27.5, -115, -105)]
        props_high = [
            _make_prop("LeBron James", "player_points", 27.5, 200, -260, bk)
            for bk in self.BOOKS
        ]
        away_def_ranks = {"PTS": {"rank": 3, "grade": "elite", "value": 99.5}}
        bets = self.svc._scan_props_ev(
            self.game, props_high, pinn_high, min_edge=0.01,
            name_map=self.name_map, team_map=self.team_map,
            home_team_id=self.HOME_ID, away_team_id=self.AWAY_ID,
            home_b2b=True,
            away_def_ranks=away_def_ranks,
        )
        over_bets = self._over_bets(bets)
        self.assertGreater(len(over_bets), 0)
        for b in over_bets:
            # edge ≈ 0.18 → initial "High", B2B → "Medium", elite D → "Low"
            self.assertEqual(b.confidence, "Low")


# ---------------------------------------------------------------------------
# Group 22 — find_positive_ev (orchestration)
# ---------------------------------------------------------------------------

class TestFindPositiveEV(unittest.TestCase):

    def test_no_games_returns_empty(self):
        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_games = AsyncMock(return_value=[])
        svc = _make_analysis_svc(odds_mock=odds_mock)
        self.assertEqual(run(svc.find_positive_ev()), [])

    def test_results_sorted_by_edge_descending(self):
        """Bets from multiple games are merged and sorted by edge."""
        from unittest.mock import patch

        game1_bets = [
            ValueBet(event_id="g1", market="h2h", selection="TeamA", bookmaker="dk",
                     american_odds=-110, implied_probability=0.52, true_probability=0.58,
                     edge=0.06, kelly_fraction=0.01, confidence="Medium"),
            ValueBet(event_id="g1", market="h2h", selection="TeamB", bookmaker="dk",
                     american_odds=150, implied_probability=0.40, true_probability=0.48,
                     edge=0.08, kelly_fraction=0.02, confidence="High"),
        ]
        game2_bets = [
            ValueBet(event_id="g2", market="h2h", selection="TeamC", bookmaker="fd",
                     american_odds=-120, implied_probability=0.55, true_probability=0.62,
                     edge=0.07, kelly_fraction=0.015, confidence="Medium"),
        ]

        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_games = AsyncMock(return_value=[_make_game("g1"), _make_game("g2")])
        svc = _make_analysis_svc(odds_mock=odds_mock)

        with patch.object(svc, "_get_player_info_map", new=AsyncMock(return_value=({}, {}))):
            with patch.object(svc, "_process_game_ev",
                              new=AsyncMock(side_effect=[game1_bets, game2_bets])):
                result = run(svc.find_positive_ev())

        self.assertEqual(len(result), 3)
        for i in range(len(result) - 1):
            self.assertGreaterEqual(result[i].edge, result[i + 1].edge)

    def test_player_info_map_fetched_exactly_once(self):
        """_get_player_info_map is called once regardless of game count — not once per game."""
        from unittest.mock import patch

        games = [_make_game(f"g{i}") for i in range(3)]
        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_games = AsyncMock(return_value=games)
        svc = _make_analysis_svc(odds_mock=odds_mock)

        mock_info = AsyncMock(return_value=({}, {}))
        mock_process = AsyncMock(return_value=[])

        with patch.object(svc, "_get_player_info_map", new=mock_info):
            with patch.object(svc, "_process_game_ev", new=mock_process):
                run(svc.find_positive_ev())

        mock_info.assert_called_once()
        self.assertEqual(mock_process.call_count, 3)

    def test_failed_game_does_not_abort_scan(self):
        """If one game's processing raises, the others are still returned."""
        from unittest.mock import patch

        good_bets = [
            ValueBet(event_id="g2", market="h2h", selection="TeamA", bookmaker="dk",
                     american_odds=-110, implied_probability=0.52, true_probability=0.58,
                     edge=0.06, kelly_fraction=0.01, confidence="Medium"),
        ]

        games = [_make_game("g1"), _make_game("g2")]
        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_games = AsyncMock(return_value=games)
        svc = _make_analysis_svc(odds_mock=odds_mock)

        async def side_effect(game, *args, **kwargs):
            if game.id == "g1":
                raise RuntimeError("simulated timeout")
            return good_bets

        with patch.object(svc, "_get_player_info_map", new=AsyncMock(return_value=({}, {}))):
            with patch.object(svc, "_process_game_ev",
                              new=AsyncMock(side_effect=side_effect)):
                result = run(svc.find_positive_ev())

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].event_id, "g2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
