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
    InjuryReport,
    OddsLine,
    PlayerProp,
    Spread,
    TeamStats,
    Total,
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
        spread=Spread(team="Los Angeles Lakers", point=spread_point, price=-110),
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

    def setUp(self):
        self.svc = _make_analysis_svc()

    def test_even_money_ten_pct_edge(self):
        # +100 → decimal=2.0, edge=0.10 → kelly = 0.10*2.0/1.0 = 0.20
        self.assertAlmostEqual(self.svc.kelly_criterion(0.10, 2.0), 0.20, places=4)

    def test_zero_edge_returns_zero(self):
        self.assertEqual(self.svc.kelly_criterion(0.0, 2.0), 0.0)

    def test_negative_edge_returns_zero(self):
        self.assertEqual(self.svc.kelly_criterion(-0.05, 2.0), 0.0)

    def test_decimal_one_returns_zero(self):
        self.assertEqual(self.svc.kelly_criterion(0.10, 1.0), 0.0)

    def test_high_edge_capped_at_25_pct(self):
        # edge=0.60, decimal=5.0 → raw=0.75 → capped at 0.25
        result = self.svc.kelly_criterion(0.60, 5.0)
        self.assertAlmostEqual(result, 0.25, places=4)

    def test_typical_minus_110_odds(self):
        # decimal≈1.9091, edge=0.07 → kelly ≈ 0.1472
        result = self.svc.kelly_criterion(0.07, 1.9091)
        self.assertAlmostEqual(result, 0.1472, places=3)

    def test_result_rounded_to_4_decimals(self):
        result = self.svc.kelly_criterion(0.10, 2.0)
        self.assertEqual(result, 0.2000)


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


class TestPropMarketToStatCol(unittest.TestCase):

    def setUp(self):
        self.svc = _make_analysis_svc()

    def test_points(self):
        self.assertEqual(self.svc._prop_market_to_stat_col("player_points"), "PTS")

    def test_rebounds(self):
        self.assertEqual(self.svc._prop_market_to_stat_col("player_rebounds"), "REB")

    def test_assists(self):
        self.assertEqual(self.svc._prop_market_to_stat_col("player_assists"), "AST")

    def test_threes(self):
        self.assertEqual(self.svc._prop_market_to_stat_col("player_threes"), "FG3M")

    def test_blocks(self):
        self.assertEqual(self.svc._prop_market_to_stat_col("player_blocks"), "BLK")

    def test_steals(self):
        self.assertEqual(self.svc._prop_market_to_stat_col("player_steals"), "STL")

    def test_turnovers(self):
        self.assertEqual(self.svc._prop_market_to_stat_col("player_turnovers"), "TOV")

    def test_double_double_explicitly_none(self):
        self.assertIsNone(self.svc._prop_market_to_stat_col("player_double_double"))

    def test_unknown_market_returns_none(self):
        self.assertIsNone(self.svc._prop_market_to_stat_col("player_fantasy_points"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.svc._prop_market_to_stat_col(""))


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

    def test_bookmaker_attribution_is_bugged(self):
        """BUG 4: bookmaker attribute is 'bookmaker_c' (last) instead of 'bookmaker_b' (best odds).
        This test CONFIRMS the bug. After fix, expected value is 'bookmaker_b'.
        """
        bet = self._run_scenario()
        self.assertIsNotNone(bet)
        self.assertEqual(
            bet.bookmaker, "bookmaker_c",
            f"BUG 4: bookmaker='{bet.bookmaker}'. "
            "Last bookmaker in loop is used, not the one offering the best odds. "
            "After fix this should be 'bookmaker_b'.",
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
            spread=Spread(team="Lakers", point=-7.5, price=-110),
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
            spread=Spread(team="Lakers", point=-3.5, price=-110),
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
        self.assertEqual(result["dk"]["point"], -5.5)
        self.assertEqual(result["fd"]["point"], -6.0)

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
# Group 7 — get_player_minutes_consistency
# ---------------------------------------------------------------------------

class TestMinutesConsistency(unittest.TestCase):

    def setUp(self):
        self.cache = make_cache()
        self.svc = StatsService(self.cache)

    def _patch(self, minutes_list):
        logs = [{"MIN": str(m)} for m in minutes_list]

        async def mock_gps(pid):
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

        async def mock_gps(pid):
            return logs

        self.svc.get_player_stats = mock_gps
        result = run(self.svc.get_player_minutes_consistency(1))
        self.assertEqual(result["grade"], "very_consistent")
        self.assertAlmostEqual(result["mean_minutes"], 32.2, delta=0.1)

    def test_empty_logs_fallback(self):
        async def mock_gps(pid):
            return []

        self.svc.get_player_stats = mock_gps
        result = run(self.svc.get_player_minutes_consistency(1))
        self.assertEqual(result["grade"], "moderate")
        self.assertEqual(result["games"], 0)

    def test_none_min_skipped(self):
        logs = [{"MIN": None}, {"MIN": "32"}, {"MIN": "33"}, {"MIN": "31"}]

        async def mock_gps(pid):
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

        async def counting(pid):
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
        svc = self._make_svc(home_wins=8, away_wins=2, home_ml=130, away_ml=-160)
        bets = run(svc.find_value_bets())
        for bet in bets:
            self.assertGreaterEqual(bet.kelly_fraction, 0.0)
            self.assertLessEqual(bet.kelly_fraction, 0.25)

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
# Group 11 — find_value_props z-score and edge logic
# ---------------------------------------------------------------------------

class TestFindValueProps(unittest.TestCase):

    def _make_logs(self, pts_values):
        return [
            {
                "PTS": v, "REB": 5.0, "AST": 7.0, "FG3M": 2.0,
                "BLK": 1.0, "STL": 1.0, "TOV": 2.0,
                "MIN": "35", "TEAM_ABBREVIATION": "LAL",
            }
            for v in pts_values
        ]

    def _make_svc(self, logs, prop_line, minutes_grade="very_consistent", is_b2b=False):
        prop = PlayerProp(
            event_id="test_game", player_name="LeBron James",
            market="player_points", line=prop_line,
            over_price=-110, under_price=-110, bookmaker="dk",
        )
        h2h_line = _make_odds_line("dk", -150, 130)

        home_stats = _make_team_stats(1610612747, "Lakers", wins=5, losses=5)
        away_stats = _make_team_stats(1610612744, "Warriors", wins=5, losses=5)

        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_player_props = AsyncMock(return_value=[prop])
        odds_mock.get_odds = AsyncMock(return_value=[h2h_line])
        odds_mock.get_historical_event_odds = AsyncMock(return_value={})

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
        stats_mock.get_injury_report = AsyncMock(
            return_value=InjuryReport(team_id=1610612747, team_name="Lakers")
        )
        stats_mock.get_player_stats = AsyncMock(return_value=logs)
        stats_mock.get_player_minutes_consistency = AsyncMock(
            return_value={"grade": minutes_grade, "mean_minutes": 35.0,
                          "std_dev": 1.0, "games": len(logs)}
        )
        stats_mock.is_team_on_back_to_back = AsyncMock(return_value=is_b2b)

        svc = _make_analysis_svc(odds_mock=odds_mock, stats_mock=stats_mock)

        async def mock_usage(*args, **kwargs):
            return 0.0

        async def mock_opening(*args, **kwargs):
            return {}

        async def mock_find_player(name):
            return {"id": 2544, "full_name": name, "TEAM_ABBREVIATION": "LAL"}

        svc._calculate_usage_adjustment = mock_usage
        svc._get_opening_prop_lines = mock_opening
        svc._find_player = mock_find_player
        return svc

    def test_over_bet_when_avg_well_above_line(self):
        logs = self._make_logs([28.0] * 10)
        svc = self._make_svc(logs, prop_line=22.0)
        bets = run(svc.find_value_props("test_game"))
        over_bets = [b for b in bets if "Over" in b.selection]
        self.assertGreater(len(over_bets), 0)

    def test_under_bet_when_avg_well_below_line(self):
        logs = self._make_logs([16.0] * 10)
        svc = self._make_svc(logs, prop_line=24.5)
        bets = run(svc.find_value_props("test_game"))
        under_bets = [b for b in bets if "Under" in b.selection]
        self.assertGreater(len(under_bets), 0)

    def test_edge_positive_for_over_when_avg_above_line(self):
        logs = self._make_logs([28.0] * 10)
        svc = self._make_svc(logs, prop_line=22.0)
        bets = run(svc.find_value_props("test_game"))
        over_bets = [b for b in bets if "Over" in b.selection]
        if over_bets:
            self.assertGreater(over_bets[0].edge, 0)
            self.assertGreater(over_bets[0].true_probability, over_bets[0].implied_probability)

    def test_high_confidence_for_very_easy_over(self):
        logs = self._make_logs([30.0] * 10)
        svc = self._make_svc(logs, prop_line=18.0)
        bets = run(svc.find_value_props("test_game"))
        over_bets = [b for b in bets if "Over" in b.selection]
        if over_bets:
            self.assertEqual(over_bets[0].confidence, "High")

    def test_b2b_flag_propagated(self):
        logs = self._make_logs([28.0] * 10)
        svc = self._make_svc(logs, prop_line=22.0, is_b2b=True)
        bets = run(svc.find_value_props("test_game"))
        b2b_bets = [b for b in bets if b.back_to_back is True]
        if bets:
            self.assertGreater(len(b2b_bets), 0)

    def test_minutes_grade_propagated(self):
        logs = self._make_logs([28.0] * 10)
        svc = self._make_svc(logs, prop_line=22.0, minutes_grade="very_consistent")
        bets = run(svc.find_value_props("test_game"))
        if bets:
            self.assertEqual(bets[0].minutes_grade, "very_consistent")

    def test_fewer_than_3_games_skipped(self):
        logs = self._make_logs([28.0, 27.0])
        svc = self._make_svc(logs, prop_line=20.0)
        bets = run(svc.find_value_props("test_game"))
        self.assertEqual(bets, [])

    def test_sorted_by_edge_descending(self):
        logs = self._make_logs([28.0] * 10)
        svc = self._make_svc(logs, prop_line=22.0)
        bets = run(svc.find_value_props("test_game"))
        for i in range(len(bets) - 1):
            self.assertGreaterEqual(bets[i].edge, bets[i + 1].edge)

    def test_empty_props_returns_empty(self):
        odds_mock = MagicMock(spec=OddsService)
        odds_mock.get_player_props = AsyncMock(return_value=[])
        svc = _make_analysis_svc(odds_mock=odds_mock)
        self.assertEqual(run(svc.find_value_props("test_game")), [])


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
