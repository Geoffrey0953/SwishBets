"""SwishBets MCP Integration Tests — calls every tool against real APIs and prints results.

Run:
    .venv/bin/python integration_test.py

Requirements:
    - Redis running (redis://localhost:6379)
    - .env file with THE_ODDS_API_KEY (and optionally SPORTSRADAR_API_KEY, OPENWEATHER_API_KEY)
"""
from __future__ import annotations

import asyncio
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache.ttl_cache import RedisCache
from config import settings
from services.analysis_service import AnalysisService
from services.defense_service import DefenseService
from services.odds_service import OddsService
from services.stats_service import StatsService
import tools.odds as odds_tools
import tools.stats as stats_tools
import tools.analysis as analysis_tools

# ---------------------------------------------------------------------------
# CaptureMCP — registers tools as plain callables without starting a server
# ---------------------------------------------------------------------------

class CaptureMCP:
    """Fake FastMCP that captures tool closures via the @mcp.tool() decorator."""
    def __init__(self):
        self.tools: dict[str, callable] = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


# ---------------------------------------------------------------------------
# Printer helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 80

def header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def section(name: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  ► {name}")
    print(f"{'─' * 60}")

def print_result(result: str) -> None:
    print(result)

def print_error(exc: Exception) -> None:
    print(f"[ERROR] {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup_services():
    cache = RedisCache(settings.redis_url)
    odds_svc = OddsService(cache)
    stats_svc = StatsService(cache)
    defense_svc = DefenseService(cache)
    analysis_svc = AnalysisService(odds_svc, stats_svc, defense_svc)

    mcp = CaptureMCP()
    odds_tools.register(mcp, cache, odds_svc)
    stats_tools.register(mcp, cache, stats_svc)
    analysis_tools.register(mcp, cache, analysis_svc, odds_svc)

    return mcp.tools, odds_svc


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

async def run_all():
    header("SwishBets MCP Integration Test Suite")
    print(f"  Odds API key : {'✓ set' if settings.the_odds_api_key else '✗ missing'}")
    print(f"  SportsRadar  : {'✓ set' if settings.sportsradar_api_key else '✗ missing'}")
    print(f"  OpenWeather  : {'✓ set' if settings.openweather_api_key else '✗ missing'}")
    print(f"  Redis URL    : {settings.redis_url}")

    tools, odds_svc = await setup_services()

    # ------------------------------------------------------------------ #
    # 1. get_tonight_games                                                  #
    # ------------------------------------------------------------------ #
    section("1 / 13 — get_tonight_games (no args → today)")
    game_id: str | None = None
    try:
        t0 = time.time()
        result = await tools["get_tonight_games"]()
        print_result(result)
        print(f"\n  [OK] {time.time() - t0:.2f}s")

        # Extract first game_id for follow-up tools
        for line in result.splitlines():
            if "`" in line:
                parts = line.split("`")
                if len(parts) >= 2 and len(parts[-2]) > 20:
                    game_id = parts[-2]
                    break
    except Exception as exc:
        print_error(exc)
        traceback.print_exc()

    if not game_id:
        # Try fetching directly as fallback
        try:
            games = await odds_svc.get_games()
            if games:
                game_id = games[0].id
                print(f"\n  (Fallback) Using game_id: {game_id}")
        except Exception:
            pass

    if not game_id:
        print("\n  [WARN] No game_id found — skipping game-specific tools.")
    else:
        print(f"\n  Using game_id: {game_id}")

    # ------------------------------------------------------------------ #
    # 2. get_odds                                                           #
    # ------------------------------------------------------------------ #
    section("2 / 13 — get_odds")
    if game_id:
        try:
            t0 = time.time()
            result = await tools["get_odds"](game_id)
            print_result(result)
            print(f"\n  [OK] {time.time() - t0:.2f}s")
        except Exception as exc:
            print_error(exc)
    else:
        print("  [SKIP] No game_id available.")

    # ------------------------------------------------------------------ #
    # 3. get_player_props                                                   #
    # ------------------------------------------------------------------ #
    section("3 / 13 — get_player_props (all players)")
    if game_id:
        try:
            t0 = time.time()
            result = await tools["get_player_props"](game_id)
            lines = result.splitlines()
            # Print first 30 rows to keep output readable
            preview = "\n".join(lines[:32])
            if len(lines) > 32:
                preview += f"\n  ... ({len(lines) - 32} more rows)"
            print_result(preview)
            print(f"\n  [OK] {time.time() - t0:.2f}s  ({len(lines)} lines total)")
        except Exception as exc:
            print_error(exc)
    else:
        print("  [SKIP] No game_id available.")

    # ------------------------------------------------------------------ #
    # 4. get_player_props (filtered by player)                              #
    # ------------------------------------------------------------------ #
    section("3b / 13 — get_player_props (filtered: LeBron James)")
    if game_id:
        try:
            t0 = time.time()
            result = await tools["get_player_props"](game_id, player_name="LeBron")
            print_result(result)
            print(f"\n  [OK] {time.time() - t0:.2f}s")
        except Exception as exc:
            print_error(exc)
    else:
        print("  [SKIP] No game_id available.")

    # ------------------------------------------------------------------ #
    # 5. compare_books — h2h                                                #
    # ------------------------------------------------------------------ #
    section("4 / 13 — compare_books (h2h)")
    if game_id:
        try:
            t0 = time.time()
            result = await tools["compare_books"](game_id, "h2h")
            print_result(result)
            print(f"\n  [OK] {time.time() - t0:.2f}s")
        except Exception as exc:
            print_error(exc)
    else:
        print("  [SKIP] No game_id available.")

    # ------------------------------------------------------------------ #
    # 6. compare_books — spreads                                            #
    # ------------------------------------------------------------------ #
    section("4b / 13 — compare_books (spreads)")
    if game_id:
        try:
            t0 = time.time()
            result = await tools["compare_books"](game_id, "spreads")
            print_result(result)
            print(f"\n  [OK] {time.time() - t0:.2f}s")
        except Exception as exc:
            print_error(exc)
    else:
        print("  [SKIP] No game_id available.")

    # ------------------------------------------------------------------ #
    # 7. compare_books — totals                                             #
    # ------------------------------------------------------------------ #
    section("4c / 13 — compare_books (totals)")
    if game_id:
        try:
            t0 = time.time()
            result = await tools["compare_books"](game_id, "totals")
            print_result(result)
            print(f"\n  [OK] {time.time() - t0:.2f}s")
        except Exception as exc:
            print_error(exc)
    else:
        print("  [SKIP] No game_id available.")

    # ------------------------------------------------------------------ #
    # 8. get_pinnacle_line                                                   #
    # ------------------------------------------------------------------ #
    section("5 / 13 — get_pinnacle_line")
    if game_id:
        try:
            t0 = time.time()
            result = await tools["get_pinnacle_line"](game_id)
            print_result(result)
            print(f"\n  [OK] {time.time() - t0:.2f}s")
        except Exception as exc:
            print_error(exc)
    else:
        print("  [SKIP] No game_id available.")

    # ------------------------------------------------------------------ #
    # 9. get_line_movement                                                   #
    # ------------------------------------------------------------------ #
    section("6 / 13 — get_line_movement")
    if game_id:
        try:
            t0 = time.time()
            result = await tools["get_line_movement"](game_id)
            print_result(result)
            print(f"\n  [OK] {time.time() - t0:.2f}s")
        except Exception as exc:
            print_error(exc)
    else:
        print("  [SKIP] No game_id available.")

    # ------------------------------------------------------------------ #
    # 10. get_injury_report                                                  #
    # ------------------------------------------------------------------ #
    section("7 / 13 — get_injury_report (Lakers)")
    try:
        t0 = time.time()
        result = await tools["get_injury_report"]("Lakers")
        print_result(result)
        print(f"\n  [OK] {time.time() - t0:.2f}s")
    except Exception as exc:
        print_error(exc)

    # ------------------------------------------------------------------ #
    # 11. get_team_stats                                                     #
    # ------------------------------------------------------------------ #
    section("8 / 13 — get_team_stats (Lakers, last 10)")
    try:
        t0 = time.time()
        result = await tools["get_team_stats"]("Lakers", 10)
        print_result(result)
        print(f"\n  [OK] {time.time() - t0:.2f}s")
    except Exception as exc:
        print_error(exc)

    section("8b / 13 — get_team_stats (Celtics, last 5)")
    try:
        t0 = time.time()
        result = await tools["get_team_stats"]("Celtics", 5)
        print_result(result)
        print(f"\n  [OK] {time.time() - t0:.2f}s")
    except Exception as exc:
        print_error(exc)

    # ------------------------------------------------------------------ #
    # 12. get_head_to_head                                                   #
    # ------------------------------------------------------------------ #
    section("9 / 13 — get_head_to_head (Lakers vs Celtics)")
    try:
        t0 = time.time()
        result = await tools["get_head_to_head"]("Lakers", "Celtics")
        print_result(result)
        print(f"\n  [OK] {time.time() - t0:.2f}s")
    except Exception as exc:
        print_error(exc)

    # ------------------------------------------------------------------ #
    # 13. find_value_bets                                                    #
    # ------------------------------------------------------------------ #
    section("10 / 13 — find_value_bets (today)")
    try:
        t0 = time.time()
        result = await tools["find_value_bets"]()
        print_result(result)
        print(f"\n  [OK] {time.time() - t0:.2f}s")
    except Exception as exc:
        print_error(exc)

    # ------------------------------------------------------------------ #
    # 14. find_positive_ev                                                   #
    # ------------------------------------------------------------------ #
    section("11 / 13 — find_positive_ev (min edge 1%)")
    print("  (This scans every game × every book × every market — may take 30-90s)")
    try:
        t0 = time.time()
        result = await tools["find_positive_ev"](1.0)
        lines = result.splitlines()
        preview = "\n".join(lines[:35])
        if len(lines) > 35:
            preview += f"\n  ... ({len(lines) - 35} more rows)"
        print_result(preview)
        print(f"\n  [OK] {time.time() - t0:.2f}s  ({len(lines)} lines total)")
    except Exception as exc:
        print_error(exc)

    # ------------------------------------------------------------------ #
    # 15. find_arb                                                           #
    # ------------------------------------------------------------------ #
    section("12 / 13 — find_arb (min arb 0.1%)")
    print("  (Scans every game × every market for cross-book arb — may take 30-90s)")
    try:
        t0 = time.time()
        result = await tools["find_arb"](0.1)
        lines = result.splitlines()
        preview = "\n".join(lines[:35])
        if len(lines) > 35:
            preview += f"\n  ... ({len(lines) - 35} more rows)"
        print_result(preview)
        print(f"\n  [OK] {time.time() - t0:.2f}s  ({len(lines)} lines total)")
    except Exception as exc:
        print_error(exc)

    # ------------------------------------------------------------------ #
    # 16. get_weather_impact                                                 #
    # ------------------------------------------------------------------ #
    section("13 / 13 — get_weather_impact")
    if game_id:
        if settings.openweather_api_key:
            try:
                t0 = time.time()
                result = await tools["get_weather_impact"](game_id)
                print_result(result)
                print(f"\n  [OK] {time.time() - t0:.2f}s")
            except Exception as exc:
                print_error(exc)
        else:
            print("  [SKIP] OPENWEATHER_API_KEY not set in .env")
    else:
        print("  [SKIP] No game_id available.")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    header("Integration Test Complete")
    print("  All 13 MCP tools exercised with live API data.\n")


if __name__ == "__main__":
    asyncio.run(run_all())
