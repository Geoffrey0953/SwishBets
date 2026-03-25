from __future__ import annotations

import logging
from typing import Any

from cache.ttl_cache import RedisCache, STATS
from config import settings

logger = logging.getLogger(__name__)

# Maps our internal stat category to the opponent column in LeagueDashTeamStats (Opponent mode)
_STAT_TO_OPP_COLUMN: dict[str, str] = {
    "PTS": "OPP_PTS",
    "REB": "OPP_REB",
    "AST": "OPP_AST",
    "FG3M": "OPP_FG3M",
}


class DefenseService:
    def __init__(self, cache: RedisCache) -> None:
        self.cache = cache

    async def get_opponent_def_rank(
        self,
        opponent_team_id: int,
        stat_category: str,
    ) -> dict[str, Any]:
        """Returns opponent defensive rank (1=best/fewest allowed, 30=worst) for a stat category.

        Returns dict with keys: rank (int), grade (str), value (float).
        Returns {} on failure or unsupported category.
        """
        opp_col = _STAT_TO_OPP_COLUMN.get(stat_category)
        if not opp_col:
            return {}

        cache_key = f"def_rank:{opponent_team_id}:{stat_category}:{settings.nba_season}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        try:
            print(
                f"[API CALL]   nba_api → LeagueDashTeamStats (Opponent) "
                f"team={opponent_team_id} stat={stat_category}"
            )
            from nba_api.stats.endpoints import leaguedashteamstats

            result = leaguedashteamstats.LeagueDashTeamStats(
                season=settings.nba_season,
                measure_type_detailed_defense="Opponent",
                per_mode_detailed="PerGame",
            )
            df = result.league_dash_team_stats.get_data_frame()

            if opp_col not in df.columns:
                logger.warning("DefenseService: column %s not found in opponent stats", opp_col)
                return {}

            # Sort ascending → rank 1 = fewest allowed = best defense
            df_sorted = df.sort_values(opp_col, ascending=True).reset_index(drop=True)
            df_sorted["_rank"] = df_sorted.index + 1

            target_row = df_sorted[df_sorted["TEAM_ID"] == opponent_team_id]
            if target_row.empty:
                return {}

            rank = int(target_row["_rank"].iloc[0])
            value = float(target_row[opp_col].iloc[0])

            if rank <= 8:
                grade = "elite"
            elif rank <= 22:
                grade = "average"
            else:
                grade = "weak"

            outcome = {"rank": rank, "grade": grade, "value": round(value, 1)}
            await self.cache.set(cache_key, outcome, STATS)
            return outcome

        except Exception as exc:
            logger.warning(
                "DefenseService.get_opponent_def_rank failed for team %d stat %s: %s",
                opponent_team_id,
                stat_category,
                exc,
            )
            return {}
