from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional

import httpx

from cache.ttl_cache import RedisCache, INJURIES, SCHEDULE, STATS
from config import settings
from models.schemas import (
    Game,
    HeadToHeadResult,
    InjuryPlayer,
    InjuryReport,
    Player,
    Team,
    TeamStats,
)

logger = logging.getLogger(__name__)


class StatsService:
    def __init__(self, cache: RedisCache) -> None:
        self.cache = cache

    # ------------------------------------------------------------------
    # nba_api helpers — imported lazily to avoid slow module-level import
    # ------------------------------------------------------------------

    def _get_scoreboard(self, game_date: date):
        from nba_api.stats.endpoints import scoreboard

        return scoreboard.Scoreboard(game_date=game_date.strftime("%Y-%m-%d"))

    def _get_team_game_logs(self, team_id: int, last_n: int):
        from nba_api.stats.endpoints import teamgamelogs

        return teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id,
            last_n_games_nullable=last_n,
            season_nullable=settings.nba_season,
        )

    def _get_player_game_logs(self, player_id: int):
        from nba_api.stats.endpoints import playergamelogs

        return playergamelogs.PlayerGameLogs(
            player_id_nullable=player_id,
            season_nullable=settings.nba_season,
        )

    def _get_standings(self):
        from nba_api.stats.endpoints import leaguestandingsv3

        return leaguestandingsv3.LeagueStandingsV3(season=settings.nba_season)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def get_schedule(self, game_date: Optional[date] = None) -> list[Game]:
        if game_date is None:
            game_date = date.today()

        cache_key = f"schedule:{game_date}"
        cached = await self.cache.get(cache_key)
        if cached:
            return [Game(**g) for g in cached]

        try:
            print(f"[API CALL]   nba_api → Scoreboard.get_data_frame() for {game_date}")
            board = self._get_scoreboard(game_date)
            games_df = board.game_header.get_data_frame()
            games: list[Game] = []
            for _, row in games_df.iterrows():
                games.append(
                    Game(
                        id=str(row.get("GAME_ID", "")),
                        home_team=str(row.get("HOME_TEAM_ABBREVIATION", "")),
                        away_team=str(row.get("VISITOR_TEAM_ABBREVIATION", "")),
                        commence_time=datetime.combine(game_date, datetime.min.time()),
                        status=str(row.get("GAME_STATUS_TEXT", "")),
                    )
                )
        except Exception as exc:
            logger.error("nba_api get_schedule error: %s", exc)
            games = []

        await self.cache.set(cache_key, [g.model_dump() for g in games], SCHEDULE)
        return games

    async def get_team_stats(self, team_id: int, last_n: int = 10) -> TeamStats:
        cache_key = f"team_stats:{team_id}:{last_n}"
        cached = await self.cache.get(cache_key)
        if cached:
            return TeamStats(**cached)

        try:
            print(f"[API CALL]   nba_api → TeamGameLogs.get_data_frame() team={team_id} last_n={last_n}")
            logs = self._get_team_game_logs(team_id, last_n)
            df = logs.team_game_logs.get_data_frame()

            if df.empty:
                raise ValueError("No game log data returned")

            wins = int((df["WL"] == "W").sum())
            losses = int((df["WL"] == "L").sum())
            played = len(df)
            ppg = float(df["PTS"].mean())
            papg = float(df["PTS_OPP"].mean()) if "PTS_OPP" in df.columns else 0.0
            pace = float(df["PACE"].mean()) if "PACE" in df.columns else None
            off_rtg = float(df["OFF_RATING"].mean()) if "OFF_RATING" in df.columns else None
            def_rtg = float(df["DEF_RATING"].mean()) if "DEF_RATING" in df.columns else None

            stats = TeamStats(
                team_id=team_id,
                team_name=str(df["TEAM_NAME"].iloc[0]),
                games_played=played,
                wins=wins,
                losses=losses,
                win_pct=round(wins / played, 3) if played else 0.0,
                points_per_game=round(ppg, 1),
                points_allowed_per_game=round(papg, 1),
                pace=round(pace, 1) if pace else None,
                offensive_rating=round(off_rtg, 1) if off_rtg else None,
                defensive_rating=round(def_rtg, 1) if def_rtg else None,
                net_rating=round(off_rtg - def_rtg, 1) if off_rtg and def_rtg else None,
                last_n_games=last_n,
            )
        except Exception as exc:
            logger.error("get_team_stats error for team %d: %s", team_id, exc)
            stats = TeamStats(
                team_id=team_id,
                team_name="Unknown",
                games_played=0,
                wins=0,
                losses=0,
                win_pct=0.0,
                points_per_game=0.0,
                points_allowed_per_game=0.0,
                last_n_games=last_n,
            )

        await self.cache.set(cache_key, stats.model_dump(), STATS)
        return stats

    async def get_player_stats(self, player_id: int) -> dict[str, Any]:
        cache_key = f"player_stats:{player_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        try:
            print(f"[API CALL]   nba_api → PlayerGameLogs.get_data_frame() player={player_id}")
            logs = self._get_player_game_logs(player_id)
            df = logs.player_game_logs.get_data_frame()
            result = df.head(10).to_dict(orient="records")
        except Exception as exc:
            logger.error("get_player_stats error for player %d: %s", player_id, exc)
            result = []

        await self.cache.set(cache_key, result, STATS)
        return result

    async def get_head_to_head(
        self, team1_id: int, team2_id: int, num_games: int = 10
    ) -> HeadToHeadResult:
        cache_key = f"h2h:{min(team1_id,team2_id)}:{max(team1_id,team2_id)}:{num_games}"
        cached = await self.cache.get(cache_key)
        if cached:
            return HeadToHeadResult(**cached)

        try:
            print(f"[API CALL]   nba_api → TeamGameLogs.get_data_frame() team={team1_id} last_n=82")
            logs1 = self._get_team_game_logs(team1_id, 82)
            df1 = logs1.team_game_logs.get_data_frame()
            team1_name = str(df1["TEAM_NAME"].iloc[0]) if not df1.empty else "Team1"

            print(f"[API CALL]   nba_api → TeamGameLogs.get_data_frame() team={team2_id} last_n=82")
            logs2 = self._get_team_game_logs(team2_id, 82)
            df2 = logs2.team_game_logs.get_data_frame()
            team2_name = str(df2["TEAM_NAME"].iloc[0]) if not df2.empty else "Team2"

            # Filter games where team1 played team2
            matchups = df1[df1["MATCHUP"].str.contains(
                df2["TEAM_ABBREVIATION"].iloc[0] if not df2.empty else "", na=False
            )].head(num_games)

            team1_wins = int((matchups["WL"] == "W").sum())
            team2_wins = int((matchups["WL"] == "L").sum())
            team1_avg = float(matchups["PTS"].mean()) if not matchups.empty else 0.0
            # PTS_OPP may not be available — use a safe fallback
            if "PTS_OPP" in matchups.columns and not matchups.empty:
                team2_avg = float(matchups["PTS_OPP"].mean())
            else:
                team2_avg = 0.0

            result = HeadToHeadResult(
                team1_id=team1_id,
                team1_name=team1_name,
                team2_id=team2_id,
                team2_name=team2_name,
                total_games=len(matchups),
                team1_wins=team1_wins,
                team2_wins=team2_wins,
                team1_avg_score=round(team1_avg, 1),
                team2_avg_score=round(team2_avg, 1),
                avg_total_score=round(team1_avg + team2_avg, 1),
            )
        except Exception as exc:
            logger.error("get_head_to_head error: %s", exc)
            result = HeadToHeadResult(
                team1_id=team1_id,
                team1_name="Team1",
                team2_id=team2_id,
                team2_name="Team2",
                total_games=0,
                team1_wins=0,
                team2_wins=0,
                team1_avg_score=0.0,
                team2_avg_score=0.0,
                avg_total_score=0.0,
            )

        await self.cache.set(cache_key, result.model_dump(), STATS)
        return result

    async def get_injury_report(self, team_id: int) -> InjuryReport:
        cache_key = f"injuries:{team_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return InjuryReport(**cached)

        players: list[InjuryPlayer] = []
        team_name = "Unknown"

        if settings.sportsradar_api_key:
            try:
                url = (
                    f"{settings.sportsradar_base_url}/seasons/{settings.nba_season_year}/REG/injuries.json"
                )
                params = {"api_key": settings.sportsradar_api_key}
                print(f"[API CALL]   SportsRadar → /seasons/{settings.nba_season_year}/REG/injuries.json team={team_id}")
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                for team in data.get("teams", []):
                    if str(team.get("id")) == str(team_id):
                        team_name = team.get("name", "Unknown")
                        for p in team.get("players", []):
                            injuries = p.get("injuries", [])
                            if injuries:
                                latest = injuries[0]
                                players.append(
                                    InjuryPlayer(
                                        player_name=p.get("full_name", ""),
                                        status=latest.get("status", "Unknown"),
                                        injury_type=latest.get("desc", None),
                                    )
                                )
                        break
            except Exception as exc:
                logger.warning("SportsRadar injury report error: %s", exc)

        report = InjuryReport(
            team_id=team_id,
            team_name=team_name,
            players=players,
        )
        await self.cache.set(cache_key, report.model_dump(), INJURIES)
        return report

    async def get_standings(self) -> list[dict[str, Any]]:
        cache_key = f"standings:{settings.nba_season}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        try:
            print(f"[API CALL]   nba_api → LeagueStandingsV3.get_data_frame()")
            standings = self._get_standings()
            df = standings.standings.get_data_frame()
            records = df[
                [
                    "TeamName",
                    "Conference",
                    "PlayoffRank",
                    "WINS",
                    "LOSSES",
                    "WinPCT",
                    "HOME",
                    "ROAD",
                    "L10",
                ]
            ].to_dict(orient="records")
        except Exception as exc:
            logger.error("get_standings error: %s", exc)
            records = []

        await self.cache.set(cache_key, records, STATS)
        return records

    async def get_all_teams(self) -> list[Team]:
        cache_key = "all_teams"
        cached = await self.cache.get(cache_key)
        if cached:
            return [Team(**t) for t in cached]

        try:
            print("[API CALL]   nba_api → teams.get_teams()")
            from nba_api.stats.static import teams as nba_teams

            raw = nba_teams.get_teams()
            team_list = [
                Team(
                    id=t["id"],
                    name=t["full_name"],
                    abbreviation=t["abbreviation"],
                    city=t["city"],
                )
                for t in raw
            ]
        except Exception as exc:
            logger.error("get_all_teams error: %s", exc)
            team_list = []

        await self.cache.set(cache_key, [t.model_dump() for t in team_list], SCHEDULE)
        return team_list

    async def get_all_players(self) -> list[Player]:
        cache_key = "all_players"
        cached = await self.cache.get(cache_key)
        if cached:
            return [Player(**p) for p in cached]

        try:
            print("[API CALL]   nba_api → players.get_active_players()")
            from nba_api.stats.static import players as nba_players

            raw = nba_players.get_active_players()
            player_list = [
                Player(
                    id=p["id"],
                    full_name=p["full_name"],
                    is_active=True,
                )
                for p in raw
            ]
        except Exception as exc:
            logger.error("get_all_players error: %s", exc)
            player_list = []

        await self.cache.set(
            cache_key, [p.model_dump() for p in player_list], SCHEDULE
        )
        return player_list

    def find_team_by_name(self, name: str) -> Optional[dict[str, Any]]:
        """Synchronous helper to resolve a team name/abbreviation to an id."""
        try:
            from nba_api.stats.static import teams as nba_teams

            name_lower = name.lower()
            for t in nba_teams.get_teams():
                if (
                    name_lower in t["full_name"].lower()
                    or name_lower == t["abbreviation"].lower()
                    or name_lower == t["nickname"].lower()
                ):
                    return t
        except Exception:
            pass
        return None
