from __future__ import annotations

import logging
import statistics
from datetime import date, datetime, timedelta
from typing import Any, Optional

import httpx

from cache.ttl_cache import RedisCache, ADVANCED_STATS, INJURIES, SCHEDULE, STATS
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
        from nba_api.stats.endpoints import scoreboardv3

        return scoreboardv3.ScoreboardV3(game_date=game_date.strftime("%Y-%m-%d"))

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

    def _get_league_team_stats_advanced(self, last_n: int):
        from nba_api.stats.endpoints import leaguedashteamstats

        return leaguedashteamstats.LeagueDashTeamStats(
            season=settings.nba_season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            last_n_games=last_n,
        )

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
            raw_games = board.get_dict()["scoreboard"]["games"]
            games: list[Game] = []
            for g in raw_games:
                games.append(
                    Game(
                        id=str(g.get("gameId", "")),
                        home_team=g["homeTeam"].get("teamTricode", ""),
                        away_team=g["awayTeam"].get("teamTricode", ""),
                        commence_time=datetime.combine(game_date, datetime.min.time()),
                        status=str(g.get("gameStatusText", "")),
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

        # --- Base stats from TeamGameLogs ---
        try:
            print(f"[API CALL]   nba_api → TeamGameLogs.get_data_frame() team={team_id} last_n={last_n}")
            logs = self._get_team_game_logs(team_id, last_n)
            df = logs.team_game_logs.get_data_frame().head(last_n)

            if df.empty:
                raise ValueError("No game log data returned")

            wins = int((df["WL"] == "W").sum())
            losses = int((df["WL"] == "L").sum())
            played = len(df)
            ppg = float(df["PTS"].mean())
            if "PTS_OPP" in df.columns:
                papg = float(df["PTS_OPP"].mean())
            elif "PLUS_MINUS" in df.columns:
                papg = float((df["PTS"] - df["PLUS_MINUS"]).mean())
            else:
                papg = 0.0
            team_name = str(df["TEAM_NAME"].iloc[0])
        except Exception as exc:
            logger.error("get_team_stats error for team %d: %s", team_id, exc)
            await self.cache.set(
                cache_key,
                TeamStats(
                    team_id=team_id, team_name="Unknown", games_played=0,
                    wins=0, losses=0, win_pct=0.0, points_per_game=0.0,
                    points_allowed_per_game=0.0, last_n_games=last_n,
                ).model_dump(),
                STATS,
            )
            return TeamStats(
                team_id=team_id, team_name="Unknown", games_played=0,
                wins=0, losses=0, win_pct=0.0, points_per_game=0.0,
                points_allowed_per_game=0.0, last_n_games=last_n,
            )

        # --- Advanced stats from LeagueDashTeamStats ---
        pace: Optional[float] = None
        off_rtg: Optional[float] = None
        def_rtg: Optional[float] = None
        net_rtg: Optional[float] = None

        adv_cache_key = f"advanced_stats:{team_id}:{last_n}"
        adv_cached = await self.cache.get(adv_cache_key)
        if adv_cached:
            pace = adv_cached.get("pace")
            off_rtg = adv_cached.get("off_rtg")
            def_rtg = adv_cached.get("def_rtg")
            net_rtg = adv_cached.get("net_rtg")
        else:
            try:
                print(f"[API CALL]   nba_api → LeagueDashTeamStats (Advanced) team={team_id} last_n={last_n}")
                adv = self._get_league_team_stats_advanced(last_n)
                adv_df = adv.league_dash_team_stats.get_data_frame()
                team_row = adv_df[adv_df["TEAM_ID"] == team_id]
                if not team_row.empty:
                    pace = float(team_row["PACE"].iloc[0]) if "PACE" in adv_df.columns else None
                    off_rtg = float(team_row["OFF_RATING"].iloc[0]) if "OFF_RATING" in adv_df.columns else None
                    def_rtg = float(team_row["DEF_RATING"].iloc[0]) if "DEF_RATING" in adv_df.columns else None
                    net_rtg = float(team_row["NET_RATING"].iloc[0]) if "NET_RATING" in adv_df.columns else None
                await self.cache.set(
                    adv_cache_key,
                    {"pace": pace, "off_rtg": off_rtg, "def_rtg": def_rtg, "net_rtg": net_rtg},
                    ADVANCED_STATS,
                )
            except Exception as exc:
                logger.warning("LeagueDashTeamStats advanced fetch failed for team %d: %s", team_id, exc)

        stats = TeamStats(
            team_id=team_id,
            team_name=team_name,
            games_played=played,
            wins=wins,
            losses=losses,
            win_pct=round(wins / played, 3) if played else 0.0,
            points_per_game=round(ppg, 1),
            points_allowed_per_game=round(papg, 1),
            pace=round(pace, 1) if pace is not None else None,
            offensive_rating=round(off_rtg, 1) if off_rtg is not None else None,
            defensive_rating=round(def_rtg, 1) if def_rtg is not None else None,
            net_rating=round(net_rtg, 1) if net_rtg is not None else None,
            last_n_games=last_n,
        )

        await self.cache.set(cache_key, stats.model_dump(), STATS)
        return stats

    async def get_player_stats(self, player_id: int, last_n: int = 10) -> list[dict[str, Any]]:
        cache_key = f"player_stats:{player_id}:{last_n}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        try:
            print(f"[API CALL]   nba_api → PlayerGameLogs.get_data_frame() player={player_id} last_n={last_n}")
            logs = self._get_player_game_logs(player_id)
            df = logs.player_game_logs.get_data_frame()
            result = df.head(last_n).to_dict(orient="records")
        except Exception as exc:
            logger.error("get_player_stats error for player %d: %s", player_id, exc)
            result = []

        await self.cache.set(cache_key, result, STATS)
        return result

    async def get_head_to_head(
        self, team1_id: int, team2_id: int, num_games: int = 10
    ) -> HeadToHeadResult:
        cache_key = f"h2h:{team1_id}:{team2_id}:{num_games}"
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
            if "PTS_OPP" in matchups.columns and not matchups.empty:
                team2_avg = float(matchups["PTS_OPP"].mean())
            elif "PLUS_MINUS" in matchups.columns and not matchups.empty:
                team2_avg = float((matchups["PTS"] - matchups["PLUS_MINUS"]).mean())
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
                url = f"{settings.sportsradar_base_url}/league/injuries.json"
                params = {"api_key": settings.sportsradar_api_key}
                print(f"[API CALL]   SportsRadar → /league/injuries.json team={team_id}")
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                for team in data.get("teams", []):
                    if str(team.get("reference")) == str(team_id):
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
                    "TeamCity",
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

    async def get_player_minutes_consistency(self, player_id: int, last_n: int = 10) -> dict:
        """Returns mean minutes, std_dev, and consistency grade for last N games."""
        cache_key = f"minutes_consistency:{player_id}:{last_n}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        try:
            game_logs = await self.get_player_stats(player_id, last_n=last_n)
            minutes: list[float] = []
            for g in game_logs[:last_n]:
                raw = g.get("MIN")
                if raw is None:
                    continue
                raw_str = str(raw)
                if ":" in raw_str:
                    parts = raw_str.split(":")
                    minutes.append(float(parts[0]) + float(parts[1]) / 60)
                else:
                    try:
                        minutes.append(float(raw_str))
                    except ValueError:
                        continue

            if not minutes:
                result = {"mean_minutes": 0.0, "std_dev": 0.0, "grade": "moderate", "games": 0}
            else:
                mean_min = statistics.mean(minutes)
                std_dev = statistics.stdev(minutes) if len(minutes) >= 2 else 0.0
                if std_dev < 3:
                    grade = "very_consistent"
                elif std_dev < 5:
                    grade = "consistent"
                elif std_dev < 8:
                    grade = "moderate"
                else:
                    grade = "volatile"
                result = {
                    "mean_minutes": round(mean_min, 1),
                    "std_dev": round(std_dev, 2),
                    "grade": grade,
                    "games": len(minutes),
                }
        except Exception as exc:
            logger.warning("get_player_minutes_consistency error for player %d: %s", player_id, exc)
            result = {"mean_minutes": 0.0, "std_dev": 0.0, "grade": "moderate", "games": 0}

        await self.cache.set(cache_key, result, STATS)
        return result

    async def is_team_on_back_to_back(self, team_id: int, game_date: date) -> bool:
        """Returns True if the team played the day before game_date."""
        cache_key = f"b2b:{team_id}:{game_date}"
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return bool(cached)

        try:
            # Resolve team abbreviation from static data
            team_abbrev: Optional[str] = None
            try:
                from nba_api.stats.static import teams as nba_teams
                for t in nba_teams.get_teams():
                    if t["id"] == team_id:
                        team_abbrev = t["abbreviation"]
                        break
            except Exception:
                pass

            yesterday = game_date - timedelta(days=1)
            schedule = await self.get_schedule(yesterday)
            is_b2b = any(
                g.home_team == team_abbrev or g.away_team == team_abbrev
                for g in schedule
            ) if team_abbrev else False
        except Exception as exc:
            logger.warning("is_team_on_back_to_back error for team %d: %s", team_id, exc)
            is_b2b = False

        await self.cache.set(cache_key, is_b2b, SCHEDULE)
        return is_b2b

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
