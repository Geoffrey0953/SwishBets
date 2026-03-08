from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from swishbets.cache.ttl_cache import RedisCache
from swishbets.services.stats_service import StatsService

_cache: RedisCache
_stats_service: StatsService


def register(mcp: FastMCP, cache: RedisCache, stats_service: StatsService) -> None:
    global _cache, _stats_service
    _cache = cache
    _stats_service = stats_service

    @mcp.tool()
    async def get_injury_report(team_name: str) -> str:
        """Return the latest injury report for an NBA team.

        Args:
            team_name: Team name, city, or abbreviation (e.g. "Lakers", "LAL", "Los Angeles Lakers").
        """
        team = _stats_service.find_team_by_name(team_name)
        if not team:
            return f"Could not find team: '{team_name}'. Try a different name or abbreviation."

        report = await _stats_service.get_injury_report(team["id"])

        if not report.players:
            return f"No injury report data available for {report.team_name}."

        rows = [f"**Injury Report — {report.team_name}**\n",
                "| Player | Status | Injury |",
                "|--------|--------|--------|"]
        for p in report.players:
            rows.append(
                f"| {p.player_name} | {p.status} | {p.injury_type or '—'} |"
            )
        return "\n".join(rows)

    @mcp.tool()
    async def get_team_stats(team_name: str, last_n_games: int = 10) -> str:
        """Return recent team performance stats including pace, offensive/defensive ratings.

        Args:
            team_name: Team name, city, or abbreviation.
            last_n_games: Number of recent games to average over (default 10).
        """
        team = _stats_service.find_team_by_name(team_name)
        if not team:
            return f"Could not find team: '{team_name}'."

        stats = await _stats_service.get_team_stats(team["id"], last_n=last_n_games)

        form = f"{stats.wins}W-{stats.losses}L ({stats.win_pct:.1%})"
        lines = [
            f"**{stats.team_name} — Last {last_n_games} Games**\n",
            f"- Record: {form}",
            f"- Points/Game: {stats.points_per_game}",
            f"- Points Allowed/Game: {stats.points_allowed_per_game}",
        ]
        if stats.pace:
            lines.append(f"- Pace: {stats.pace}")
        if stats.offensive_rating:
            lines.append(f"- Offensive Rating: {stats.offensive_rating}")
        if stats.defensive_rating:
            lines.append(f"- Defensive Rating: {stats.defensive_rating}")
        if stats.net_rating:
            lines.append(f"- Net Rating: {stats.net_rating:+.1f}")
        return "\n".join(lines)

    @mcp.tool()
    async def get_head_to_head(team1: str, team2: str) -> str:
        """Return historical head-to-head record and average scores between two teams.

        Args:
            team1: First team name, city, or abbreviation.
            team2: Second team name, city, or abbreviation.
        """
        t1 = _stats_service.find_team_by_name(team1)
        t2 = _stats_service.find_team_by_name(team2)

        if not t1:
            return f"Could not find team: '{team1}'."
        if not t2:
            return f"Could not find team: '{team2}'."

        h2h = await _stats_service.get_head_to_head(t1["id"], t2["id"])

        return (
            f"**Head-to-Head: {h2h.team1_name} vs {h2h.team2_name}**\n\n"
            f"- Games analyzed: {h2h.total_games}\n"
            f"- {h2h.team1_name}: **{h2h.team1_wins}W** | Avg score: {h2h.team1_avg_score}\n"
            f"- {h2h.team2_name}: **{h2h.team2_wins}W** | Avg score: {h2h.team2_avg_score}\n"
            f"- Average total: {h2h.avg_total_score}"
        )
