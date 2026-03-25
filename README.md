# SwishBets

NBA sports betting analysis server for [Claude Desktop](https://claude.ai/download) via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). Gives Claude live access to odds, player props, injury reports, team stats, and statistical value-bet recommendations.

## Tools

| Tool | Description |
|------|-------------|
| `get_tonight_games` | Tonight's NBA games with tip-off times and game IDs |
| `get_odds` | Moneyline, spread, and total odds across all sportsbooks |
| `get_player_props` | Player prop lines (points / assists / rebounds / threes) with optional player filter |
| `compare_books` | Best available number for a given market across all books |
| `get_line_movement` | How the spread has shifted since open (48-hour delta) |
| `get_injury_report` | Latest injury report for a team (via SportsRadar) |
| `get_team_stats` | Recent team performance — pace, offensive/defensive ratings, record |
| `get_head_to_head` | Historical head-to-head record and scoring averages between two teams |
| `find_value_bets` | Moneyline bets with >3% edge vs. 10-game win-rate baseline (home-court adjusted) |
| `find_value_props` | Player props with >3% edge, adjusted for trend, minutes consistency, back-to-back fatigue, pace, injury-driven usage shifts, and opponent defensive rank |
| `get_weather_impact` | Weather context for the venue city (NBA games are indoors — useful for travel/conditions) |

## Resources

| URI | Description |
|-----|-------------|
| `nba://teams` | All active NBA teams |
| `nba://players` | All active NBA players |
| `nba://standings` | Current NBA standings |
| `nba://schedule` | Today's NBA schedule |

## Requirements

- Python 3.11+
- Redis

## Setup

**1. Clone and install**
```bash
git clone https://github.com/your-username/SwishBets.git
cd SwishBets
python3 -m venv .venv
.venv/bin/pip install -e .
```

**2. Configure environment**
```bash
cp .env.example .env
```

Edit `.env` with your API keys:

| Variable | Required | Source |
|----------|----------|--------|
| `THE_ODDS_API_KEY` | Yes | [the-odds-api.com](https://the-odds-api.com) |
| `SPORTSRADAR_API_KEY` | No | [developer.sportradar.com](https://developer.sportradar.com) — enables injury reports |
| `OPENWEATHER_API_KEY` | No | [openweathermap.org](https://openweathermap.org/api) — enables weather context |
| `REDIS_URL` | No | Defaults to `redis://localhost:6379` |
| `SERVER_HOST` | No | Defaults to `0.0.0.0` |
| `SERVER_PORT` | No | Defaults to `8000` |

**3. Start Redis**
```bash
# macOS (Homebrew)
brew services start redis
```

## Running

```bash
.venv/bin/python server.py
```

Server starts at `http://localhost:8000/sse`.

## Claude Desktop Integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "swishbets": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

Restart Claude Desktop. The tools will be available in your conversations.

## Data Sources

- **Odds & Props**: [The Odds API](https://the-odds-api.com) — live lines from DraftKings, FanDuel, BetMGM, and more
- **Team & Player Stats**: [nba_api](https://github.com/swar/nba_api) — official NBA stats (free)
- **Injury Reports**: SportsRadar NBA API (optional)
- **Weather**: OpenWeatherMap (optional, indoor sport context)

## Architecture

```
config.py              # Environment settings (pydantic-settings)
server.py              # FastMCP server, SSE transport, resource registration
models/schemas.py      # Pydantic v2 data models
cache/ttl_cache.py     # Redis cache with per-data-type TTLs
services/
  odds_service.py      # The Odds API client
  stats_service.py     # nba_api + SportsRadar client
  defense_service.py   # Opponent defensive rankings (nba_api)
  analysis_service.py  # Value bet engine (Kelly criterion, z-score props, contextual signals)
tools/
  odds.py              # 5 MCP tools
  stats.py             # 3 MCP tools
  analysis.py          # 3 MCP tools
```

### Value Bet Engine

`find_value_bets` computes edge as `true_probability − implied_probability`, where true probability comes from each team's 10-game win rate normalized against the opponent (with a ±3% home-court adjustment). Bets above the 3% edge threshold are sized with the Kelly criterion.

`find_value_props` runs the same edge/Kelly math on player props using a z-score against the player's last-N game distribution, then applies six contextual adjustments before the threshold check:

| Signal | What it measures |
|--------|-----------------|
| **Trend** | Last-3 vs last-10 performance delta (heating up / cooling off / stable) |
| **Minutes consistency** | Volatility of playing time over last N games |
| **Back-to-back** | Fatigue penalty on points and assists markets |
| **Pace** | Combined possessions-per-game for the matchup |
| **Usage boost** | Edge from injured teammates' usage redistributed to active players |
| **Opponent defense** | Opponent's defensive rank for the relevant stat category (elite / average / weak) |
