# SwishBets

NBA sports betting analysis server for [Claude Desktop](https://claude.ai/download) via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). Gives Claude live access to odds, player props, injury reports, team stats, and statistical value-bet recommendations.

## Tools

| Tool | Description |
|------|-------------|
| `get_tonight_games` | Tonight's NBA games with tip-off times |
| `get_odds` | Moneyline, spread, and total odds across all books |
| `get_player_props` | Player prop lines (points/assists/rebounds/threes) |
| `compare_books` | Best number for a market across all sportsbooks |
| `get_line_movement` | How the spread has shifted since open |
| `get_injury_report` | Latest injury report for a team |
| `get_team_stats` | Recent team performance (pace, ratings, record) |
| `get_head_to_head` | Historical head-to-head record between two teams |
| `find_value_bets` | Moneyline bets with >3% edge vs. team win-rate baseline |
| `find_value_props` | Player props with >3% edge vs. historical distribution |
| `get_weather_impact` | Weather context for the venue city |

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
.venv/bin/python -m swishbets.server
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
swishbets/
  server.py              # FastMCP server, SSE transport, resource registration
  config.py              # Environment settings (pydantic-settings)
  models/schemas.py      # Pydantic v2 data models
  cache/ttl_cache.py     # Redis cache with per-data-type TTLs
  services/
    odds_service.py      # The Odds API client
    stats_service.py     # nba_api + SportsRadar client
    analysis_service.py  # Value bet engine (Kelly criterion, z-score props)
  tools/
    odds.py              # 5 MCP tools
    stats.py             # 3 MCP tools
    analysis.py          # 3 MCP tools
```
