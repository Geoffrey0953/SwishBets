# SwishBets

NBA sports betting analysis server for [Claude Desktop](https://claude.ai/download) via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). Gives Claude live access to odds, player props, injury reports, team stats, and statistical value-bet recommendations.

## Tools

### Odds (6)

| Tool | Description |
|------|-------------|
| `get_tonight_games` | Tonight's NBA games with tip-off times and game IDs |
| `get_odds` | Moneyline, spread, and total odds across all sportsbooks |
| `get_player_props` | Player prop lines (points / assists / rebounds / threes) with optional player filter |
| `compare_books` | Best available number for a given market across all books |
| `get_pinnacle_line` | Pinnacle's no-vig line for a game — the sharp reference used by the EV engine |
| `get_line_movement` | How the spread has shifted since open (48-hour delta via historical odds) |

### Stats (3)

| Tool | Description |
|------|-------------|
| `get_injury_report` | Latest injury report for a team (via SportsRadar) |
| `get_team_stats` | Recent team performance — pace, offensive/defensive ratings, record |
| `get_head_to_head` | Historical head-to-head record and scoring averages between two teams |

### Analysis (4)

| Tool | Description |
|------|-------------|
| `find_value_bets` | Moneyline value using Pinnacle no-vig as true probability (falls back to 10-game win rate). Returns bets above the 1% edge threshold with quarter-Kelly sizing and High/Medium/Low confidence |
| `find_positive_ev` | +EV scan across moneyline, spreads, totals, and player props — Pinnacle no-vig reference, concurrent game processing, contextual signals (usage boost, back-to-back, opponent defensive grade) |
| `find_arb` | Guaranteed-profit arbitrage across all books and markets (h2h, spreads, totals, props). Groups props by exact player/market/line, deduplicates same-operator books, returns optimal stake split per $100 |
| `get_weather_impact` | Weather context for the venue city (NBA games are indoors — useful for travel/player conditions) |

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
  analysis_service.py  # Value bet engine (EV math, Kelly sizing, contextual signals)
tools/
  odds.py              # 6 MCP tools
  stats.py             # 3 MCP tools
  analysis.py          # 4 MCP tools
```

### Value Bet Engine

All EV calculations use the same core formula: `edge = true_probability − implied_probability`. Bets at or above the **1% edge threshold** are included.

**True probability source:** Pinnacle's no-vig line is the primary reference — vig is stripped by normalizing both sides so they sum to 1.0. If Pinnacle isn't available for a game, the engine falls back to each team's 10-game win rate normalized against the opponent.

**Bet sizing:** Quarter-Kelly (`full_kelly × 0.25`, capped at 6.25% of bankroll). Quarter-Kelly is used instead of full Kelly to account for model estimation uncertainty in sports betting.

**Confidence grading:**

| Grade | Criteria |
|-------|----------|
| High | Edge ≥ 8% and ≥ 5 books offering the market |
| Medium | Edge ≥ 4% and ≥ 5 books offering the market |
| Low | Edge < 4%, or fewer than 5 books (thin market) |

### Book Quality Tiers

The engine filters and flags books by tier before any EV or arb calculation:

| Tier | Books | Behavior |
|------|-------|----------|
| Sharp | Pinnacle, BetOnlineAG, LowVig | Used as reference; included normally |
| Standard | DraftKings, FanDuel, BetMGM, etc. | Included normally |
| Exchange | Novig, BetOpenly, ProphetX | Included, flagged `[EX]` |
| Prediction Market | Kalshi, Polymarket | Included, flagged `[PM]` |
| Skip | Fliff, ESPN Bet | Excluded entirely |

Same-operator books (e.g. `hardrockbet_fl` / `hardrockbet_az`, `lowvig` / `betonlineag`) are deduplicated so they can't be used on opposite sides of an arb.

### Player Prop Signals

`find_positive_ev` applies these contextual adjustments after the base Pinnacle edge is calculated:

| Signal | What it measures |
|--------|-----------------|
| **Usage boost** | Edge delta (+0.012) when an injured teammate's usage (>5% absorbed) shifts to the player |
| **Back-to-back** | Confidence downgraded one level (High→Medium, Medium/Low→Low) |
| **Opponent defense** | Over on scoring props downgraded vs. elite defense; Under downgraded vs. weak defense |
| **Thin market** | Props with fewer than 3 books offering them are skipped — reference line unreliable |
| **Prop outlier filter** | Props above +1000 skipped entirely; +500–+1000 flagged with ⚠ |

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

153 tests across `tests/test.py` (unit + integration) and `tests/integration_test.py` (live API), covering all math helpers, value bet logic, arb scanning, line movement, usage boost, contextual prop signals, and back-to-back/defensive grade adjustments.
