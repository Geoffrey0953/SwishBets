from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from cache.ttl_cache import RedisCache, HISTORICAL, ODDS, PROPS
from config import settings
from models.schemas import (
    Game,
    LineMovement,
    OddsLine,
    PlayerProp,
    Spread,
    Total,
)

logger = logging.getLogger(__name__)


class OddsService:
    def __init__(self, cache: RedisCache) -> None:
        self.cache = cache
        self.base_url = settings.odds_api_base_url
        self.api_key = settings.the_odds_api_key

    async def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        merged = {"apiKey": self.api_key, **(params or {})}
        print(f"[API CALL]   The Odds API → {path}")
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=merged)
            if not resp.is_success:
                try:
                    body = resp.json()
                    msg = body.get("message", resp.text)
                    code = body.get("error_code", "")
                    raise ValueError(f"Odds API error ({resp.status_code}){f' [{code}]' if code else ''}: {msg}")
                except (ValueError, KeyError):
                    raise
                except Exception:
                    resp.raise_for_status()
            return resp.json()

    async def get_games(
        self,
        sport: str = "basketball_nba",
        game_date: Optional[date] = None,
    ) -> list[Game]:
        cache_key = f"games:{sport}:{game_date or 'today'}"
        cached = await self.cache.get(cache_key)
        if cached:
            return [Game(**g) for g in cached]

        data = await self._get(f"/sports/{sport}/events")
        games: list[Game] = []
        for event in data:
            commence = datetime.fromisoformat(
                event["commence_time"].replace("Z", "+00:00")
            )
            if game_date and commence.astimezone().date() != game_date:
                continue
            games.append(
                Game(
                    id=event["id"],
                    home_team=event["home_team"],
                    away_team=event["away_team"],
                    commence_time=commence,
                )
            )

        await self.cache.set(cache_key, [g.model_dump() for g in games], ODDS)
        return games

    async def get_odds(
        self,
        event_id: str,
        markets: Optional[list[str]] = None,
    ) -> list[OddsLine]:
        if markets is None:
            markets = ["h2h", "spreads", "totals"]

        cache_key = f"odds:{event_id}:{','.join(markets)}"
        cached = await self.cache.get(cache_key)
        if cached:
            return [OddsLine(**o) for o in cached]

        data = await self._get(
            f"/sports/basketball_nba/events/{event_id}/odds",
            params={"markets": ",".join(markets), "oddsFormat": "american", "regions": "us"},
        )

        lines: list[OddsLine] = []
        for bookmaker in data.get("bookmakers", []):
            line = OddsLine(
                event_id=event_id,
                home_team=data["home_team"],
                away_team=data["away_team"],
                commence_time=datetime.fromisoformat(
                    data["commence_time"].replace("Z", "+00:00")
                ),
                bookmaker=bookmaker["key"],
            )
            for market in bookmaker.get("markets", []):
                key = market["key"]
                outcomes = {o["name"]: o for o in market["outcomes"]}
                if key == "h2h":
                    home_out = outcomes.get(data["home_team"])
                    away_out = outcomes.get(data["away_team"])
                    if home_out:
                        line.moneyline_home = int(home_out["price"])
                    if away_out:
                        line.moneyline_away = int(away_out["price"])
                elif key == "spreads":
                    home_out = outcomes.get(data["home_team"])
                    if home_out:
                        line.spread = Spread(
                            team=data["home_team"],
                            point=home_out["point"],
                            price=int(home_out["price"]),
                        )
                elif key == "totals":
                    over_out = outcomes.get("Over")
                    under_out = outcomes.get("Under")
                    if over_out and under_out:
                        line.total = Total(
                            point=over_out["point"],
                            over_price=int(over_out["price"]),
                            under_price=int(under_out["price"]),
                        )
            lines.append(line)

        await self.cache.set(cache_key, [l.model_dump() for l in lines], ODDS)
        return lines

    async def get_player_props(
        self,
        event_id: str,
        markets: Optional[list[str]] = None,
    ) -> list[PlayerProp]:
        if markets is None:
            markets = [
                "player_points",
                "player_rebounds",
                "player_assists",
                "player_threes",
            ]

        cache_key = f"props:{event_id}:{','.join(markets)}"
        cached = await self.cache.get(cache_key)
        if cached:
            return [PlayerProp(**p) for p in cached]

        data = await self._get(
            f"/sports/basketball_nba/events/{event_id}/odds",
            params={"markets": ",".join(markets), "oddsFormat": "american", "regions": "us"},
        )

        props: list[PlayerProp] = []
        for bookmaker in data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                outcomes = market.get("outcomes", [])
                # Group by player description (name)
                players: dict[str, dict] = {}
                for outcome in outcomes:
                    name = outcome.get("description", outcome.get("name", ""))
                    side = outcome["name"]  # "Over" or "Under"
                    players.setdefault(name, {})
                    players[name]["line"] = outcome.get("point", 0)
                    if side == "Over":
                        players[name]["over_price"] = int(outcome["price"])
                    else:
                        players[name]["under_price"] = int(outcome["price"])

                for player_name, pdata in players.items():
                    if "over_price" in pdata and "under_price" in pdata:
                        props.append(
                            PlayerProp(
                                event_id=event_id,
                                player_name=player_name,
                                market=market_key,
                                line=pdata["line"],
                                over_price=pdata["over_price"],
                                under_price=pdata["under_price"],
                                bookmaker=bookmaker["key"],
                            )
                        )

        await self.cache.set(cache_key, [p.model_dump() for p in props], PROPS)
        return props

    async def get_historical_event_odds(
        self,
        event_id: str,
        date: str,
        markets: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Fetch historical odds snapshot for an event at the given ISO timestamp."""
        markets_key = ",".join(markets) if markets else "all"
        cache_key = f"hist_odds:{event_id}:{date}:{markets_key}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        params: dict[str, Any] = {
            "regions": "us",
            "oddsFormat": "american",
            "date": date,
        }
        if markets:
            params["markets"] = ",".join(markets)

        try:
            data = await self._get(
                f"/historical/sports/basketball_nba/events/{event_id}/odds",
                params=params,
            )
        except Exception as exc:
            logger.warning("Historical odds fetch failed for %s at %s: %s", event_id, date, exc)
            return {}

        await self.cache.set(cache_key, data, HISTORICAL)
        return data

    async def compare_books(self, event_id: str, market: str) -> dict[str, Any]:
        lines = await self.get_odds(event_id, markets=[market])
        result: dict[str, Any] = {}
        for line in lines:
            if market == "h2h":
                result[line.bookmaker] = {
                    "home": line.moneyline_home,
                    "away": line.moneyline_away,
                }
            elif market == "spreads" and line.spread:
                result[line.bookmaker] = {
                    "point": line.spread.point,
                    "price": line.spread.price,
                }
            elif market == "totals" and line.total:
                result[line.bookmaker] = {
                    "point": line.total.point,
                    "over": line.total.over_price,
                    "under": line.total.under_price,
                }
        return result

    async def get_line_movement(self, event_id: str) -> LineMovement:
        cache_key = f"line_movement:{event_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return LineMovement(**cached)

        lines = await self.get_odds(event_id, markets=["spreads"])
        if not lines or not lines[0].spread:
            return LineMovement(
                event_id=event_id,
                market="spreads",
                bookmaker="consensus",
                opening_line=0,
                current_line=0,
                delta=0,
                direction="Flat",
            )

        ref = lines[0]
        current_point = ref.spread.point

        # Compute opening timestamp: 48 hours before tip-off
        opening_ts = (ref.commence_time - timedelta(hours=48)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        opening_point = current_point  # fallback
        hist = await self.get_historical_event_odds(event_id, opening_ts, markets=["spreads"])
        hist_data = hist.get("data") or {}
        for bookmaker in hist_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "spreads":
                    for outcome in market.get("outcomes", []):
                        if "point" in outcome:
                            opening_point = float(outcome["point"])
                            break
                    break
            break

        delta = current_point - opening_point
        direction = "Up" if delta > 0 else ("Down" if delta < 0 else "Flat")

        movement = LineMovement(
            event_id=event_id,
            market="spreads",
            bookmaker=ref.bookmaker,
            opening_line=opening_point,
            current_line=current_point,
            delta=delta,
            direction=direction,
            snapshots=[
                {"timestamp": opening_ts, "point": opening_point},
                {"timestamp": "current", "point": current_point},
            ],
        )
        await self.cache.set(cache_key, movement.model_dump(), ODDS)
        return movement
