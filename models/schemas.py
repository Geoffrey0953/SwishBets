from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class Team(BaseModel):
    id: int
    name: str
    abbreviation: str
    city: str
    conference: Optional[str] = None
    division: Optional[str] = None


class Player(BaseModel):
    id: int
    full_name: str
    team_id: Optional[int] = None
    team_abbreviation: Optional[str] = None
    position: Optional[str] = None
    is_active: bool = True


class Spread(BaseModel):
    home_team: str
    home_point: float
    home_price: int
    away_team: str
    away_point: float
    away_price: int


class Total(BaseModel):
    point: float
    over_price: int
    under_price: int


class OddsLine(BaseModel):
    event_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    bookmaker: str
    moneyline_home: Optional[int] = None
    moneyline_away: Optional[int] = None
    spread: Optional[Spread] = None
    total: Optional[Total] = None


class Game(BaseModel):
    id: str
    home_team: str
    away_team: str
    commence_time: datetime
    sport: str = "basketball_nba"
    status: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None


class PlayerProp(BaseModel):
    event_id: str
    player_name: str
    market: str  # e.g. "player_points", "player_rebounds"
    line: float
    over_price: int
    under_price: int
    bookmaker: str


class InjuryPlayer(BaseModel):
    player_id: Optional[int] = None
    player_name: str
    status: str  # "Out", "Questionable", "Doubtful", "Day-To-Day"
    injury_type: Optional[str] = None
    updated_at: Optional[datetime] = None


class InjuryReport(BaseModel):
    team_id: int
    team_name: str
    players: list[InjuryPlayer] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TeamStats(BaseModel):
    team_id: int
    team_name: str
    games_played: int
    wins: int
    losses: int
    win_pct: float
    points_per_game: float
    points_allowed_per_game: float
    pace: Optional[float] = None
    offensive_rating: Optional[float] = None
    defensive_rating: Optional[float] = None
    net_rating: Optional[float] = None
    last_n_games: Optional[int] = None


class HeadToHeadResult(BaseModel):
    team1_id: int
    team1_name: str
    team2_id: int
    team2_name: str
    total_games: int
    team1_wins: int
    team2_wins: int
    team1_avg_score: float
    team2_avg_score: float
    avg_total_score: float


class ValueBet(BaseModel):
    event_id: str
    market: str
    selection: str  # team name or "Over"/"Under"
    bookmaker: str
    american_odds: int
    implied_probability: float
    true_probability: float
    edge: float  # true_prob - implied_prob
    kelly_fraction: float
    confidence: str  # "Low", "Medium", "High"
    line_movement: Optional[str] = None  # e.g. "+1.5 ↑ (agrees)" or None
    trend: Optional[str] = None  # "heating_up", "cooling_off", "stable"
    minutes_grade: Optional[str] = None  # "very_consistent", "consistent", "moderate", "volatile"
    back_to_back: Optional[bool] = None
    pace_grade: Optional[str] = None  # "fast", "average", "slow"
    usage_boost: Optional[float] = None
    opponent_def_rank: Optional[int] = None
    opponent_def_grade: Optional[str] = None  # "elite", "average", "weak"


class ArbOpportunity(BaseModel):
    event_id: str
    market: str           # "h2h", "spreads", "totals", "player_points", etc.
    home_team: str
    away_team: str
    side_a: str           # e.g. "Detroit Pistons" or "Over 211.5"
    side_a_book: str
    side_a_odds: int
    side_b: str           # e.g. "Orlando Magic" or "Under 211.5"
    side_b_book: str
    side_b_odds: int
    arb_pct: float        # guaranteed profit % per $100 total staked
    side_a_stake: float   # optimal stake on side A per $100 total
    side_b_stake: float   # optimal stake on side B per $100 total


class LineMovement(BaseModel):
    event_id: str
    market: str
    bookmaker: str
    opening_line: float
    current_line: float
    delta: float
    direction: str  # "Up", "Down", "Flat"
    snapshots: list[dict] = Field(default_factory=list)
