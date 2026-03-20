from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API Keys
    the_odds_api_key: str = ""
    sportsradar_api_key: str = ""
    openweather_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # NBA Season
    nba_season: str = "2025-26"
    nba_season_year: int = 2025

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # Odds API base URL
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # SportsRadar base URL
    sportsradar_base_url: str = "https://api.sportradar.com/nba/trial/v5/en"

    # OpenWeatherMap base URL
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5"


settings = Settings()
