from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = Field(default="Fathy", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    secret_key: str = Field(default="changeme", alias="SECRET_KEY")
    database_url: str = Field(default="sqlite:///./fathy.db", alias="DATABASE_URL")
    cors_origins_raw: str = Field(default="*", alias="CORS_ORIGINS")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    serving_auth_enabled: bool = Field(default=False, alias="SERVING_AUTH_ENABLED")
    serving_api_keys_raw: str = Field(default="", alias="SERVING_API_KEYS")
    serving_jwt_secret: str | None = Field(default=None, alias="SERVING_JWT_SECRET")
    serving_rate_limit_rpm: int = Field(default=120, alias="SERVING_RATE_LIMIT_RPM")

    @property
    def cors_origins(self) -> List[str]:
        raw = (self.cors_origins_raw or "").strip()
        if raw == "*" or raw == "":
            return ["*"]
        return [part.strip() for part in raw.split(",") if part.strip()]

    @property
    def serving_api_keys(self) -> set[str]:
        raw = (self.serving_api_keys_raw or "").strip()
        if not raw:
            return set()
        return {part.strip() for part in raw.split(",") if part.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
