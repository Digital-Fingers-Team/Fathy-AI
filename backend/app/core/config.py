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

    @property
    def cors_origins(self) -> List[str]:
        raw = (self.cors_origins_raw or "").strip()
        if raw == "*" or raw == "":
            return ["*"]
        return [part.strip() for part in raw.split(",") if part.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
