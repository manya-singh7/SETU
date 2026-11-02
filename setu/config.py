"""
Configuration settings for Project SETU.

Loads configuration from environment variables and .env file.
Adheres strictly to the principle: no secrets in code.
"""

from typing import List, Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env."""

    # Meta Model API (SAM 3.1 & Muse Spark 1.3)
    META_API_KEY: Optional[str] = None
    META_API_BASE_URL: str = "https://api.meta.ai/v1"
    MUSE_MODEL: str = "muse-spark-1.3"
    MUSE_MODEL_NAME: Optional[str] = None
    SAM_MODEL: str = "sam-3.1"
    SAM_MODEL_NAME: str = "sam-3.1"
    SAM_COST_PER_IMAGE: float = 0.0025  # $2.50 per 1,000 images

    # Meta Execution Mode & Budget Safety
    SETU_META_MODE: str = "mock"  # "mock" | "cached" | "live"
    SETU_BUDGET_CAP: float = 50.0  # Hard budget limit in USD
    USAGE_LOG_PATH: str = "usage.jsonl"
    CACHE_META_DIR: str = "cache/meta"

    # Execution Mode: allow offline/local fallback during hackathon
    USE_LOCAL_FALLBACKS: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./setu.db"

    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"

    # OSM Overpass API
    OVERPASS_API_URL: str = "https://overpass-api.de/api/interpreter"

    # Storage
    STORAGE_DIR: str = "./uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def sync_models(self) -> "Settings":
        """Keep model aliases in sync."""
        if self.MUSE_MODEL_NAME and not self.MUSE_MODEL:
            self.MUSE_MODEL = self.MUSE_MODEL_NAME
        elif self.MUSE_MODEL and not self.MUSE_MODEL_NAME:
            self.MUSE_MODEL_NAME = self.MUSE_MODEL

        if self.SAM_MODEL_NAME and not self.SAM_MODEL:
            self.SAM_MODEL = self.SAM_MODEL_NAME
        elif self.SAM_MODEL and not self.SAM_MODEL_NAME:
            self.SAM_MODEL_NAME = self.SAM_MODEL
        return self

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
