"""
Configuration settings for Project SETU.

Loads configuration from environment variables and .env file.
Adheres strictly to the principle: no secrets in code.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env."""

    # Meta Model API (SAM 3.1 & Muse Spark 1.3)
    META_API_KEY: Optional[str] = None
    META_API_BASE_URL: str = "https://api.meta.ai/v1"
    MUSE_MODEL_NAME: str = "muse-spark-1.3"
    SAM_MODEL_NAME: str = "facebook/sam3.1"

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

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
