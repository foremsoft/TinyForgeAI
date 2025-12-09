"""
Configuration module for TinyForgeAI.

Loads application settings from environment variables with sensible defaults.
Uses pydantic-settings for validation and type coercion.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    APP_ENV: str = "development"
    PORT: int = 8000
    MODEL_REGISTRY_PATH: str = "./model_registry"
    GOOGLE_OAUTH_DISABLED: bool = True
    CONNECTOR_MOCK: bool = True
    DB_URL: str = "sqlite:///:memory:"


settings = Settings()
