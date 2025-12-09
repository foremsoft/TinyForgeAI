"""
Database configuration settings for TinyForgeAI connectors.

Provides configuration for database connections via environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    DB_URL: str = "sqlite:///:memory:"


db_settings = DBSettings()
