"""
Logging setup module for TinyForgeAI.

Provides centralized logging configuration for the application.
"""

import logging
import os


def setup_logging() -> None:
    """
    Configure application logging.

    Sets up basic logging with a timestamp, level, and message format.
    Log level can be configured via LOG_LEVEL environment variable.
    """
    # Get log level from environment, default to INFO
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Configure logging with simple format
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
