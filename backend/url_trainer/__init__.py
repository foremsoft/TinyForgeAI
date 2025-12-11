"""
One-Click URL Training Module.

Train models directly from URLs - no data preparation needed.
Supports Notion, Google Docs, websites, GitHub, and more.
"""

from .extractor import URLExtractor, ExtractedData
from .trainer import URLTrainer, URLTrainConfig

__all__ = [
    "URLExtractor",
    "ExtractedData",
    "URLTrainer",
    "URLTrainConfig",
]
