"""
Training Data Augmentation Module.

This module provides tools for generating synthetic training data
from a small set of examples using various augmentation strategies.
"""

from .generator import DataGenerator, AugmentConfig
from .strategies import (
    ParaphraseStrategy,
    BackTranslationStrategy,
    SynonymStrategy,
    TemplateStrategy,
    LLMStrategy,
)

__all__ = [
    "DataGenerator",
    "AugmentConfig",
    "ParaphraseStrategy",
    "BackTranslationStrategy",
    "SynonymStrategy",
    "TemplateStrategy",
    "LLMStrategy",
]
