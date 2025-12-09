"""
PEFT/LoRA Adapter Hook for TinyForgeAI.

Provides a stub implementation for applying LoRA-style adapters to models.
This module defines the API surface for PEFT integration; actual LoRA
implementations would replace the stub logic with real adapter application.

Example usage:
    from backend.training.peft_adapter import apply_lora

    model = {"model_type": "tinyforge_stub", "n_records": 3}
    patched = apply_lora(model)
"""

from datetime import datetime, timezone
from typing import Optional


def get_default_lora_config() -> dict:
    """
    Return default LoRA configuration parameters.

    Returns:
        Dictionary with default LoRA hyperparameters.
    """
    return {
        "r": 8,
        "alpha": 16,
        "target_modules": ["q", "v"],
    }


def apply_lora(model: dict, config: Optional[dict] = None) -> dict:
    """
    Apply a LoRA-style adapter to a model stub and return the patched model.

    This is a stub implementation that simulates LoRA adapter application.
    In a production system, this would:
    1. Load the base model
    2. Create low-rank adapter matrices for target modules
    3. Inject adapters into the model's forward pass
    4. Return the patched model ready for fine-tuning

    The function is idempotent: calling it multiple times updates the timestamp
    but does not duplicate configuration.

    Args:
        model: Dictionary representing the model stub.
               Expected to have at least a 'model_type' key.
        config: Optional LoRA configuration dictionary.
                If None, uses default configuration from get_default_lora_config().

    Returns:
        Dictionary representing the patched model with LoRA metadata added.
        Includes:
        - All original model keys
        - 'lora_applied': True
        - 'lora_config': The configuration used
        - 'lora_timestamp': ISO8601 UTC timestamp

    Raises:
        ValueError: If model is None or not a dictionary.
    """
    if model is None or not isinstance(model, dict):
        raise ValueError("Model must be a non-None dictionary")

    # Use default config if not provided
    if config is None:
        config = get_default_lora_config()

    # Create patched model (preserve original keys)
    patched_model = model.copy()

    # Apply LoRA metadata (idempotent - updates timestamp, keeps config consistent)
    patched_model["lora_applied"] = True
    patched_model["lora_config"] = config
    patched_model["lora_timestamp"] = datetime.now(timezone.utc).isoformat()

    return patched_model
