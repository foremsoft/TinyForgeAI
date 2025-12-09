"""Tests for the PEFT/LoRA adapter hook."""

import pytest

from backend.training.peft_adapter import apply_lora, get_default_lora_config


def test_apply_lora_sets_lora_applied_true():
    """Test that apply_lora sets lora_applied to True."""
    model = {"model_type": "tinyforge_stub", "n_records": 3}
    patched = apply_lora(model)
    assert patched["lora_applied"] is True


def test_apply_lora_includes_default_config():
    """Test that apply_lora includes the default LoRA config."""
    model = {"model_type": "tinyforge_stub", "n_records": 3}
    patched = apply_lora(model)

    expected_config = get_default_lora_config()
    assert patched["lora_config"] == expected_config


def test_apply_lora_includes_timestamp():
    """Test that apply_lora includes a non-empty ISO timestamp."""
    model = {"model_type": "tinyforge_stub", "n_records": 3}
    patched = apply_lora(model)

    assert "lora_timestamp" in patched
    assert isinstance(patched["lora_timestamp"], str)
    assert len(patched["lora_timestamp"]) > 0
    # Check it looks like an ISO timestamp (contains T for datetime separator)
    assert "T" in patched["lora_timestamp"]


def test_apply_lora_preserves_original_keys():
    """Test that apply_lora preserves all original model keys."""
    model = {"model_type": "tinyforge_stub", "n_records": 3, "custom_key": "value"}
    patched = apply_lora(model)

    assert patched["model_type"] == "tinyforge_stub"
    assert patched["n_records"] == 3
    assert patched["custom_key"] == "value"


def test_apply_lora_idempotent():
    """Test that calling apply_lora twice is idempotent."""
    model = {"model_type": "tinyforge_stub", "n_records": 3}

    # First application
    patched1 = apply_lora(model)
    first_config = patched1["lora_config"]

    # Second application
    patched2 = apply_lora(patched1)

    # Should still have lora_applied True (not duplicated)
    assert patched2["lora_applied"] is True

    # Config should remain the same
    assert patched2["lora_config"] == first_config

    # Should still have only one lora_applied key
    assert list(patched2.keys()).count("lora_applied") == 1


def test_apply_lora_with_custom_config():
    """Test that apply_lora accepts custom configuration."""
    model = {"model_type": "tinyforge_stub", "n_records": 3}
    custom_config = {"r": 16, "alpha": 32, "target_modules": ["q", "k", "v"]}

    patched = apply_lora(model, config=custom_config)

    assert patched["lora_config"] == custom_config


def test_apply_lora_raises_on_none_model():
    """Test that apply_lora raises ValueError for None model."""
    with pytest.raises(ValueError, match="must be a non-None dictionary"):
        apply_lora(None)


def test_apply_lora_raises_on_non_dict_model():
    """Test that apply_lora raises ValueError for non-dict model."""
    with pytest.raises(ValueError, match="must be a non-None dictionary"):
        apply_lora("not a dict")


def test_get_default_lora_config_structure():
    """Test that default config has expected keys and values."""
    config = get_default_lora_config()

    assert config["r"] == 8
    assert config["alpha"] == 16
    assert config["target_modules"] == ["q", "v"]
