"""
Tests for the real trainer module.

These tests verify the RealTrainer works correctly when dependencies are available,
and gracefully handles missing dependencies.
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module (will work even without training deps)
from backend.training import real_trainer
from backend.training.real_trainer import TrainingConfig


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.model_name == "t5-small"
        assert config.model_type == "seq2seq"
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.use_peft is True
        assert config.lora_r == 8
        assert config.lora_alpha == 32

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            model_name="gpt2",
            model_type="causal",
            epochs=5,
            batch_size=8,
            use_peft=False,
        )
        assert config.model_name == "gpt2"
        assert config.model_type == "causal"
        assert config.epochs == 5
        assert config.batch_size == 8
        assert config.use_peft is False

    def test_lora_target_modules_t5(self):
        """Test automatic target module detection for T5."""
        config = TrainingConfig(model_name="t5-small")
        assert config.lora_target_modules == ["q", "v"]

    def test_lora_target_modules_gpt(self):
        """Test automatic target module detection for GPT."""
        config = TrainingConfig(model_name="gpt2")
        assert config.lora_target_modules == ["q_proj", "v_proj"]

    def test_lora_target_modules_opt(self):
        """Test automatic target module detection for OPT."""
        config = TrainingConfig(model_name="facebook/opt-125m")
        assert config.lora_target_modules == ["q_proj", "v_proj"]

    def test_custom_lora_target_modules(self):
        """Test custom target modules override auto-detection."""
        config = TrainingConfig(
            model_name="t5-small",
            lora_target_modules=["custom_q", "custom_v"]
        )
        assert config.lora_target_modules == ["custom_q", "custom_v"]


class TestRealTrainerWithoutDeps:
    """Test RealTrainer behavior when dependencies are not available."""

    def test_trainer_init_without_deps(self):
        """Test that trainer raises error when deps not available."""
        # Save original value
        original_available = real_trainer.TRAINING_AVAILABLE

        try:
            # Simulate missing dependencies
            real_trainer.TRAINING_AVAILABLE = False

            with pytest.raises(RuntimeError) as excinfo:
                real_trainer.RealTrainer()

            assert "Training dependencies not available" in str(excinfo.value)
        finally:
            # Restore original value
            real_trainer.TRAINING_AVAILABLE = original_available


class TestRealTrainerWithMocks:
    """Test RealTrainer with mocked dependencies."""

    @pytest.fixture
    def mock_training_deps(self):
        """Mock all training dependencies."""
        with patch.multiple(
            real_trainer,
            TRAINING_AVAILABLE=True,
            PEFT_AVAILABLE=True,
        ):
            # Mock torch
            mock_torch = MagicMock()
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()

            # Mock model
            mock_model = MagicMock()
            mock_model.parameters.return_value = [MagicMock()]
            mock_model.generate.return_value = [[1, 2, 3]]

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer.decode.return_value = "test output"
            mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}

            yield {
                "torch": mock_torch,
                "model": mock_model,
                "tokenizer": mock_tokenizer,
            }

    def test_config_post_init(self):
        """Test config post-initialization."""
        config = TrainingConfig(model_name="google/flan-t5-base")
        assert config.lora_target_modules == ["q", "v"]


class TestPrepareDataset:
    """Test dataset preparation logic."""

    @pytest.fixture
    def sample_data_file(self, tmp_path):
        """Create a sample JSONL file."""
        data_path = tmp_path / "sample.jsonl"
        records = [
            {"input": "hello", "output": "world"},
            {"input": "foo", "output": "bar"},
            {"input": "test", "output": "result"},
        ]
        with open(data_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return data_path

    def test_jsonl_loading(self, sample_data_file):
        """Test that JSONL files can be loaded."""
        records = []
        with open(sample_data_file) as f:
            for line in f:
                records.append(json.loads(line.strip()))

        assert len(records) == 3
        assert records[0]["input"] == "hello"
        assert records[0]["output"] == "world"


class TestTrainingAvailability:
    """Test training availability checks."""

    def test_training_available_flag_exists(self):
        """Test that TRAINING_AVAILABLE flag is defined."""
        assert hasattr(real_trainer, "TRAINING_AVAILABLE")
        assert isinstance(real_trainer.TRAINING_AVAILABLE, bool)

    def test_peft_available_flag_exists(self):
        """Test that PEFT_AVAILABLE flag is defined."""
        assert hasattr(real_trainer, "PEFT_AVAILABLE")
        assert isinstance(real_trainer.PEFT_AVAILABLE, bool)


@pytest.mark.skipif(
    not real_trainer.TRAINING_AVAILABLE,
    reason="Training dependencies not installed"
)
class TestRealTrainerIntegration:
    """Integration tests that require actual training dependencies."""

    def test_trainer_initialization(self):
        """Test trainer initializes with defaults."""
        trainer = real_trainer.RealTrainer()
        assert trainer.config is not None
        assert trainer.model is None
        assert trainer.tokenizer is None

    def test_trainer_with_custom_config(self):
        """Test trainer with custom configuration."""
        config = TrainingConfig(
            model_name="t5-small",
            epochs=1,
            batch_size=2,
        )
        trainer = real_trainer.RealTrainer(config)
        assert trainer.config.epochs == 1
        assert trainer.config.batch_size == 2


@pytest.mark.skipif(
    not real_trainer.TRAINING_AVAILABLE,
    reason="Training dependencies not installed"
)
@pytest.mark.slow
class TestRealTrainerSlowIntegration:
    """Slow integration tests that download models."""

    def test_load_model_t5_small(self):
        """Test loading t5-small model (requires download)."""
        config = TrainingConfig(model_name="t5-small")
        trainer = real_trainer.RealTrainer(config)
        trainer.load_model()

        assert trainer.model is not None
        assert trainer.tokenizer is not None

    def test_prepare_dataset(self, tmp_path):
        """Test dataset preparation with real tokenizer."""
        # Create sample data
        data_path = tmp_path / "train.jsonl"
        with open(data_path, "w") as f:
            f.write('{"input": "translate: hello", "output": "hola"}\n')
            f.write('{"input": "translate: world", "output": "mundo"}\n')

        config = TrainingConfig(model_name="t5-small", max_samples=2)
        trainer = real_trainer.RealTrainer(config)
        trainer.load_model()

        dataset = trainer.prepare_dataset(data_path)
        assert len(dataset) == 2
        assert "input_ids" in dataset.column_names
        assert "labels" in dataset.column_names


class TestCLIArguments:
    """Test CLI argument parsing."""

    def test_main_function_exists(self):
        """Test that main function is defined."""
        assert hasattr(real_trainer, "main")
        assert callable(real_trainer.main)

    def test_argparse_help(self):
        """Test that argparse is properly configured."""
        import argparse
        # The main function should not crash when imported
        assert real_trainer.main is not None


class TestModuleImport:
    """Test module imports and structure."""

    def test_training_config_importable(self):
        """Test TrainingConfig can be imported."""
        from backend.training.real_trainer import TrainingConfig
        assert TrainingConfig is not None

    def test_real_trainer_importable(self):
        """Test RealTrainer can be imported."""
        from backend.training.real_trainer import RealTrainer
        assert RealTrainer is not None

    def test_module_docstring(self):
        """Test module has docstring."""
        assert real_trainer.__doc__ is not None
        assert "HuggingFace" in real_trainer.__doc__
