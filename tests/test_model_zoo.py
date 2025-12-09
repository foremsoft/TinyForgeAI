"""
Tests for the Model Zoo.
"""

import json
import pytest
from pathlib import Path


class TestModelRegistry:
    """Test model registry functionality."""

    def test_import_model_zoo(self):
        """Test that model_zoo can be imported."""
        from model_zoo import list_models, get_model_info, load_model_config
        assert callable(list_models)
        assert callable(get_model_info)
        assert callable(load_model_config)

    def test_list_models_returns_list(self):
        """Test list_models returns a list."""
        from model_zoo import list_models
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_list_models_structure(self):
        """Test list_models returns correct structure."""
        from model_zoo import list_models
        models = list_models()
        for model in models:
            assert "name" in model
            assert "display_name" in model
            assert "task_type" in model
            assert "base_model" in model
            assert "description" in model

    def test_list_models_by_task(self):
        """Test filtering models by task type."""
        from model_zoo import list_models
        from model_zoo.registry import TaskType

        qa_models = list_models(TaskType.QUESTION_ANSWERING)
        assert len(qa_models) > 0
        for m in qa_models:
            assert m["task_type"] == "question_answering"

    def test_get_model_info(self):
        """Test getting detailed model info."""
        from model_zoo import get_model_info
        info = get_model_info("qa_flan_t5_small")

        assert info["name"] == "qa_flan_t5_small"
        assert info["base_model"] == "google/flan-t5-small"
        assert info["task_type"] == "question_answering"
        assert "training_defaults" in info
        assert "lora" in info
        assert "resources" in info
        assert "data_format" in info

    def test_get_model_info_invalid(self):
        """Test getting info for invalid model raises error."""
        from model_zoo import get_model_info
        with pytest.raises(ValueError):
            get_model_info("nonexistent_model")

    def test_load_model_config(self):
        """Test loading model configuration."""
        from model_zoo import load_model_config
        config = load_model_config("qa_flan_t5_small")

        assert config.name == "qa_flan_t5_small"
        assert config.base_model == "google/flan-t5-small"
        assert config.default_epochs > 0
        assert config.default_batch_size > 0
        assert config.default_learning_rate > 0

    def test_load_model_config_invalid(self):
        """Test loading invalid model raises error."""
        from model_zoo import load_model_config
        with pytest.raises(ValueError):
            load_model_config("nonexistent_model")


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_to_training_config(self):
        """Test conversion to training config."""
        from model_zoo import load_model_config
        config = load_model_config("qa_flan_t5_small")
        training = config.to_training_config()

        assert "model_name" in training
        assert "num_epochs" in training
        assert "batch_size" in training
        assert "learning_rate" in training
        assert training["model_name"] == config.base_model
        assert training["num_epochs"] == config.default_epochs

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from model_zoo import load_model_config
        config = load_model_config("qa_flan_t5_small")
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == config.name
        assert d["base_model"] == config.base_model
        assert "training_defaults" in d
        assert "lora" in d
        assert "resources" in d


class TestTaskTypes:
    """Test different task type models."""

    def test_qa_models_exist(self):
        """Test Q&A models are registered."""
        from model_zoo import list_models
        from model_zoo.registry import TaskType
        models = list_models(TaskType.QUESTION_ANSWERING)
        assert len(models) >= 2  # small and base

    def test_summarization_models_exist(self):
        """Test summarization models are registered."""
        from model_zoo import list_models
        from model_zoo.registry import TaskType
        models = list_models(TaskType.SUMMARIZATION)
        assert len(models) >= 1

    def test_classification_models_exist(self):
        """Test classification models are registered."""
        from model_zoo import list_models
        from model_zoo.registry import TaskType
        models = list_models(TaskType.CLASSIFICATION)
        assert len(models) >= 1

    def test_code_gen_models_exist(self):
        """Test code generation models are registered."""
        from model_zoo import list_models
        from model_zoo.registry import TaskType
        models = list_models(TaskType.CODE_GENERATION)
        assert len(models) >= 1

    def test_conversation_models_exist(self):
        """Test conversation models are registered."""
        from model_zoo import list_models
        from model_zoo.registry import TaskType
        models = list_models(TaskType.CONVERSATION)
        assert len(models) >= 1


class TestModelMetadata:
    """Test model metadata fields."""

    def test_all_models_have_required_fields(self):
        """Test all models have required metadata."""
        from model_zoo.registry import MODEL_REGISTRY

        for name, config in MODEL_REGISTRY.items():
            assert config.name, f"{name} missing name"
            assert config.display_name, f"{name} missing display_name"
            assert config.description, f"{name} missing description"
            assert config.base_model, f"{name} missing base_model"
            assert config.model_type in ["seq2seq", "causal", "encoder"], f"{name} invalid model_type"
            assert config.default_epochs > 0, f"{name} invalid epochs"
            assert config.default_batch_size > 0, f"{name} invalid batch_size"
            assert config.default_learning_rate > 0, f"{name} invalid learning_rate"

    def test_all_models_have_data_format_example(self):
        """Test all models have data format examples."""
        from model_zoo.registry import MODEL_REGISTRY

        for name, config in MODEL_REGISTRY.items():
            assert config.data_format_example, f"{name} missing data_format_example"
            assert len(config.data_format_example) >= 2, f"{name} needs at least 2 fields in example"

    def test_lora_models_have_target_modules(self):
        """Test models recommending LoRA have target modules."""
        from model_zoo.registry import MODEL_REGISTRY

        for name, config in MODEL_REGISTRY.items():
            if config.lora_recommended:
                # Target modules may be empty if using defaults
                assert config.lora_rank > 0, f"{name} needs lora_rank when recommending LoRA"


class TestSampleDatasets:
    """Test sample datasets."""

    def test_qa_dataset_exists(self):
        """Test Q&A sample dataset exists."""
        path = Path(__file__).parent.parent / "model_zoo" / "datasets" / "qa_samples.jsonl"
        assert path.exists()

    def test_summarization_dataset_exists(self):
        """Test summarization sample dataset exists."""
        path = Path(__file__).parent.parent / "model_zoo" / "datasets" / "summarization_samples.jsonl"
        assert path.exists()

    def test_classification_dataset_exists(self):
        """Test classification sample dataset exists."""
        path = Path(__file__).parent.parent / "model_zoo" / "datasets" / "classification_samples.jsonl"
        assert path.exists()

    def test_code_dataset_exists(self):
        """Test code sample dataset exists."""
        path = Path(__file__).parent.parent / "model_zoo" / "datasets" / "code_samples.jsonl"
        assert path.exists()

    def test_chat_dataset_exists(self):
        """Test chat sample dataset exists."""
        path = Path(__file__).parent.parent / "model_zoo" / "datasets" / "chat_samples.jsonl"
        assert path.exists()

    def test_datasets_are_valid_jsonl(self):
        """Test all datasets are valid JSONL."""
        datasets_dir = Path(__file__).parent.parent / "model_zoo" / "datasets"
        for f in datasets_dir.glob("*.jsonl"):
            with open(f) as fh:
                for i, line in enumerate(fh, 1):
                    if line.strip():
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            pytest.fail(f"Invalid JSON on line {i} of {f.name}")

    def test_datasets_have_minimum_samples(self):
        """Test datasets have at least 10 samples."""
        datasets_dir = Path(__file__).parent.parent / "model_zoo" / "datasets"
        for f in datasets_dir.glob("*.jsonl"):
            with open(f) as fh:
                count = sum(1 for line in fh if line.strip())
            assert count >= 10, f"{f.name} has only {count} samples (need >= 10)"


class TestCLI:
    """Test CLI functionality."""

    def test_cli_module_imports(self):
        """Test CLI module can be imported."""
        from model_zoo.cli import cmd_list, cmd_info, cmd_train, main
        assert callable(cmd_list)
        assert callable(cmd_info)
        assert callable(cmd_train)
        assert callable(main)

    def test_cli_list_help(self):
        """Test CLI list command works."""
        import argparse
        from model_zoo.cli import cmd_list

        args = argparse.Namespace(task=None, json=False, verbose=False)
        result = cmd_list(args)
        assert result == 0

    def test_cli_list_json(self):
        """Test CLI list with JSON output."""
        import argparse
        from model_zoo.cli import cmd_list

        args = argparse.Namespace(task=None, json=True, verbose=False)
        result = cmd_list(args)
        assert result == 0

    def test_cli_info(self):
        """Test CLI info command."""
        import argparse
        from model_zoo.cli import cmd_info

        args = argparse.Namespace(model="qa_flan_t5_small", json=False)
        result = cmd_info(args)
        assert result == 0

    def test_cli_info_invalid_model(self):
        """Test CLI info with invalid model."""
        import argparse
        from model_zoo.cli import cmd_info

        args = argparse.Namespace(model="nonexistent", json=False)
        result = cmd_info(args)
        assert result == 1

    def test_cli_train_dry_run(self):
        """Test CLI train dry run."""
        import argparse
        from model_zoo.cli import cmd_train

        args = argparse.Namespace(
            model="qa_flan_t5_small",
            data=None,
            output=None,
            epochs=None,
            batch_size=None,
            learning_rate=None,
            lora=None,
            dry_run=True
        )
        result = cmd_train(args)
        assert result == 0


class TestIntegration:
    """Integration tests."""

    def test_full_workflow_dry_run(self):
        """Test full workflow from listing to training config."""
        from model_zoo import list_models, get_model_info, load_model_config

        # List models
        models = list_models()
        assert len(models) > 0

        # Get first model name
        model_name = models[0]["name"]

        # Get info
        info = get_model_info(model_name)
        assert info["name"] == model_name

        # Load config
        config = load_model_config(model_name)
        assert config.name == model_name

        # Get training config
        training = config.to_training_config()
        assert "model_name" in training
        assert "num_epochs" in training

    def test_model_count(self):
        """Test we have a reasonable number of models."""
        from model_zoo import list_models
        models = list_models()
        # We should have at least 10 models
        assert len(models) >= 10, f"Only {len(models)} models registered"
