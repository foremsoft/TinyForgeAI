"""
Real Training Module using HuggingFace Transformers + PEFT.

This module provides actual model fine-tuning capabilities for TinyForgeAI,
supporting both full fine-tuning and parameter-efficient methods (LoRA).

Usage:
    from backend.training.real_trainer import RealTrainer, TrainingConfig

    config = TrainingConfig(
        model_name="t5-small",
        use_peft=True,
        epochs=3,
        batch_size=4
    )
    trainer = RealTrainer(config)
    trainer.train(data_path="data.jsonl", output_dir="./model")

CLI Usage:
    python -m backend.training.real_trainer \\
        --data examples/data/demo_dataset.jsonl \\
        --model t5-small \\
        --out ./real_model \\
        --epochs 1 \\
        --use-lora
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

# Flag to check if real training dependencies are available
TRAINING_AVAILABLE = False
try:
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    logger.warning(
        "Training dependencies not installed. "
        "Install with: pip install transformers datasets torch"
    )

# PEFT availability
PEFT_AVAILABLE = False
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    logger.warning(
        "PEFT not installed. LoRA training disabled. "
        "Install with: pip install peft"
    )


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model settings
    model_name: str = "t5-small"
    model_type: str = "seq2seq"  # "seq2seq" or "causal"

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_length: int = 512
    gradient_accumulation_steps: int = 1

    # PEFT/LoRA settings
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None

    # Output settings
    output_dir: str = "./output"
    save_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500

    # Hardware settings
    fp16: bool = False
    bf16: bool = False
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Data settings
    max_samples: Optional[int] = None  # Limit samples for testing
    validation_split: float = 0.1

    def __post_init__(self):
        """Set default target modules based on model type."""
        if self.lora_target_modules is None:
            if "t5" in self.model_name.lower():
                self.lora_target_modules = ["q", "v"]
            elif "gpt" in self.model_name.lower() or "opt" in self.model_name.lower():
                self.lora_target_modules = ["q_proj", "v_proj"]
            else:
                self.lora_target_modules = ["q_proj", "v_proj"]


class RealTrainer:
    """
    Real model trainer using HuggingFace Transformers.

    Supports:
    - Seq2Seq models (T5, BART, etc.)
    - Causal LM models (GPT-2, OPT, etc.)
    - PEFT/LoRA for efficient fine-tuning
    - CPU and GPU training
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the trainer.

        Args:
            config: Training configuration. Uses defaults if not provided.
        """
        if not TRAINING_AVAILABLE:
            raise RuntimeError(
                "Training dependencies not available. "
                "Install with: pip install transformers datasets torch"
            )

        self.config = config or TrainingConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model based on type
        if self.config.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )

        logger.info(f"Model loaded: {self.model.__class__.__name__}")

    def apply_peft(self) -> None:
        """Apply PEFT/LoRA to the model."""
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "PEFT not available. Install with: pip install peft"
            )

        if not self.config.use_peft:
            logger.info("PEFT disabled, using full fine-tuning")
            return

        logger.info("Applying LoRA configuration...")

        # Determine task type
        if self.config.model_type == "seq2seq":
            task_type = TaskType.SEQ_2_SEQ_LM
        else:
            task_type = TaskType.CAUSAL_LM

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type=task_type,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self, data_path: Union[str, Path]) -> "Dataset":
        """
        Load and prepare dataset from JSONL file.

        Args:
            data_path: Path to JSONL file with 'input' and 'output' fields.

        Returns:
            HuggingFace Dataset ready for training.
        """
        logger.info(f"Loading dataset from: {data_path}")

        records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records.append(record)

        if self.config.max_samples:
            records = records[:self.config.max_samples]
            logger.info(f"Limited to {len(records)} samples")

        logger.info(f"Loaded {len(records)} records")

        # Convert to HF Dataset
        dataset = Dataset.from_list(records)

        # Tokenize
        def tokenize_function(examples):
            # For seq2seq: input -> output
            if self.config.model_type == "seq2seq":
                model_inputs = self.tokenizer(
                    examples["input"],
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length",
                )
                labels = self.tokenizer(
                    examples["output"],
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length",
                )
                model_inputs["labels"] = labels["input_ids"]
            else:
                # For causal LM: concatenate input + output
                texts = [
                    f"{inp} {out}"
                    for inp, out in zip(examples["input"], examples["output"])
                ]
                model_inputs = self.tokenizer(
                    texts,
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length",
                )
                model_inputs["labels"] = model_inputs["input_ids"].copy()

            return model_inputs

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return tokenized_dataset

    def train(
        self,
        data_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Train the model on the provided dataset.

        Args:
            data_path: Path to JSONL training data.
            output_dir: Directory to save the trained model.

        Returns:
            Dictionary with training results and metadata.
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Apply PEFT if configured
        if self.config.use_peft and PEFT_AVAILABLE:
            self.apply_peft()

        # Prepare dataset
        dataset = self.prepare_dataset(data_path)

        # Split into train/validation
        if self.config.validation_split > 0:
            split = dataset.train_test_split(
                test_size=self.config.validation_split,
                seed=42
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            load_best_model_at_end=eval_dataset is not None,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            report_to="none",  # Disable wandb/tensorboard by default
            remove_unused_columns=False,
        )

        # Setup data collator
        if self.config.model_type == "seq2seq":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True,
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train()

        # Save model
        logger.info(f"Saving model to: {output_dir}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        # Save training metadata
        metadata = {
            "model_type": "tinyforge_real",
            "model_name": self.config.model_name,
            "base_model": self.config.model_name,
            "version": "0.1.0",
            "training_config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "use_peft": self.config.use_peft,
                "lora_r": self.config.lora_r if self.config.use_peft else None,
            },
            "training_results": {
                "train_loss": train_result.training_loss,
                "train_steps": train_result.global_step,
            },
            "data_path": str(data_path),
            "n_records": len(train_dataset),
        }

        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Training complete! Model saved to: {output_dir}")

        return metadata

    def predict(self, text: str, max_new_tokens: int = 100) -> str:
        """
        Generate prediction for input text.

        Args:
            text: Input text.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )

        # Decode
        if self.config.model_type == "seq2seq":
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # For causal LM, skip the input tokens
            input_length = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )

        return decoded

    @classmethod
    def load_trained_model(
        cls,
        model_dir: Union[str, Path],
        config: Optional[TrainingConfig] = None
    ) -> "RealTrainer":
        """
        Load a previously trained model.

        Args:
            model_dir: Directory containing the trained model.
            config: Optional config override.

        Returns:
            RealTrainer instance with loaded model.
        """
        model_dir = Path(model_dir)

        # Load metadata
        metadata_path = model_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            base_model = metadata.get("base_model", "t5-small")
            use_peft = metadata.get("training_config", {}).get("use_peft", False)
        else:
            base_model = "t5-small"
            use_peft = False

        # Create config
        if config is None:
            config = TrainingConfig(model_name=base_model, use_peft=use_peft)

        trainer = cls(config)

        # Load tokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load model
        if use_peft and PEFT_AVAILABLE:
            # Load base model first
            if config.model_type == "seq2seq":
                base_model_instance = AutoModelForSeq2SeqLM.from_pretrained(base_model)
            else:
                base_model_instance = AutoModelForCausalLM.from_pretrained(base_model)
            # Load PEFT adapter
            trainer.model = PeftModel.from_pretrained(base_model_instance, model_dir)
        else:
            if config.model_type == "seq2seq":
                trainer.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            else:
                trainer.model = AutoModelForCausalLM.from_pretrained(model_dir)

        return trainer


def main():
    """CLI entry point for real training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model using HuggingFace Transformers + PEFT"
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to JSONL training data"
    )
    parser.add_argument(
        "--model", "-m",
        default="t5-small",
        help="Model name or path (default: t5-small)"
    )
    parser.add_argument(
        "--out", "-o",
        default="./real_model",
        help="Output directory (default: ./real_model)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA/PEFT training"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of training samples (for testing)"
    )
    parser.add_argument(
        "--model-type",
        choices=["seq2seq", "causal"],
        default="seq2seq",
        help="Model type (default: seq2seq)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create config
    config = TrainingConfig(
        model_name=args.model,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_peft=args.use_lora,
        lora_r=args.lora_r,
        max_samples=args.max_samples,
        output_dir=args.out,
        device="cpu" if args.cpu else "auto",
    )

    # Train
    trainer = RealTrainer(config)
    result = trainer.train(data_path=args.data, output_dir=args.out)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Model saved to: {args.out}")
    print(f"Training loss: {result['training_results']['train_loss']:.4f}")
    print(f"Total steps: {result['training_results']['train_steps']}")


if __name__ == "__main__":
    main()
