"""
TinyForgeAI Training Worker

Background worker service that processes training jobs from the dashboard.
Picks up pending jobs, runs training with the RealTrainer, and reports progress.

Usage:
    # Start the worker
    python -m services.training_worker.worker

    # Or as a module
    from services.training_worker.worker import TrainingWorker
    worker = TrainingWorker(api_url="http://localhost:8001")
    worker.start()
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from threading import Thread, Event

import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Training worker configuration."""
    api_url: str = "http://localhost:8001"
    poll_interval: int = 5  # seconds
    output_base_dir: str = "./output"
    max_concurrent_jobs: int = 1


class ProgressCallback:
    """
    Callback handler for training progress updates.

    Sends progress to the dashboard API via HTTP.
    """

    def __init__(self, job_id: str, api_url: str):
        self.job_id = job_id
        self.api_url = api_url
        self.client = httpx.Client(timeout=30.0)
        self._last_progress = -1

    def __call__(self, progress: float, message: str = ""):
        """Report progress update."""
        # Only report if progress changed significantly (avoid spam)
        if abs(progress - self._last_progress) < 1.0 and progress < 100:
            return

        self._last_progress = progress

        try:
            response = self.client.post(
                f"{self.api_url}/api/jobs/{self.job_id}/progress",
                params={"progress": progress}
            )
            if response.status_code == 200:
                logger.debug(f"Progress reported: {progress:.1f}%")
            else:
                logger.warning(f"Failed to report progress: {response.status_code}")
        except Exception as e:
            logger.error(f"Error reporting progress: {e}")

    def report_failure(self, error_message: str):
        """Report job failure."""
        try:
            response = self.client.post(
                f"{self.api_url}/api/jobs/{self.job_id}/fail",
                params={"error_message": error_message}
            )
            if response.status_code == 200:
                logger.info(f"Failure reported for job {self.job_id}")
            else:
                logger.warning(f"Failed to report failure: {response.status_code}")
        except Exception as e:
            logger.error(f"Error reporting failure: {e}")

    def close(self):
        """Close the HTTP client."""
        self.client.close()


class TrainingExecutor:
    """
    Executes training jobs using RealTrainer with progress callbacks.
    """

    def __init__(self, output_base_dir: str = "./output"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        job_id: str,
        job_config: Dict[str, Any],
        progress_callback: Callable[[float, str], None]
    ) -> Dict[str, Any]:
        """
        Execute a training job.

        Args:
            job_id: Unique job identifier.
            job_config: Job configuration from the API.
            progress_callback: Callback function for progress updates.

        Returns:
            Training result metadata.
        """
        from backend.training.real_trainer import RealTrainer, TrainingConfig, TRAINING_AVAILABLE

        if not TRAINING_AVAILABLE:
            raise RuntimeError("Training dependencies not installed")

        # Extract config
        dataset_path = job_config.get("dataset_path", "")
        model_name = job_config.get("model_name", "t5-small")
        config = job_config.get("config", {})

        # Create output directory for this job
        output_dir = self.output_base_dir / f"job_{job_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build training config
        training_config = TrainingConfig(
            model_name=model_name,
            model_type=config.get("model_type", "seq2seq"),
            epochs=config.get("epochs", 3),
            batch_size=config.get("batch_size", 4),
            learning_rate=config.get("learning_rate", 2e-4),
            use_peft=config.get("use_lora", True),
            lora_r=config.get("lora_r", 8),
            output_dir=str(output_dir),
            logging_steps=10,  # More frequent logging for progress
        )

        logger.info(f"Starting training job {job_id}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Dataset: {dataset_path}")
        logger.info(f"  Epochs: {training_config.epochs}")

        # Report starting
        progress_callback(5.0, "Loading model...")

        # Create trainer with progress hooks
        trainer = RealTrainer(training_config)

        # Load model
        trainer.load_model()
        progress_callback(15.0, "Model loaded")

        # Apply PEFT if configured
        if training_config.use_peft:
            from backend.training.real_trainer import PEFT_AVAILABLE
            if PEFT_AVAILABLE:
                trainer.apply_peft()
                progress_callback(20.0, "LoRA applied")

        # Prepare dataset
        progress_callback(25.0, "Preparing dataset...")
        dataset = trainer.prepare_dataset(dataset_path)
        progress_callback(30.0, f"Dataset prepared: {len(dataset)} samples")

        # Split dataset
        if training_config.validation_split > 0:
            from datasets import Dataset
            split = dataset.train_test_split(
                test_size=training_config.validation_split,
                seed=42
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None

        # Custom progress callback for HuggingFace Trainer
        class ProgressTrainerCallback:
            """Custom callback to report training progress."""

            def __init__(self, total_steps: int, progress_fn: Callable):
                self.total_steps = total_steps
                self.progress_fn = progress_fn
                self.training_base = 30.0  # Progress starts at 30%
                self.training_range = 65.0  # Training takes 30-95%

            def on_step_end(self, step: int, logs: Dict = None):
                """Called after each training step."""
                if self.total_steps > 0:
                    step_progress = step / self.total_steps
                    overall_progress = self.training_base + (step_progress * self.training_range)
                    self.progress_fn(overall_progress, f"Step {step}/{self.total_steps}")

        # Calculate total steps
        from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config.epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.batch_size,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            load_best_model_at_end=eval_dataset is not None,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            report_to="none",
            remove_unused_columns=False,
        )

        # Calculate total training steps
        total_steps = (
            len(train_dataset) // training_config.batch_size
        ) * training_config.epochs

        # Create progress tracker
        progress_tracker = ProgressTrainerCallback(total_steps, progress_callback)

        # Custom Trainer subclass with progress reporting
        from transformers import TrainerCallback

        class ProgressReportingCallback(TrainerCallback):
            def __init__(self, tracker):
                self.tracker = tracker

            def on_step_end(self, args, state, control, **kwargs):
                self.tracker.on_step_end(state.global_step, state.log_history[-1] if state.log_history else {})

        # Setup data collator
        if training_config.model_type == "seq2seq":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=trainer.tokenizer,
                model=trainer.model,
                padding=True,
            )
        else:
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=trainer.tokenizer,
                mlm=False,
            )

        # Create HF Trainer with callback
        hf_trainer = Trainer(
            model=trainer.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[ProgressReportingCallback(progress_tracker)],
        )

        # Train
        logger.info("Starting training loop...")
        train_result = hf_trainer.train()

        progress_callback(95.0, "Saving model...")

        # Save model
        hf_trainer.save_model()
        trainer.tokenizer.save_pretrained(output_dir)

        # Save metadata
        metadata = {
            "model_type": "tinyforge_real",
            "model_name": model_name,
            "base_model": model_name,
            "version": "0.2.0",
            "job_id": job_id,
            "training_config": {
                "epochs": training_config.epochs,
                "batch_size": training_config.batch_size,
                "learning_rate": training_config.learning_rate,
                "use_peft": training_config.use_peft,
                "lora_r": training_config.lora_r if training_config.use_peft else None,
            },
            "training_results": {
                "train_loss": train_result.training_loss,
                "train_steps": train_result.global_step,
            },
            "data_path": str(dataset_path),
            "n_records": len(train_dataset),
            "output_dir": str(output_dir),
        }

        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        progress_callback(100.0, "Training complete!")

        logger.info(f"Training complete! Model saved to: {output_dir}")

        return metadata


class TrainingWorker:
    """
    Background worker that polls for and processes training jobs.
    """

    def __init__(self, config: Optional[WorkerConfig] = None):
        self.config = config or WorkerConfig()
        self.executor = TrainingExecutor(self.config.output_base_dir)
        self.client = httpx.Client(timeout=30.0)
        self._stop_event = Event()
        self._worker_thread: Optional[Thread] = None

    def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """Fetch pending jobs from the API."""
        try:
            response = self.client.get(
                f"{self.config.api_url}/api/jobs",
                params={"status": "pending", "limit": self.config.max_concurrent_jobs}
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch jobs: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching jobs: {e}")
            return []

    def process_job(self, job: Dict[str, Any]):
        """Process a single training job."""
        job_id = job["id"]
        logger.info(f"Processing job: {job_id} - {job.get('name', 'Unknown')}")

        # Create progress callback
        callback = ProgressCallback(job_id, self.config.api_url)

        try:
            # Execute training
            result = self.executor.execute(
                job_id=job_id,
                job_config=job,
                progress_callback=callback
            )
            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Job {job_id} failed: {error_msg}")
            callback.report_failure(error_msg)

        finally:
            callback.close()

    def _run_loop(self):
        """Main worker loop."""
        logger.info(f"Training worker started. Polling {self.config.api_url}")

        while not self._stop_event.is_set():
            try:
                # Get pending jobs
                jobs = self.get_pending_jobs()

                for job in jobs:
                    if self._stop_event.is_set():
                        break
                    self.process_job(job)

                # Wait before next poll
                self._stop_event.wait(timeout=self.config.poll_interval)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                self._stop_event.wait(timeout=self.config.poll_interval)

        logger.info("Training worker stopped")

    def start(self, blocking: bool = True):
        """
        Start the worker.

        Args:
            blocking: If True, blocks the current thread. If False, runs in background.
        """
        if blocking:
            self._run_loop()
        else:
            self._worker_thread = Thread(target=self._run_loop, daemon=True)
            self._worker_thread.start()

    def stop(self):
        """Stop the worker."""
        logger.info("Stopping training worker...")
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        self.client.close()


def main():
    """CLI entry point for the training worker."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TinyForgeAI Training Worker"
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("TINYFORGE_API_URL", "http://localhost:8001"),
        help="Dashboard API URL (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=int(os.getenv("TINYFORGE_POLL_INTERVAL", "5")),
        help="Polling interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("TINYFORGE_OUTPUT_DIR", "./output"),
        help="Base output directory for trained models"
    )

    args = parser.parse_args()

    config = WorkerConfig(
        api_url=args.api_url,
        poll_interval=args.poll_interval,
        output_base_dir=args.output_dir,
    )

    worker = TrainingWorker(config)

    try:
        worker.start(blocking=True)
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
