"""
TinyForgeAI CLI - foremforge.

A user-friendly command-line interface for common TinyForgeAI developer tasks:
- init: Initialize project structure
- train: Train a model (dry-run supported)
- train-url: Train directly from a URL (no data prep needed)
- augment: Generate synthetic training data
- export: Export model to microservice
- serve: Serve an exported microservice
- playground: Create shareable model playground

Usage:
    foremforge [COMMAND] [OPTIONS]

Examples:
    foremforge init --yes
    foremforge train --data examples/sample_qna.jsonl --out /tmp/model --dry-run
    foremforge train-url https://notion.so/my-faq-page --out /tmp/model
    foremforge augment --input data.jsonl --output augmented.jsonl --count 500
    foremforge export --model /tmp/model/model_stub.json --out /tmp/service
    foremforge serve --dir /tmp/service --port 8000
    foremforge playground --model /tmp/model --share
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Standard project directories to create
PROJECT_DIRS = [
    "backend",
    "backend/api",
    "backend/api/routes",
    "backend/training",
    "backend/exporter",
    "connectors",
    "inference_server",
    "examples",
    "docs",
    "tests",
    "docker",
    "cli",
]

# Starter files to create if missing
STARTER_FILES = {
    "README.md": """# TinyForgeAI Project

A lightweight platform for fine-tuning language models.

## Quick Start

```bash
# Initialize project structure
foremforge init --yes

# Train a model (dry-run)
foremforge train --data examples/sample_qna.jsonl --out /tmp/model --dry-run

# Export to microservice
foremforge export --model /tmp/model/model_stub.json --out /tmp/service

# Serve the microservice
foremforge serve --dir /tmp/service --port 8000
```

## Documentation

See `docs/` for detailed documentation.
""",
    ".env.example": """# TinyForgeAI Environment Configuration

# Application settings
APP_NAME=TinyForgeAI
LOG_LEVEL=INFO
DEBUG=false

# Database settings
DB_URL=sqlite:///:memory:

# Google Docs connector (mock mode by default)
GOOGLE_OAUTH_DISABLED=true
""",
}


@click.group()
@click.version_option(version="0.1.0", prog_name="foremforge")
def cli():
    """TinyForgeAI CLI - Developer tools for model fine-tuning and deployment."""
    pass


@cli.command()
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompt and create all directories/files.",
)
def init(yes: bool):
    """Initialize TinyForgeAI project structure.

    Creates standard project folders and starter files if they don't exist.
    """
    cwd = Path.cwd()
    created_dirs = []
    created_files = []

    # Check what needs to be created
    dirs_to_create = [d for d in PROJECT_DIRS if not (cwd / d).exists()]
    files_to_create = [f for f in STARTER_FILES.keys() if not (cwd / f).exists()]

    if not dirs_to_create and not files_to_create:
        click.echo("Project structure already exists. Nothing to create.")
        return

    # Show what will be created
    if dirs_to_create:
        click.echo("Directories to create:")
        for d in dirs_to_create:
            click.echo(f"  - {d}/")

    if files_to_create:
        click.echo("Files to create:")
        for f in files_to_create:
            click.echo(f"  - {f}")

    # Confirm unless --yes provided
    if not yes:
        if not click.confirm("Do you want to proceed?"):
            click.echo("Aborted.")
            raise SystemExit(1)

    # Create directories
    for dir_path in dirs_to_create:
        full_path = cwd / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        created_dirs.append(dir_path)
        click.echo(f"Created directory: {dir_path}/")

    # Create starter files
    for file_name, content in STARTER_FILES.items():
        file_path = cwd / file_name
        if not file_path.exists():
            file_path.write_text(content)
            created_files.append(file_name)
            click.echo(f"Created file: {file_name}")

    # Summary
    click.echo("")
    click.echo(f"Initialization complete!")
    click.echo(f"  Created {len(created_dirs)} directories")
    click.echo(f"  Created {len(created_files)} files")


@cli.command()
@click.option(
    "--data", "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to the JSONL training data file.",
)
@click.option(
    "--out", "-o",
    required=True,
    type=click.Path(),
    help="Directory to write model artifacts.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate data and create stub artifact without training.",
)
@click.option(
    "--use-lora",
    is_flag=True,
    help="Apply LoRA adapter to the model.",
)
def train(data: str, out: str, dry_run: bool, use_lora: bool):
    """Train a TinyForgeAI model on JSONL data.

    Loads the dataset, validates it, and creates a model artifact.
    Use --dry-run to validate without actual training.
    """
    try:
        # Import training module
        from backend.training.train import run_training
        from backend.training.dataset import load_jsonl, summarize_dataset

        click.echo(f"Training model from: {data}")
        click.echo(f"Output directory: {out}")

        if dry_run:
            click.echo("Mode: dry-run (validation only)")

        if use_lora:
            click.echo("LoRA adapter: enabled")

        # Run training
        run_training(
            data_path=data,
            output_dir=out,
            dry_run=dry_run,
            use_lora=use_lora,
        )

        # Load and display summary
        records = load_jsonl(data)
        summary = summarize_dataset(records)

        click.echo("")
        click.echo("Training Summary:")
        click.echo(f"  n_records: {summary['n_records']}")
        click.echo(f"  artifact: {Path(out) / 'model_stub.json'}")

        if dry_run:
            click.echo("  status: dry-run complete")
        else:
            click.echo("  status: training complete")

    except FileNotFoundError as e:
        click.echo(f"Error: File not found - {e}", err=True)
        raise SystemExit(1)
    except ValueError as e:
        click.echo(f"Error: Data validation failed - {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command(name="export")
@click.option(
    "--model", "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to the model artifact file (JSON).",
)
@click.option(
    "--out", "-o",
    required=True,
    type=click.Path(),
    help="Directory where the microservice will be created.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output directory if it exists.",
)
@click.option(
    "--export-onnx",
    is_flag=True,
    help="Export model to ONNX format and create quantized version.",
)
def export_cmd(model: str, out: str, overwrite: bool, export_onnx: bool):
    """Export a trained model to an inference microservice.

    Creates a self-contained microservice directory with all necessary
    files for deployment with uvicorn or Docker.
    """
    try:
        from backend.exporter.builder import build

        click.echo(f"Exporting model: {model}")
        click.echo(f"Output directory: {out}")

        if overwrite:
            click.echo("Overwrite mode: enabled")

        if export_onnx:
            click.echo("ONNX export: enabled")

        # Run builder
        report = build(
            model_path=model,
            output_dir=out,
            overwrite=overwrite,
            export_onnx=export_onnx,
        )

        click.echo("")
        click.echo("Export Summary:")
        click.echo(f"  microservice: {out}")
        click.echo(f"  metadata: {Path(out) / 'model_metadata.json'}")

        if export_onnx and report:
            click.echo(f"  onnx_path: {report['onnx_path']}")
            click.echo(f"  quantized_path: {report['quantized_path']}")

        click.echo("  status: export complete")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except FileExistsError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option(
    "--dir", "-d",
    "service_dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing the exported microservice.",
)
@click.option(
    "--port", "-p",
    default=8000,
    type=int,
    help="Port to run the server on (default: 8000).",
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the server to (default: 0.0.0.0).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the uvicorn command without executing it.",
)
def serve(service_dir: str, port: int, host: str, dry_run: bool):
    """Serve an exported microservice with uvicorn.

    Starts a uvicorn server to serve the inference API.
    Use --dry-run to see the command without executing it.
    """
    service_path = Path(service_dir)

    # Verify app.py exists
    app_file = service_path / "app.py"
    if not app_file.exists():
        click.echo(f"Error: app.py not found in {service_dir}", err=True)
        raise SystemExit(1)

    # Build uvicorn command
    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn",
        "app:app",
        "--host", host,
        "--port", str(port),
    ]

    cmd_str = f"uvicorn app:app --host {host} --port {port}"

    if dry_run:
        click.echo("Dry-run mode: would execute the following command:")
        click.echo(f"  cd {service_dir}")
        click.echo(f"  {cmd_str}")
        return

    click.echo(f"Starting server in: {service_dir}")
    click.echo(f"Command: {cmd_str}")
    click.echo(f"Server will be available at: http://{host}:{port}")
    click.echo("")

    try:
        # Run uvicorn in the service directory
        process = subprocess.Popen(
            uvicorn_cmd,
            cwd=service_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output
        for line in iter(process.stdout.readline, ""):
            click.echo(line, nl=False)

        process.wait()
        if process.returncode != 0:
            raise SystemExit(process.returncode)

    except KeyboardInterrupt:
        click.echo("\nShutting down server...")
        process.terminate()
        process.wait()
    except FileNotFoundError:
        click.echo("Error: uvicorn not found. Install it with: pip install uvicorn", err=True)
        raise SystemExit(1)


@cli.command(name="train-url")
@click.argument("url")
@click.option(
    "--out", "-o",
    default="./url_model",
    type=click.Path(),
    help="Output directory for the trained model.",
)
@click.option(
    "--model", "-m",
    default="t5-small",
    help="Base model to fine-tune (default: t5-small).",
)
@click.option(
    "--epochs", "-e",
    default=3,
    type=int,
    help="Number of training epochs (default: 3).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview extraction and create stub model without training.",
)
@click.option(
    "--no-augment",
    is_flag=True,
    help="Disable automatic data augmentation.",
)
@click.option(
    "--preview",
    is_flag=True,
    help="Only preview extracted data, don't train.",
)
@click.option(
    "--notion-token",
    envvar="NOTION_TOKEN",
    help="Notion API token for private pages.",
)
def train_url(url: str, out: str, model: str, epochs: int, dry_run: bool,
              no_augment: bool, preview: bool, notion_token: str):
    """Train a model directly from a URL.

    Supports: Notion, Google Docs, GitHub, websites, and raw files.
    No data preparation needed - just paste the URL!

    Examples:
        foremforge train-url https://notion.so/my-faq
        foremforge train-url https://docs.google.com/document/d/xxx
        foremforge train-url https://example.com/faq --preview
    """
    try:
        from backend.url_trainer import URLTrainer, URLTrainConfig

        config = URLTrainConfig(
            model_name=model,
            epochs=epochs,
            augment=not no_augment,
            dry_run=dry_run,
            output_dir=out,
            notion_token=notion_token,
        )

        trainer = URLTrainer(config)

        if preview:
            click.echo(f"\nPreviewing: {url}\n")
            data = trainer.preview(url)

            click.echo(f"Source Type: {data.source_type}")
            click.echo(f"Title: {data.title}")
            click.echo(f"Samples Extracted: {len(data.samples)}")
            click.echo("\nSample Preview (first 3):")
            click.echo("-" * 50)

            for i, sample in enumerate(data.samples[:3]):
                click.echo(f"\n[{i+1}] Input: {sample['input'][:80]}...")
                click.echo(f"    Output: {sample['output'][:80]}...")
            return

        click.echo(f"\nTraining from URL: {url}\n")
        result = trainer.train_from_url(url)

        click.echo("\n" + "=" * 50)
        click.echo("Training Complete!")
        click.echo("=" * 50)
        click.echo(f"Samples Extracted: {result['total_samples']}")
        click.echo(f"Augmented: {result['augmented']}")
        click.echo(f"Output: {out}")

        status = result.get("training_result", {}).get("status", "unknown")
        if status == "success":
            loss = result["training_result"].get("training_loss", "N/A")
            click.echo(f"Training Loss: {loss}")
        elif status == "stub_created":
            click.echo("Status: Dry run - stub model created")

    except ImportError as e:
        click.echo(f"Error: Missing dependency - {e}", err=True)
        click.echo("Install with: pip install httpx beautifulsoup4", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option(
    "--input", "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input JSONL file with training examples.",
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output JSONL file for augmented data.",
)
@click.option(
    "--count", "-n",
    default=500,
    type=int,
    help="Target number of samples (default: 500).",
)
@click.option(
    "--strategies", "-s",
    multiple=True,
    default=["synonym", "template", "paraphrase"],
    help="Augmentation strategies to use.",
)
@click.option(
    "--use-llm",
    is_flag=True,
    help="Use LLM for high-quality augmentation (requires API key).",
)
@click.option(
    "--llm-provider",
    default="anthropic",
    type=click.Choice(["anthropic", "openai"]),
    help="LLM provider (default: anthropic).",
)
def augment(input_file: str, output: str, count: int, strategies: tuple,
            use_llm: bool, llm_provider: str):
    """Generate synthetic training data from examples.

    Takes a small set of examples and generates more training samples
    using various augmentation strategies.

    Examples:
        foremforge augment -i examples.jsonl -o augmented.jsonl -n 500
        foremforge augment -i data.jsonl -o more_data.jsonl --use-llm
    """
    try:
        from backend.augment import DataGenerator, AugmentConfig

        strategy_list = list(strategies)
        if use_llm:
            strategy_list.append("llm")

        config = AugmentConfig(
            target_count=count,
            strategies=strategy_list,
            llm_provider=llm_provider,
        )

        click.echo(f"Input: {input_file}")
        click.echo(f"Target count: {count}")
        click.echo(f"Strategies: {', '.join(strategy_list)}")

        # Load examples
        examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        click.echo(f"Loaded {len(examples)} examples")

        # Generate
        generator = DataGenerator(config)
        samples = generator.generate(examples, count)

        # Save
        generator.save(samples, output)

        click.echo("\n" + "=" * 50)
        click.echo("Augmentation Complete!")
        click.echo("=" * 50)
        click.echo(f"Generated: {len(samples)} samples")
        click.echo(f"Output: {output}")

    except ImportError as e:
        click.echo(f"Error: Missing module - {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option(
    "--model", "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained model directory.",
)
@click.option(
    "--port", "-p",
    default=8080,
    type=int,
    help="Port to run playground on (default: 8080).",
)
@click.option(
    "--title", "-t",
    default="TinyForge Playground",
    help="Playground title.",
)
@click.option(
    "--share",
    is_flag=True,
    help="Create a public shareable link.",
)
@click.option(
    "--export",
    "export_path",
    type=click.Path(),
    help="Export as standalone HTML file instead of serving.",
)
@click.option(
    "--examples",
    multiple=True,
    help="Example inputs to show in the playground.",
)
def playground(model: str, port: int, title: str, share: bool,
               export_path: str, examples: tuple):
    """Start a shareable playground for your model.

    Creates a beautiful web interface for testing your trained model.
    Use --share to get a public URL anyone can use.
    Use --export to create a standalone HTML file.

    Examples:
        foremforge playground --model ./my_model
        foremforge playground --model ./my_model --share
        foremforge playground --model ./my_model --export demo.html
    """
    try:
        if export_path:
            # Export as HTML
            from backend.playground import PlaygroundExporter, ExportConfig

            config = ExportConfig(title=title)
            exporter = PlaygroundExporter(config)
            output = exporter.export(model, export_path)

            click.echo("\n" + "=" * 50)
            click.echo("Playground Exported!")
            click.echo("=" * 50)
            click.echo(f"Output: {output}")
            click.echo("\nOpen in any browser - works offline!")

        else:
            # Start server
            from backend.playground import PlaygroundServer, PlaygroundConfig

            config = PlaygroundConfig(
                model_path=model,
                title=title,
                port=port,
                share=share,
                examples=list(examples),
            )

            server = PlaygroundServer(config)

            click.echo("\n" + "=" * 50)
            click.echo("Starting Playground...")
            click.echo("=" * 50)

            server.start()

    except ImportError as e:
        click.echo(f"Error: Missing dependency - {e}", err=True)
        click.echo("Install with: pip install fastapi uvicorn", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
