"""
TinyForgeAI Exporter Builder.

Packages trained model artifacts into self-contained inference microservice
templates ready for deployment with uvicorn or Docker.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_template_dir() -> Path:
    """Get the path to the inference template directory."""
    return Path(__file__).parent / "templates" / "inference_template"


def build(model_path: str, output_dir: str, overwrite: bool = False) -> None:
    """
    Build an inference microservice from a model artifact.

    Args:
        model_path: Path to the model artifact file or directory.
        output_dir: Directory where the microservice will be created.
        overwrite: If True, overwrite existing output directory.

    Raises:
        FileNotFoundError: If model_path does not exist.
        FileExistsError: If output_dir exists and overwrite is False.
    """
    model_path_obj = Path(model_path)
    output_path = Path(output_dir)
    template_dir = get_template_dir()

    # Validate model path exists
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Check if output directory already exists
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    # Copy the template directory to output
    shutil.copytree(template_dir, output_path)

    # Create model metadata
    metadata = {
        "model_path": str(model_path),
        "created_time": datetime.now(timezone.utc).isoformat(),
        "source": "tinyforge-exporter",
        "model_stub": True,
    }

    # Write model_metadata.json to output directory
    metadata_path = output_path / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully built inference service at: {output_path}")


def main() -> None:
    """CLI entry point for the exporter builder."""
    parser = argparse.ArgumentParser(
        description="Build an inference microservice from a trained model."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the model artifact file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the microservice will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists.",
    )

    args = parser.parse_args()

    try:
        build(
            model_path=args.model_path,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
