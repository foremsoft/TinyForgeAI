#!/usr/bin/env python3
"""
Generate OpenAPI specification for TinyForgeAI Dashboard API.

This script generates a static OpenAPI JSON/YAML file that can be used for:
- API documentation hosting
- Client SDK generation
- API testing tools
- CI/CD validation

Usage:
    python scripts/generate_openapi.py
    python scripts/generate_openapi.py --format yaml
    python scripts/generate_openapi.py --output docs/api/openapi.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_openapi(output_format: str = "json", output_path: str = None) -> str:
    """
    Generate OpenAPI specification.

    Args:
        output_format: Output format ('json' or 'yaml')
        output_path: Optional output file path

    Returns:
        OpenAPI spec as string
    """
    # Import the app to get the OpenAPI schema
    from services.dashboard_api.main import app

    # Get the OpenAPI schema
    openapi_schema = app.openapi()

    # Add additional metadata
    openapi_schema["info"]["x-logo"] = {
        "url": "https://raw.githubusercontent.com/foremsoft/tinyforgeai/main/docs/logo.png",
        "altText": "TinyForgeAI Logo",
    }

    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8001",
            "description": "Local development server",
        },
        {
            "url": "https://api.tinyforge.example.com",
            "description": "Production server (example)",
        },
    ]

    # Format output
    if output_format == "yaml":
        try:
            import yaml
            spec_content = yaml.dump(openapi_schema, default_flow_style=False, sort_keys=False)
        except ImportError:
            print("Warning: PyYAML not installed, falling back to JSON")
            spec_content = json.dumps(openapi_schema, indent=2)
            output_format = "json"
    else:
        spec_content = json.dumps(openapi_schema, indent=2)

    # Write to file if path specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(spec_content)
        print(f"OpenAPI spec written to: {output_file}")
    else:
        # Default output location
        default_path = project_root / "docs" / "api" / f"openapi.{output_format}"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        default_path.write_text(spec_content)
        print(f"OpenAPI spec written to: {default_path}")

    return spec_content


def validate_openapi(spec_path: str) -> bool:
    """
    Validate an OpenAPI specification file.

    Args:
        spec_path: Path to OpenAPI spec file

    Returns:
        True if valid, False otherwise
    """
    try:
        from openapi_spec_validator import validate_spec
        from openapi_spec_validator.readers import read_from_filename

        spec_dict, _ = read_from_filename(spec_path)
        validate_spec(spec_dict)
        print(f"OpenAPI spec is valid: {spec_path}")
        return True
    except ImportError:
        print("Warning: openapi-spec-validator not installed, skipping validation")
        return True
    except Exception as e:
        print(f"OpenAPI validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI specification for TinyForgeAI Dashboard API"
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: docs/api/openapi.{format})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated spec",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print spec to stdout",
    )

    args = parser.parse_args()

    # Generate spec
    spec_content = generate_openapi(
        output_format=args.format,
        output_path=args.output,
    )

    if args.print:
        print(spec_content)

    # Validate if requested
    if args.validate:
        spec_path = args.output or str(project_root / "docs" / "api" / f"openapi.{args.format}")
        if not validate_openapi(spec_path):
            sys.exit(1)

    print("OpenAPI specification generated successfully!")


if __name__ == "__main__":
    main()
