"""
Model Registry CLI - Command-line interface for model versioning operations.

Usage:
    python -m backend.model_registry.cli list
    python -m backend.model_registry.cli versions <model_name>
    python -m backend.model_registry.cli info <model_name> [version]
    python -m backend.model_registry.cli register <model_name> <version> [--artifacts PATH]
    python -m backend.model_registry.cli activate <model_name> <version>
    python -m backend.model_registry.cli deprecate <model_name> <version> [--reason TEXT]
    python -m backend.model_registry.cli compare <model_name> <version1> <version2>
    python -m backend.model_registry.cli export <model_name> <version> <output_path>
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from backend.model_registry.registry import (
    ModelRegistry,
    ModelMetadata,
    TrainingConfig,
    TrainingMetrics,
    ModelCard,
    VersionStatus,
)
from backend.model_registry.versioning import SemanticVersion, parse_version


# Default registry path
DEFAULT_REGISTRY_PATH = os.environ.get(
    "MODEL_REGISTRY_PATH",
    "./model_registry"
)


def get_registry(registry_path: Optional[str] = None) -> ModelRegistry:
    """Get model registry instance."""
    path = registry_path or DEFAULT_REGISTRY_PATH
    return ModelRegistry(path)


def cmd_list(args):
    """List all models in the registry."""
    registry = get_registry(args.registry_path)
    models = registry.list_models()

    if not models:
        print("No models registered.")
        return

    print(f"\nModels in registry ({len(models)} total):\n")
    print(f"{'Model Name':<30} {'Versions':<10} {'Current':<15} {'Status':<12}")
    print("-" * 70)

    for model_name in sorted(models):
        versions = registry.list_versions(model_name)
        current = registry.get_current_version(model_name)
        current_str = current.version if current else "None"
        status_str = current.status.value if current else "N/A"
        print(f"{model_name:<30} {len(versions):<10} {current_str:<15} {status_str:<12}")


def cmd_versions(args):
    """List all versions of a model."""
    registry = get_registry(args.registry_path)
    versions = registry.list_versions(args.model_name)

    if not versions:
        print(f"No versions found for model '{args.model_name}'")
        return

    print(f"\nVersions of '{args.model_name}' ({len(versions)} total):\n")
    print(f"{'Version':<15} {'Status':<12} {'Created':<25} {'Tags':<30}")
    print("-" * 85)

    for v in versions:
        created = v.metadata.created_at[:19] if v.metadata.created_at else "Unknown"
        tags = ", ".join(v.metadata.tags[:3]) if v.metadata.tags else ""
        if len(v.metadata.tags) > 3:
            tags += "..."
        print(f"{v.version:<15} {v.status.value:<12} {created:<25} {tags:<30}")


def cmd_info(args):
    """Show detailed info about a model version."""
    registry = get_registry(args.registry_path)

    version = registry.get_version(args.model_name, args.version)
    if version is None:
        if args.version:
            print(f"Version '{args.version}' not found for model '{args.model_name}'")
        else:
            print(f"Model '{args.model_name}' not found")
        return 1

    m = version.metadata
    print(f"\n{'='*60}")
    print(f"Model: {m.name}")
    print(f"Version: {m.version}")
    print(f"{'='*60}")

    print(f"\nStatus: {m.status.value.upper()}")
    print(f"Model Type: {m.model_type}")
    if m.base_model:
        print(f"Base Model: {m.base_model}")
    if m.task_type:
        print(f"Task Type: {m.task_type}")
    if m.parent_version:
        print(f"Parent Version: {m.parent_version}")

    print(f"\nCreated: {m.created_at}")
    if m.activated_at:
        print(f"Activated: {m.activated_at}")
    if m.deprecated_at:
        print(f"Deprecated: {m.deprecated_at}")

    if m.data_path:
        print(f"\nTraining Data: {m.data_path}")
        print(f"Records: {m.n_records:,}")

    if m.training_config:
        print(f"\nTraining Configuration:")
        config = m.training_config.to_dict()
        for key, value in config.items():
            if key != "custom_config":
                print(f"  {key}: {value}")

    if m.training_metrics:
        print(f"\nTraining Metrics:")
        metrics = m.training_metrics.to_dict()
        for key, value in metrics.items():
            if key != "custom_metrics":
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    if m.tags:
        print(f"\nTags: {', '.join(m.tags)}")

    print(f"License: {m.license}")

    if m.changelog:
        print(f"\nChangelog:\n{m.changelog}")

    # List artifacts
    artifacts = version.list_artifacts()
    if artifacts:
        print(f"\nArtifacts ({len(artifacts)}):")
        for artifact in artifacts:
            print(f"  - {artifact}")


def cmd_register(args):
    """Register a new model version."""
    registry = get_registry(args.registry_path)

    # Check if version already exists
    existing = registry.get_version(args.model_name, args.version)
    if existing:
        print(f"Version '{args.version}' already exists for model '{args.model_name}'")
        return 1

    # Create metadata
    metadata = ModelMetadata(
        name=args.model_name,
        version=args.version,
        model_type=args.model_type or "custom",
        base_model=args.base_model,
        task_type=args.task_type,
        changelog=args.changelog or "",
        tags=args.tags.split(",") if args.tags else [],
    )

    # Check for artifacts
    artifacts_path = Path(args.artifacts) if args.artifacts else None
    if artifacts_path and not artifacts_path.exists():
        print(f"Artifacts path '{artifacts_path}' does not exist")
        return 1

    try:
        version = registry.register_version(
            args.model_name,
            args.version,
            metadata=metadata,
            artifacts_path=artifacts_path,
            activate=args.activate,
        )
        print(f"Registered {args.model_name} v{args.version}")
        if args.activate:
            print(f"  Status: ACTIVE (set as current version)")
        else:
            print(f"  Status: DRAFT")
        if artifacts_path:
            print(f"  Artifacts copied from: {artifacts_path}")

    except Exception as e:
        print(f"Failed to register version: {e}")
        return 1


def cmd_activate(args):
    """Activate a model version."""
    registry = get_registry(args.registry_path)

    try:
        version = registry.activate_version(args.model_name, args.version)
        print(f"Activated {args.model_name} v{args.version}")
        print(f"  This is now the current version")
    except ValueError as e:
        print(f"Failed to activate: {e}")
        return 1


def cmd_deprecate(args):
    """Deprecate a model version."""
    registry = get_registry(args.registry_path)

    try:
        version = registry.deprecate_version(
            args.model_name,
            args.version,
            reason=args.reason or "",
        )
        print(f"Deprecated {args.model_name} v{args.version}")
        if args.reason:
            print(f"  Reason: {args.reason}")
    except ValueError as e:
        print(f"Failed to deprecate: {e}")
        return 1


def cmd_archive(args):
    """Archive a model version."""
    registry = get_registry(args.registry_path)

    try:
        version = registry.archive_version(args.model_name, args.version)
        print(f"Archived {args.model_name} v{args.version}")
    except ValueError as e:
        print(f"Failed to archive: {e}")
        return 1


def cmd_delete(args):
    """Delete a model version."""
    registry = get_registry(args.registry_path)

    if args.all_versions:
        # Delete entire model
        try:
            result = registry.delete_model(args.model_name, force=args.force)
            if result:
                print(f"Deleted model '{args.model_name}' and all versions")
            else:
                print(f"Model '{args.model_name}' not found")
        except ValueError as e:
            print(f"Failed to delete: {e}")
            return 1
    else:
        # Delete specific version
        if not args.version:
            print("Version required. Use --all to delete entire model.")
            return 1
        try:
            result = registry.delete_version(
                args.model_name,
                args.version,
                force=args.force,
            )
            if result:
                print(f"Deleted {args.model_name} v{args.version}")
            else:
                print(f"Version not found")
        except ValueError as e:
            print(f"Failed to delete: {e}")
            return 1


def cmd_compare(args):
    """Compare two model versions."""
    registry = get_registry(args.registry_path)

    try:
        comparison = registry.compare_versions(
            args.model_name,
            args.version1,
            args.version2,
        )
    except ValueError as e:
        print(f"Failed to compare: {e}")
        return 1

    print(f"\nComparison: {args.model_name}")
    print(f"  v{comparison['version1']} vs v{comparison['version2']}")
    print("=" * 50)

    metrics = comparison.get("metrics_comparison", {})
    if metrics:
        print("\nMetrics Comparison:")
        for metric_name, data in metrics.items():
            v1 = data['v1']
            v2 = data['v2']
            diff = data['diff']
            improved = data['improved']
            indicator = "+" if improved else "-" if diff != 0 else "="

            if isinstance(v1, float):
                print(f"  {metric_name}:")
                print(f"    v{comparison['version1']}: {v1:.4f}")
                print(f"    v{comparison['version2']}: {v2:.4f}")
                print(f"    Change: {diff:+.4f} [{indicator}]")
            else:
                print(f"  {metric_name}:")
                print(f"    v{comparison['version1']}: {v1}")
                print(f"    v{comparison['version2']}: {v2}")

    config_changes = comparison.get("config_changes", {})
    if config_changes:
        print("\nConfiguration Changes:")
        for key, data in config_changes.items():
            print(f"  {key}:")
            print(f"    v{comparison['version1']}: {data['v1']}")
            print(f"    v{comparison['version2']}: {data['v2']}")

    if not metrics and not config_changes:
        print("\nNo differences found in metrics or configuration.")


def cmd_export(args):
    """Export a model version."""
    registry = get_registry(args.registry_path)

    output_path = Path(args.output_path)
    if output_path.exists() and not args.force:
        print(f"Output path '{output_path}' already exists. Use --force to overwrite.")
        return 1

    try:
        result = registry.export_model(
            args.model_name,
            args.version,
            output_path,
        )
        print(f"Exported {args.model_name} v{args.version} to {result}")
    except ValueError as e:
        print(f"Failed to export: {e}")
        return 1


def cmd_rollback(args):
    """Rollback to a previous version."""
    registry = get_registry(args.registry_path)

    try:
        version = registry.rollback(args.model_name, args.target_version)
        print(f"Rolled back {args.model_name} to v{args.target_version}")
        print(f"  Status: ACTIVE")
    except ValueError as e:
        print(f"Failed to rollback: {e}")
        return 1


def cmd_search(args):
    """Search for models."""
    registry = get_registry(args.registry_path)

    status = None
    if args.status:
        try:
            status = VersionStatus(args.status)
        except ValueError:
            print(f"Invalid status: {args.status}")
            print(f"Valid values: {', '.join(s.value for s in VersionStatus)}")
            return 1

    tags = args.tags.split(",") if args.tags else None

    results = list(registry.search_models(
        task_type=args.task_type,
        tags=tags,
        status=status,
    ))

    if not results:
        print("No models found matching criteria.")
        return

    print(f"\nFound {len(results)} model version(s):\n")
    print(f"{'Model':<25} {'Version':<12} {'Task Type':<15} {'Status':<10}")
    print("-" * 65)

    for v in results:
        task = v.metadata.task_type or "N/A"
        print(f"{v.name:<25} {v.version:<12} {task:<15} {v.status.value:<10}")


def cmd_next_version(args):
    """Get the next version number."""
    registry = get_registry(args.registry_path)

    next_v = registry.get_next_version(args.model_name, args.bump)
    print(next_v)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Model Registry CLI - Manage model versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--registry-path",
        "-r",
        help=f"Path to model registry (default: {DEFAULT_REGISTRY_PATH})",
        default=None,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List all models")

    # versions command
    versions_parser = subparsers.add_parser("versions", help="List versions of a model")
    versions_parser.add_argument("model_name", help="Model name")

    # info command
    info_parser = subparsers.add_parser("info", help="Show model version info")
    info_parser.add_argument("model_name", help="Model name")
    info_parser.add_argument("version", nargs="?", help="Version (default: current)")

    # register command
    register_parser = subparsers.add_parser("register", help="Register a new version")
    register_parser.add_argument("model_name", help="Model name")
    register_parser.add_argument("version", help="Version string (e.g., 1.0.0)")
    register_parser.add_argument("--artifacts", "-a", help="Path to artifacts directory")
    register_parser.add_argument("--model-type", help="Model type")
    register_parser.add_argument("--base-model", help="Base model name")
    register_parser.add_argument("--task-type", help="Task type")
    register_parser.add_argument("--changelog", "-c", help="Changelog entry")
    register_parser.add_argument("--tags", "-t", help="Comma-separated tags")
    register_parser.add_argument("--activate", action="store_true", help="Activate immediately")

    # activate command
    activate_parser = subparsers.add_parser("activate", help="Activate a version")
    activate_parser.add_argument("model_name", help="Model name")
    activate_parser.add_argument("version", help="Version to activate")

    # deprecate command
    deprecate_parser = subparsers.add_parser("deprecate", help="Deprecate a version")
    deprecate_parser.add_argument("model_name", help="Model name")
    deprecate_parser.add_argument("version", help="Version to deprecate")
    deprecate_parser.add_argument("--reason", help="Reason for deprecation")

    # archive command
    archive_parser = subparsers.add_parser("archive", help="Archive a version")
    archive_parser.add_argument("model_name", help="Model name")
    archive_parser.add_argument("version", help="Version to archive")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a version or model")
    delete_parser.add_argument("model_name", help="Model name")
    delete_parser.add_argument("version", nargs="?", help="Version to delete")
    delete_parser.add_argument("--all", dest="all_versions", action="store_true", help="Delete entire model")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Force delete")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two versions")
    compare_parser.add_argument("model_name", help="Model name")
    compare_parser.add_argument("version1", help="First version")
    compare_parser.add_argument("version2", help="Second version")

    # export command
    export_parser = subparsers.add_parser("export", help="Export a model version")
    export_parser.add_argument("model_name", help="Model name")
    export_parser.add_argument("version", help="Version to export")
    export_parser.add_argument("output_path", help="Output directory")
    export_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing")

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to a previous version")
    rollback_parser.add_argument("model_name", help="Model name")
    rollback_parser.add_argument("target_version", help="Version to rollback to")

    # search command
    search_parser = subparsers.add_parser("search", help="Search for models")
    search_parser.add_argument("--task-type", help="Filter by task type")
    search_parser.add_argument("--tags", help="Filter by tags (comma-separated)")
    search_parser.add_argument("--status", help="Filter by status")

    # next-version command
    next_parser = subparsers.add_parser("next-version", help="Get next version number")
    next_parser.add_argument("model_name", help="Model name")
    next_parser.add_argument("--bump", choices=["major", "minor", "patch"], default="patch", help="Bump type")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to command handler
    commands = {
        "list": cmd_list,
        "versions": cmd_versions,
        "info": cmd_info,
        "register": cmd_register,
        "activate": cmd_activate,
        "deprecate": cmd_deprecate,
        "archive": cmd_archive,
        "delete": cmd_delete,
        "compare": cmd_compare,
        "export": cmd_export,
        "rollback": cmd_rollback,
        "search": cmd_search,
        "next-version": cmd_next_version,
    }

    handler = commands.get(args.command)
    if handler:
        result = handler(args)
        return result if result else 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
