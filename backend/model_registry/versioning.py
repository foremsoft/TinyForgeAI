"""
Semantic versioning utilities for model versioning.

Implements semantic versioning (semver) for models with support for
major, minor, patch versions and pre-release tags.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union


@dataclass(order=True)
class SemanticVersion:
    """
    Semantic version representation following SemVer 2.0.0.

    Format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

    Examples:
        1.0.0
        1.2.3-alpha
        2.0.0-beta.1
        1.0.0+build.123
    """

    major: int = 0
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = field(default=None, compare=False)
    build: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        """Validate version components."""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError("Version components must be non-negative")

    def __str__(self) -> str:
        """Convert to string representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __hash__(self) -> int:
        """Make hashable for use in sets/dicts."""
        return hash((self.major, self.minor, self.patch, self.prerelease))

    @classmethod
    def from_string(cls, version_str: str) -> "SemanticVersion":
        """
        Parse a version string into SemanticVersion.

        Args:
            version_str: Version string like "1.2.3" or "v1.2.3-alpha"

        Returns:
            SemanticVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        # Remove leading 'v' if present
        if version_str.startswith("v"):
            version_str = version_str[1:]

        # Regex for semver with optional prerelease and build
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$"
        match = re.match(pattern, version_str)

        if not match:
            # Try simple vN format (e.g., v1, v2)
            simple_pattern = r"^v?(\d+)$"
            simple_match = re.match(simple_pattern, version_str)
            if simple_match:
                return cls(major=int(simple_match.group(1)), minor=0, patch=0)
            raise ValueError(f"Invalid version string: {version_str}")

        major, minor, patch, prerelease, build = match.groups()
        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            prerelease=prerelease,
            build=build,
        )

    def bump_major(self) -> "SemanticVersion":
        """Return new version with major bumped, minor/patch reset."""
        return SemanticVersion(major=self.major + 1, minor=0, patch=0)

    def bump_minor(self) -> "SemanticVersion":
        """Return new version with minor bumped, patch reset."""
        return SemanticVersion(major=self.major, minor=self.minor + 1, patch=0)

    def bump_patch(self) -> "SemanticVersion":
        """Return new version with patch bumped."""
        return SemanticVersion(major=self.major, minor=self.minor, patch=self.patch + 1)

    def with_prerelease(self, prerelease: str) -> "SemanticVersion":
        """Return new version with prerelease tag."""
        return SemanticVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            prerelease=prerelease,
            build=self.build,
        )

    def with_build(self, build: str) -> "SemanticVersion":
        """Return new version with build metadata."""
        return SemanticVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            prerelease=self.prerelease,
            build=build,
        )

    def to_tuple(self) -> Tuple[int, int, int]:
        """Return version as (major, minor, patch) tuple."""
        return (self.major, self.minor, self.patch)

    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version."""
        return self.prerelease is not None

    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """
        Check if this version is API-compatible with another.

        Compatible means same major version (for major > 0) or
        same major.minor (for major == 0).
        """
        if self.major == 0 and other.major == 0:
            return self.minor == other.minor
        return self.major == other.major


def parse_version(version_str: str) -> SemanticVersion:
    """
    Parse a version string into SemanticVersion.

    Convenience function that wraps SemanticVersion.from_string().

    Args:
        version_str: Version string

    Returns:
        SemanticVersion instance
    """
    return SemanticVersion.from_string(version_str)


def compare_versions(v1: Union[str, SemanticVersion], v2: Union[str, SemanticVersion]) -> int:
    """
    Compare two versions.

    Args:
        v1: First version (string or SemanticVersion)
        v2: Second version (string or SemanticVersion)

    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    if isinstance(v1, str):
        v1 = parse_version(v1)
    if isinstance(v2, str):
        v2 = parse_version(v2)

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    return 0


def get_next_version(
    current: Union[str, SemanticVersion],
    bump_type: str = "patch",
) -> SemanticVersion:
    """
    Get the next version based on bump type.

    Args:
        current: Current version
        bump_type: One of "major", "minor", "patch"

    Returns:
        Next SemanticVersion
    """
    if isinstance(current, str):
        current = parse_version(current)

    if bump_type == "major":
        return current.bump_major()
    elif bump_type == "minor":
        return current.bump_minor()
    elif bump_type == "patch":
        return current.bump_patch()
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
