#!/bin/bash
# TinyForgeAI Release Preparation Script
# Usage: ./releases/prepare_release.sh <version>
# Example: ./releases/prepare_release.sh 0.1.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.0"
    exit 1
fi

VERSION="$1"

# Validate version format (semver)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_error "Invalid version format: $VERSION"
    echo "Version must be in semver format: X.Y.Z (e.g., 0.1.0)"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           TinyForgeAI Release Preparation                  ║"
echo "║                    Version: $VERSION                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check working directory is clean
print_step "Checking git status..."
if [ -n "$(git status --porcelain)" ]; then
    print_warning "Working directory has uncommitted changes"
    git status --short
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "Working directory is clean"
fi

# Step 2: Run tests
print_step "Running test suite..."
if python -m pytest -q; then
    print_success "All tests passed"
else
    print_error "Tests failed! Fix tests before releasing."
    exit 1
fi

# Step 3: Run linting (if flake8 is available)
print_step "Running linter..."
if command -v flake8 &> /dev/null; then
    if flake8 cli backend connectors inference_server --count --select=E9,F63,F7,F82 --show-source --statistics 2>/dev/null; then
        print_success "Linting passed (critical errors only)"
    else
        print_warning "Linting found issues (non-blocking)"
    fi
else
    print_warning "flake8 not installed, skipping lint check"
fi

# Step 4: Update version in pyproject.toml
print_step "Updating version in pyproject.toml..."
CURRENT_VERSION=$(grep -E '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
echo "  Current version: $CURRENT_VERSION"
echo "  New version: $VERSION"

if [ "$CURRENT_VERSION" == "$VERSION" ]; then
    print_success "Version already set to $VERSION"
else
    sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
    rm -f pyproject.toml.bak
    print_success "Updated pyproject.toml to version $VERSION"
fi

# Step 5: Update version in __init__.py
print_step "Updating version in tinyforgeai/__init__.py..."
if [ -f "tinyforgeai/__init__.py" ]; then
    sed -i.bak "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" tinyforgeai/__init__.py
    rm -f tinyforgeai/__init__.py.bak
    print_success "Updated tinyforgeai/__init__.py"
else
    print_warning "tinyforgeai/__init__.py not found, skipping"
fi

# Step 6: Update CHANGELOG.md
print_step "Checking CHANGELOG.md..."
if grep -q "## \[Unreleased\]" CHANGELOG.md; then
    TODAY=$(date +%Y-%m-%d)
    print_warning "Remember to update CHANGELOG.md:"
    echo "  - Move [Unreleased] items to [$VERSION] - $TODAY"
    echo "  - Add new [Unreleased] section at top"
else
    print_success "CHANGELOG.md appears ready"
fi

# Step 7: Build distribution packages
print_step "Building distribution packages..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Check if build module is available
if python -c "import build" 2>/dev/null; then
    python -m build
    print_success "Built packages:"
    ls -la dist/
else
    print_warning "python-build not installed. Install with: pip install build"
    echo "  Then run: python -m build"
fi

# Step 8: Verify package
print_step "Verifying package..."
if command -v twine &> /dev/null && [ -d "dist" ]; then
    twine check dist/* && print_success "Package verification passed"
else
    print_warning "twine not installed or no dist/ folder. Install with: pip install twine"
fi

# Final summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Release Summary                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
print_success "Release preparation complete for version $VERSION"
echo ""

echo "Next steps to finalize the release:"
echo ""
echo "1. Review and commit changes:"
echo "   ${YELLOW}git add pyproject.toml tinyforgeai/__init__.py CHANGELOG.md${NC}"
echo "   ${YELLOW}git commit -m \"chore: bump version to $VERSION\"${NC}"
echo ""
echo "2. Create and push the tag:"
echo "   ${YELLOW}git tag -a v$VERSION -m \"Release v$VERSION\"${NC}"
echo "   ${YELLOW}git push origin main${NC}"
echo "   ${YELLOW}git push origin v$VERSION${NC}"
echo ""
echo "3. Upload to PyPI (if applicable):"
echo "   ${YELLOW}twine upload dist/*${NC}"
echo ""
echo "4. Create GitHub Release:"
echo "   ${YELLOW}gh release create v$VERSION dist/* --title \"v$VERSION\" --notes-file releases/notes_initial_release.md${NC}"
echo ""

print_success "Done! Review the above commands before executing."
