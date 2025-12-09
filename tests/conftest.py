"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest before running tests.
"""

import os
import sys

# Set test environment variables BEFORE any imports
# Use in-memory storage (disable database) for simpler, faster tests
os.environ["TINYFORGE_USE_DATABASE"] = "false"
os.environ["TINYFORGE_AUTH_ENABLED"] = "false"

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
