"""
TinyForgeAI Dashboard API Authentication

Simple API key and optional JWT authentication for the dashboard API.
Designed to be lightweight and easy to configure.

Usage:
    # Enable via environment variables:
    TINYFORGE_API_KEY=your-secret-key
    TINYFORGE_AUTH_ENABLED=true

    # Or disable authentication (default for development):
    TINYFORGE_AUTH_ENABLED=false
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBasic, HTTPBasicCredentials

# =============================================================================
# Configuration
# =============================================================================

# Auth settings from environment
AUTH_ENABLED = os.getenv("TINYFORGE_AUTH_ENABLED", "false").lower() == "true"
API_KEY = os.getenv("TINYFORGE_API_KEY", "")
API_KEY_NAME = "X-API-Key"

# Default credentials (for development only - override in production!)
DEFAULT_USERNAME = os.getenv("TINYFORGE_USERNAME", "admin")
DEFAULT_PASSWORD = os.getenv("TINYFORGE_PASSWORD", "tinyforge")

# =============================================================================
# Security Schemes
# =============================================================================

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
basic_auth = HTTPBasic(auto_error=False)


# =============================================================================
# Authentication Functions
# =============================================================================

def generate_api_key() -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(32)


def verify_api_key(api_key: str) -> bool:
    """Verify an API key against the configured key."""
    if not API_KEY:
        return False
    return hmac.compare_digest(api_key, API_KEY)


def verify_basic_credentials(username: str, password: str) -> bool:
    """Verify HTTP Basic credentials."""
    correct_username = hmac.compare_digest(username, DEFAULT_USERNAME)
    correct_password = hmac.compare_digest(password, DEFAULT_PASSWORD)
    return correct_username and correct_password


# =============================================================================
# Dependency Functions
# =============================================================================

async def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """Extract API key from header."""
    return api_key


async def get_basic_credentials(
    credentials: Optional[HTTPBasicCredentials] = Depends(basic_auth)
) -> Optional[HTTPBasicCredentials]:
    """Extract HTTP Basic credentials."""
    return credentials


async def verify_auth(
    api_key: Optional[str] = Depends(get_api_key),
    credentials: Optional[HTTPBasicCredentials] = Depends(get_basic_credentials),
) -> bool:
    """
    Verify authentication via API key or HTTP Basic auth.

    Returns True if:
    - Auth is disabled (development mode)
    - Valid API key is provided
    - Valid Basic credentials are provided
    """
    # If auth is disabled, allow all requests
    if not AUTH_ENABLED:
        return True

    # Try API key first
    if api_key and verify_api_key(api_key):
        return True

    # Try HTTP Basic auth
    if credentials and verify_basic_credentials(credentials.username, credentials.password):
        return True

    # No valid auth provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Basic"},
    )


async def require_auth(verified: bool = Depends(verify_auth)) -> bool:
    """
    Dependency that requires authentication.
    Use this for protected endpoints.

    Example:
        @app.get("/api/protected")
        async def protected_endpoint(auth: bool = Depends(require_auth)):
            return {"message": "You are authenticated!"}
    """
    return verified


# =============================================================================
# Optional: Simple Token-Based Auth
# =============================================================================

class SimpleTokenAuth:
    """
    Simple token-based authentication for session management.
    Tokens are stored in memory (not suitable for production with multiple workers).
    """

    def __init__(self):
        self.tokens: dict[str, dict] = {}
        self.token_lifetime = timedelta(hours=24)

    def create_token(self, username: str) -> str:
        """Create a new auth token for a user."""
        token = secrets.token_urlsafe(32)
        self.tokens[token] = {
            "username": username,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + self.token_lifetime,
        }
        return token

    def verify_token(self, token: str) -> Optional[str]:
        """Verify a token and return the username if valid."""
        if token not in self.tokens:
            return None

        token_data = self.tokens[token]
        if datetime.utcnow() > token_data["expires_at"]:
            del self.tokens[token]
            return None

        return token_data["username"]

    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        if token in self.tokens:
            del self.tokens[token]
            return True
        return False

    def cleanup_expired(self):
        """Remove expired tokens."""
        now = datetime.utcnow()
        expired = [t for t, d in self.tokens.items() if now > d["expires_at"]]
        for token in expired:
            del self.tokens[token]


# Global token auth instance
token_auth = SimpleTokenAuth()


# =============================================================================
# Auth Endpoints (to be added to main.py)
# =============================================================================

def get_auth_status() -> dict:
    """Get current authentication status and configuration."""
    return {
        "auth_enabled": AUTH_ENABLED,
        "api_key_configured": bool(API_KEY),
        "basic_auth_available": True,
        "message": "Authentication is " + ("enabled" if AUTH_ENABLED else "disabled"),
    }
