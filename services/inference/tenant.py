"""
Tenant Management for Multi-Tenant Inference Service

Handles tenant creation, configuration, and lifecycle management.
Supports both in-memory and SQLite persistence.
"""

import hashlib
import json
import logging
import secrets
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class TenantStatus(str, Enum):
    """Tenant status values."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"
    PENDING = "pending"


class TenantTier(str, Enum):
    """Tenant tier for quota management."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuota:
    """Quota limits for a tenant."""
    requests_per_minute: int = 60
    requests_per_day: int = 1000
    max_input_tokens: int = 4096
    max_output_tokens: int = 2048
    max_models: int = 1
    max_concurrent_requests: int = 5

    @classmethod
    def for_tier(cls, tier: TenantTier) -> "TenantQuota":
        """Get default quota for a tier."""
        quotas = {
            TenantTier.FREE: cls(
                requests_per_minute=10,
                requests_per_day=100,
                max_input_tokens=1024,
                max_output_tokens=512,
                max_models=1,
                max_concurrent_requests=1,
            ),
            TenantTier.STARTER: cls(
                requests_per_minute=30,
                requests_per_day=1000,
                max_input_tokens=2048,
                max_output_tokens=1024,
                max_models=3,
                max_concurrent_requests=3,
            ),
            TenantTier.PROFESSIONAL: cls(
                requests_per_minute=100,
                requests_per_day=10000,
                max_input_tokens=4096,
                max_output_tokens=2048,
                max_models=10,
                max_concurrent_requests=10,
            ),
            TenantTier.ENTERPRISE: cls(
                requests_per_minute=1000,
                requests_per_day=100000,
                max_input_tokens=8192,
                max_output_tokens=4096,
                max_models=100,
                max_concurrent_requests=50,
            ),
        }
        return quotas.get(tier, cls())


@dataclass
class TenantConfig:
    """Configuration settings for a tenant."""
    default_model: Optional[str] = None
    allowed_models: List[str] = field(default_factory=list)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    webhook_url: Optional[str] = None
    callback_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class TenantUsage:
    """Usage statistics for a tenant."""
    total_requests: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    requests_today: int = 0
    requests_this_minute: int = 0
    last_request_at: Optional[str] = None
    last_minute_reset: Optional[str] = None
    last_day_reset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TenantUsage":
        return cls(**data)


@dataclass
class Tenant:
    """Represents a tenant in the multi-tenant inference system."""
    id: str
    name: str
    api_key_hash: str
    status: TenantStatus = TenantStatus.ACTIVE
    tier: TenantTier = TenantTier.FREE
    quota: TenantQuota = field(default_factory=TenantQuota)
    config: TenantConfig = field(default_factory=TenantConfig)
    usage: TenantUsage = field(default_factory=TenantUsage)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "api_key_hash": self.api_key_hash,
            "status": self.status.value,
            "tier": self.tier.value,
            "quota": asdict(self.quota),
            "config": asdict(self.config),
            "usage": self.usage.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tenant":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            api_key_hash=data["api_key_hash"],
            status=TenantStatus(data.get("status", "active")),
            tier=TenantTier(data.get("tier", "free")),
            quota=TenantQuota(**data.get("quota", {})),
            config=TenantConfig(**data.get("config", {})),
            usage=TenantUsage.from_dict(data.get("usage", {})),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )

    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE

    def can_make_request(self) -> tuple[bool, Optional[str]]:
        """Check if tenant can make a request based on quotas."""
        if not self.is_active():
            return False, f"Tenant is {self.status.value}"

        now = datetime.utcnow()

        # Check per-minute limit
        if self.usage.last_minute_reset:
            last_reset = datetime.fromisoformat(self.usage.last_minute_reset)
            if (now - last_reset).total_seconds() < 60:
                if self.usage.requests_this_minute >= self.quota.requests_per_minute:
                    return False, "Rate limit exceeded (per minute)"

        # Check per-day limit
        if self.usage.last_day_reset:
            last_reset = datetime.fromisoformat(self.usage.last_day_reset)
            if (now - last_reset).total_seconds() < 86400:
                if self.usage.requests_today >= self.quota.requests_per_day:
                    return False, "Rate limit exceeded (per day)"

        return True, None

    def record_request(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Record a request for usage tracking."""
        now = datetime.utcnow()

        # Reset minute counter if needed
        if self.usage.last_minute_reset:
            last_reset = datetime.fromisoformat(self.usage.last_minute_reset)
            if (now - last_reset).total_seconds() >= 60:
                self.usage.requests_this_minute = 0
                self.usage.last_minute_reset = now.isoformat()
        else:
            self.usage.last_minute_reset = now.isoformat()

        # Reset day counter if needed
        if self.usage.last_day_reset:
            last_reset = datetime.fromisoformat(self.usage.last_day_reset)
            if (now - last_reset).total_seconds() >= 86400:
                self.usage.requests_today = 0
                self.usage.last_day_reset = now.isoformat()
        else:
            self.usage.last_day_reset = now.isoformat()

        # Update counters
        self.usage.total_requests += 1
        self.usage.requests_this_minute += 1
        self.usage.requests_today += 1
        self.usage.total_tokens_in += tokens_in
        self.usage.total_tokens_out += tokens_out
        self.usage.last_request_at = now.isoformat()


class TenantManager:
    """
    Manages tenants with optional SQLite persistence.

    Thread-safe for concurrent access.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize tenant manager.

        Args:
            db_path: Path to SQLite database. If None, uses in-memory storage.
        """
        self.db_path = db_path
        self._tenants: Dict[str, Tenant] = {}
        self._api_key_index: Dict[str, str] = {}  # hash -> tenant_id
        self._lock = Lock()

        if db_path:
            self._init_db()
            self._load_from_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                api_key_hash TEXT NOT NULL UNIQUE,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_key_hash ON tenants(api_key_hash)
        """)

        conn.commit()
        conn.close()

    def _load_from_db(self) -> None:
        """Load tenants from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, data FROM tenants")
        for row in cursor.fetchall():
            tenant_id, data = row
            tenant = Tenant.from_dict(json.loads(data))
            self._tenants[tenant_id] = tenant
            self._api_key_index[tenant.api_key_hash] = tenant_id

        conn.close()
        logger.info(f"Loaded {len(self._tenants)} tenants from database")

    def _save_to_db(self, tenant: Tenant) -> None:
        """Save tenant to database."""
        if not self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO tenants (id, name, api_key_hash, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            tenant.id,
            tenant.name,
            tenant.api_key_hash,
            json.dumps(tenant.to_dict()),
            tenant.created_at,
            tenant.updated_at,
        ))

        conn.commit()
        conn.close()

    def _delete_from_db(self, tenant_id: str) -> None:
        """Delete tenant from database."""
        if not self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tenants WHERE id = ?", (tenant_id,))
        conn.commit()
        conn.close()

    @staticmethod
    def generate_api_key() -> str:
        """Generate a new API key."""
        return f"tf_{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_tenant(
        self,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[Tenant, str]:
        """
        Create a new tenant.

        Args:
            name: Tenant name.
            tier: Tenant tier for quota limits.
            metadata: Optional metadata.

        Returns:
            Tuple of (tenant, api_key). Store the API key securely!
        """
        api_key = self.generate_api_key()
        api_key_hash = self.hash_api_key(api_key)

        tenant_id = hashlib.md5(f"{name}:{api_key_hash[:8]}:{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]

        tenant = Tenant(
            id=tenant_id,
            name=name,
            api_key_hash=api_key_hash,
            tier=tier,
            quota=TenantQuota.for_tier(tier),
            metadata=metadata or {},
        )

        with self._lock:
            self._tenants[tenant_id] = tenant
            self._api_key_index[api_key_hash] = tenant_id
            self._save_to_db(tenant)

        logger.info(f"Created tenant: {tenant_id} ({name})")
        return tenant, api_key

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)

    def get_tenant_by_api_key(self, api_key: str) -> Optional[Tenant]:
        """Get tenant by API key."""
        api_key_hash = self.hash_api_key(api_key)
        tenant_id = self._api_key_index.get(api_key_hash)
        if tenant_id:
            return self._tenants.get(tenant_id)
        return None

    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
    ) -> List[Tenant]:
        """List tenants with optional filtering."""
        tenants = list(self._tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status]
        if tier:
            tenants = [t for t in tenants if t.tier == tier]

        return tenants

    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
        config: Optional[TenantConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tenant]:
        """Update tenant properties."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        with self._lock:
            if name is not None:
                tenant.name = name
            if status is not None:
                tenant.status = status
            if tier is not None:
                tenant.tier = tier
                tenant.quota = TenantQuota.for_tier(tier)
            if config is not None:
                tenant.config = config
            if metadata is not None:
                tenant.metadata.update(metadata)

            tenant.updated_at = datetime.utcnow().isoformat()
            self._save_to_db(tenant)

        return tenant

    def rotate_api_key(self, tenant_id: str) -> Optional[str]:
        """
        Rotate API key for a tenant.

        Returns the new API key or None if tenant not found.
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        new_api_key = self.generate_api_key()
        new_hash = self.hash_api_key(new_api_key)
        old_hash = tenant.api_key_hash

        with self._lock:
            # Update tenant
            tenant.api_key_hash = new_hash
            tenant.updated_at = datetime.utcnow().isoformat()

            # Update index
            del self._api_key_index[old_hash]
            self._api_key_index[new_hash] = tenant_id

            self._save_to_db(tenant)

        logger.info(f"Rotated API key for tenant: {tenant_id}")
        return new_api_key

    def delete_tenant(self, tenant_id: str, hard_delete: bool = False) -> bool:
        """
        Delete a tenant.

        Args:
            tenant_id: Tenant ID.
            hard_delete: If True, permanently remove. Otherwise, mark as deleted.

        Returns:
            True if tenant was deleted.
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        with self._lock:
            if hard_delete:
                del self._tenants[tenant_id]
                del self._api_key_index[tenant.api_key_hash]
                self._delete_from_db(tenant_id)
            else:
                tenant.status = TenantStatus.DELETED
                tenant.updated_at = datetime.utcnow().isoformat()
                self._save_to_db(tenant)

        logger.info(f"Deleted tenant: {tenant_id} (hard={hard_delete})")
        return True

    def suspend_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Suspend a tenant."""
        return self.update_tenant(tenant_id, status=TenantStatus.SUSPENDED)

    def activate_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Activate a suspended tenant."""
        return self.update_tenant(tenant_id, status=TenantStatus.ACTIVE)

    def record_usage(
        self,
        tenant_id: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> bool:
        """Record usage for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        with self._lock:
            tenant.record_request(tokens_in, tokens_out)
            self._save_to_db(tenant)

        return True

    def get_usage_stats(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        return {
            "tenant_id": tenant_id,
            "usage": tenant.usage.to_dict(),
            "quota": asdict(tenant.quota),
            "tier": tenant.tier.value,
        }
