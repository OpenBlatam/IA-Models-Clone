"""
Record Storage Integration Examples

Shows how to integrate RecordStorage with other parts of the codebase.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .record_storage import RecordStorage

logger = logging.getLogger(__name__)


class UserStorage(RecordStorage):
    """Example: User-specific storage extending RecordStorage."""
    
    def __init__(self, user_id: str, base_path: str = "data/users"):
        file_path = Path(base_path) / f"{user_id}.json"
        super().__init__(str(file_path))
        self.user_id = user_id
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences."""
        record = self.get("preferences")
        return record if record else {}
    
    def update_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        if not self.get("preferences"):
            return self.add({"id": "preferences", **preferences})
        return self.update("preferences", preferences)


class CacheStorage(RecordStorage):
    """Example: Cache storage with TTL support."""
    
    def __init__(self, file_path: str, ttl_seconds: int = 3600):
        super().__init__(file_path)
        self.ttl_seconds = ttl_seconds
    
    def set(self, key: str, value: Any) -> bool:
        """Set a cache value."""
        record = {
            "id": key,
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        existing = self.get(key)
        if existing:
            return self.update(key, record)
        return self.add(record)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a cache value if not expired."""
        record = super().get(key)
        if not record:
            return default
        
        timestamp_str = record.get("timestamp")
        if not timestamp_str:
            return default
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            age = (datetime.utcnow() - timestamp).total_seconds()
            
            if age > self.ttl_seconds:
                self.delete(key)
                return default
            
            return record.get("value", default)
        except (ValueError, TypeError):
            return default
    
    def clear_expired(self) -> int:
        """Clear all expired cache entries."""
        records = self.read()
        now = datetime.utcnow()
        expired = []
        
        for record in records:
            timestamp_str = record.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    age = (now - timestamp).total_seconds()
                    if age > self.ttl_seconds:
                        expired.append(record.get("id"))
                except (ValueError, TypeError):
                    continue
        
        for key in expired:
            self.delete(key)
        
        return len(expired)


class ConfigStorage(RecordStorage):
    """Example: Configuration storage with validation."""
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._ensure_default_config()
    
    def _ensure_default_config(self):
        """Ensure default configuration exists."""
        if not self.get("default"):
            self.add({
                "id": "default",
                "theme": "light",
                "language": "en",
                "notifications": True
            })
    
    def get_config(self, key: str = "default") -> Dict[str, Any]:
        """Get configuration by key."""
        config = self.get(key)
        if not config:
            default = self.get("default")
            return default.copy() if default else {}
        return config
    
    def update_config(self, key: str, updates: Dict[str, Any]) -> bool:
        """Update configuration with validation."""
        valid_keys = {"theme", "language", "notifications", "timezone"}
        updates = {k: v for k, v in updates.items() if k in valid_keys}
        
        if not updates:
            logger.warning("No valid configuration keys to update")
            return False
        
        config = self.get(key)
        if not config:
            return self.add({"id": key, **updates})
        
        return self.update(key, updates)


class AuditLogStorage(RecordStorage):
    """Example: Audit log storage with automatic timestamps."""
    
    def log_event(self, event_type: str, details: Dict[str, Any]) -> bool:
        """Log an audit event."""
        event_id = f"{event_type}_{datetime.utcnow().timestamp()}"
        event = {
            "id": event_id,
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
        return self.add(event)
    
    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events of a specific type."""
        records = self.read()
        events = [
            r for r in records
            if isinstance(r, dict) and r.get("type") == event_type
        ]
        events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return events[:limit]
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most recent events."""
        records = self.read()
        events = [r for r in records if isinstance(r, dict)]
        events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return events[:limit]


def example_user_storage():
    """Example: Using UserStorage."""
    storage = UserStorage("user123")
    
    storage.update_user_preferences({
        "theme": "dark",
        "language": "es",
        "notifications": False
    })
    
    prefs = storage.get_user_preferences()
    print(f"User preferences: {prefs}")


def example_cache_storage():
    """Example: Using CacheStorage."""
    cache = CacheStorage("data/cache.json", ttl_seconds=60)
    
    cache.set("api_response", {"data": "cached"})
    value = cache.get("api_response")
    print(f"Cached value: {value}")
    
    expired = cache.clear_expired()
    print(f"Cleared {expired} expired entries")


def example_config_storage():
    """Example: Using ConfigStorage."""
    config = ConfigStorage("data/config.json")
    
    config.update_config("default", {
        "theme": "dark",
        "language": "en"
    })
    
    settings = config.get_config()
    print(f"Configuration: {settings}")


def example_audit_log():
    """Example: Using AuditLogStorage."""
    audit = AuditLogStorage("data/audit.json")
    
    audit.log_event("login", {"user_id": "123", "ip": "192.168.1.1"})
    audit.log_event("logout", {"user_id": "123"})
    
    events = audit.get_events_by_type("login", limit=10)
    print(f"Login events: {len(events)}")


if __name__ == "__main__":
    print("Record Storage Integration Examples")
    print("=" * 50)
    
    example_user_storage()
    example_cache_storage()
    example_config_storage()
    example_audit_log()
    
    print("\n✅ All integration examples completed!")


