"""
Record Storage - Real-World Use Cases

Practical examples of how to use RecordStorage in real applications.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

from .record_storage import RecordStorage


class UserPreferencesStorage(RecordStorage):
    """Store user preferences with validation."""
    
    def save_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Save user preferences."""
        record = {
            "id": user_id,
            "preferences": preferences,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        existing = self.get(user_id)
        if existing:
            return self.update(user_id, {
                "preferences": preferences,
                "updated_at": record["updated_at"]
            })
        return self.add(record)
    
    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        record = self.get(user_id)
        if record:
            return record.get("preferences", {})
        return {}


class SessionStorage(RecordStorage):
    """Store user sessions with expiration."""
    
    def create_session(self, session_id: str, user_id: str, expires_at: str) -> bool:
        """Create a new session."""
        session = {
            "id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at,
            "active": True
        }
        return self.add(session)
    
    def get_active_session(self, session_id: str) -> Dict[str, Any]:
        """Get active session if not expired."""
        session = self.get(session_id)
        if not session:
            return None
        
        if not session.get("active", False):
            return None
        
        expires_at = datetime.fromisoformat(session["expires_at"])
        if datetime.utcnow() > expires_at:
            self.update(session_id, {"active": False})
            return None
        
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        return self.update(session_id, {"active": False})


class CacheStorage(RecordStorage):
    """Simple cache with TTL."""
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set cache value with TTL."""
        expires_at = datetime.utcnow().timestamp() + ttl_seconds
        record = {
            "id": key,
            "value": value,
            "expires_at": expires_at
        }
        
        existing = self.get(key)
        if existing:
            return self.update(key, record)
        return self.add(record)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cache value if not expired."""
        record = super().get(key)
        if not record:
            return default
        
        expires_at = record.get("expires_at", 0)
        if datetime.utcnow().timestamp() > expires_at:
            self.delete(key)
            return default
        
        return record.get("value", default)


class ConfigurationStorage(RecordStorage):
    """Application configuration storage."""
    
    def get_config(self, key: str = "default") -> Dict[str, Any]:
        """Get configuration value."""
        config = self.get(key)
        if config:
            return config.get("value", {})
        return {}
    
    def set_config(self, key: str, value: Dict[str, Any]) -> bool:
        """Set configuration value."""
        record = {
            "id": key,
            "value": value,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        existing = self.get(key)
        if existing:
            return self.update(key, {
                "value": value,
                "updated_at": record["updated_at"]
            })
        return self.add(record)


class AuditLogStorage(RecordStorage):
    """Audit log storage."""
    
    def log_event(self, event_type: str, user_id: str, details: Dict[str, Any]) -> bool:
        """Log an audit event."""
        event_id = f"{event_type}_{datetime.utcnow().timestamp()}"
        event = {
            "id": event_id,
            "type": event_type,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
        return self.add(event)
    
    def get_user_events(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get events for a specific user."""
        records = self.read()
        events = [
            r for r in records
            if isinstance(r, dict) and r.get("user_id") == user_id
        ]
        events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return events[:limit]
    
    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get events of a specific type."""
        records = self.read()
        events = [
            r for r in records
            if isinstance(r, dict) and r.get("type") == event_type
        ]
        events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return events[:limit]


class FeatureFlagsStorage(RecordStorage):
    """Feature flags storage."""
    
    def enable_feature(self, feature_name: str, enabled: bool = True) -> bool:
        """Enable or disable a feature."""
        record = {
            "id": feature_name,
            "enabled": enabled,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        existing = self.get(feature_name)
        if existing:
            return self.update(feature_name, {
                "enabled": enabled,
                "updated_at": record["updated_at"]
            })
        return self.add(record)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        feature = self.get(feature_name)
        if feature:
            return feature.get("enabled", False)
        return False


def example_user_preferences():
    """Example: User preferences."""
    storage = UserPreferencesStorage("data/user_preferences.json")
    
    storage.save_preferences("user123", {
        "theme": "dark",
        "language": "en",
        "notifications": True
    })
    
    prefs = storage.get_preferences("user123")
    print(f"User preferences: {prefs}")


def example_sessions():
    """Example: Session management."""
    storage = SessionStorage("data/sessions.json")
    
    expires = (datetime.utcnow().timestamp() + 3600)
    expires_at = datetime.fromtimestamp(expires).isoformat()
    
    storage.create_session("session123", "user123", expires_at)
    
    session = storage.get_active_session("session123")
    if session:
        print(f"Active session: {session['user_id']}")
    
    storage.invalidate_session("session123")


def example_cache():
    """Example: Caching."""
    cache = CacheStorage("data/cache.json")
    
    cache.set("api_data", {"result": "cached"}, ttl_seconds=60)
    
    data = cache.get("api_data")
    if data:
        print(f"Cached data: {data}")
    else:
        print("Cache expired or not found")


def example_config():
    """Example: Configuration."""
    config = ConfigurationStorage("data/config.json")
    
    config.set_config("database", {
        "host": "localhost",
        "port": 5432,
        "name": "mydb"
    })
    
    db_config = config.get_config("database")
    print(f"Database config: {db_config}")


def example_audit_log():
    """Example: Audit logging."""
    audit = AuditLogStorage("data/audit.json")
    
    audit.log_event("login", "user123", {"ip": "192.168.1.1"})
    audit.log_event("logout", "user123", {})
    
    events = audit.get_user_events("user123", limit=10)
    print(f"User events: {len(events)}")


def example_feature_flags():
    """Example: Feature flags."""
    flags = FeatureFlagsStorage("data/feature_flags.json")
    
    flags.enable_feature("new_ui", True)
    flags.enable_feature("beta_features", False)
    
    if flags.is_feature_enabled("new_ui"):
        print("New UI is enabled")
    
    if not flags.is_feature_enabled("beta_features"):
        print("Beta features are disabled")


if __name__ == "__main__":
    print("Real-World Use Cases Examples")
    print("=" * 50)
    
    example_user_preferences()
    example_sessions()
    example_cache()
    example_config()
    example_audit_log()
    example_feature_flags()
    
    print("\n✅ All examples completed!")


