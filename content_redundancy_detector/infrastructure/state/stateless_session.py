"""
Stateless Session Management
Leverages external storage (Redis) for session state persistence
Following microservices stateless principles
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Session data structure"""
    session_id: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    created_at: float
    last_activity: float
    expires_at: float
    data: Dict[str, Any]
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class StatelessSessionManager:
    """
    Stateless session manager using Redis
    No server-side state, all state in external storage
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._client = None
    
    async def _get_client(self):
        """Get Redis client"""
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(self.redis_url, decode_responses=True)
                await self._client.ping()
            except ImportError:
                logger.error("redis.asyncio not available")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._client
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        ttl: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """
        Create new session
        
        Returns:
            SessionData with session_id
        """
        session_id = str(uuid.uuid4())
        now = time.time()
        expires_at = now + (ttl or self.default_ttl)
        
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now,
            last_activity=now,
            expires_at=expires_at,
            data=data or {}
        )
        
        # Store in Redis
        client = await self._get_client()
        await client.setex(
            f"session:{session_id}",
            int(expires_at - now),
            json.dumps(session.to_dict())
        )
        
        logger.info(f"Session created: {session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID"""
        client = await self._get_client()
        data = await client.get(f"session:{session_id}")
        
        if not data:
            return None
        
        try:
            session_dict = json.loads(data)
            session = SessionData(**session_dict)
            
            if session.is_expired():
                await self.delete_session(session_id)
                return None
            
            return session
        except Exception as e:
            logger.error(f"Error parsing session data: {e}")
            return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.data.update(data)
        session.last_activity = time.time()
        
        client = await self._get_client()
        ttl = int(session.expires_at - time.time())
        if ttl > 0:
            await client.setex(
                f"session:{session_id}",
                ttl,
                json.dumps(session.to_dict())
            )
            return True
        
        return False
    
    async def refresh_session(self, session_id: str, ttl: Optional[int] = None) -> bool:
        """Refresh session expiration"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        new_ttl = ttl or self.default_ttl
        session.expires_at = time.time() + new_ttl
        session.last_activity = time.time()
        
        client = await self._get_client()
        await client.setex(
            f"session:{session_id}",
            new_ttl,
            json.dumps(session.to_dict())
        )
        
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        client = await self._get_client()
        deleted = await client.delete(f"session:{session_id}")
        return deleted > 0
    
    async def cleanup_expired_sessions(self) -> int:
        """Cleanup expired sessions (optional, Redis handles TTL automatically)"""
        # Redis automatically expires keys, but we can scan for cleanup
        client = await self._get_client()
        count = 0
        
        async for key in client.scan_iter("session:*"):
            data = await client.get(key)
            if data:
                try:
                    session_dict = json.loads(data)
                    session = SessionData(**session_dict)
                    if session.is_expired():
                        await client.delete(key)
                        count += 1
                except Exception:
                    await client.delete(key)
                    count += 1
        
        return count


# Global session manager instance
_session_manager: Optional[StatelessSessionManager] = None


def get_session_manager() -> StatelessSessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = StatelessSessionManager()
    return _session_manager






