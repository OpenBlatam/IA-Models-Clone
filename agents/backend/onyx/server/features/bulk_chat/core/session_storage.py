"""
Session Storage - Persistencia de sesiones
===========================================

Sistema de almacenamiento persistente para sesiones de chat.
Soporta Redis y almacenamiento en archivo JSON.
"""

import json
import asyncio
import logging
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
import pickle

from .chat_session import ChatSession, ChatMessage, ChatState

logger = logging.getLogger(__name__)


class SessionStorage:
    """Interface para almacenamiento de sesiones."""
    
    async def save_session(self, session: ChatSession) -> bool:
        """Guardar sesión."""
        raise NotImplementedError
    
    async def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Cargar sesión."""
        raise NotImplementedError
    
    async def delete_session(self, session_id: str) -> bool:
        """Eliminar sesión."""
        raise NotImplementedError
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """Listar IDs de sesiones."""
        raise NotImplementedError


class JSONSessionStorage(SessionStorage):
    """Almacenamiento en archivo JSON."""
    
    def __init__(self, storage_path: str = "sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    def _get_session_path(self, session_id: str) -> Path:
        """Obtener ruta del archivo de sesión."""
        return self.storage_path / f"{session_id}.json"
    
    async def save_session(self, session: ChatSession) -> bool:
        """Guardar sesión en archivo JSON."""
        try:
            async with self._lock:
                session_data = {
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "state": session.state.value,
                    "is_paused": session.is_paused,
                    "pause_reason": session.pause_reason,
                    "auto_continue": session.auto_continue,
                    "max_messages": session.max_messages,
                    "auto_respond_interval": session.auto_respond_interval,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "messages": [
                        {
                            "id": msg.id,
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat(),
                            "metadata": msg.metadata,
                        }
                        for msg in session.messages
                    ],
                }
                
                file_path = self._get_session_path(session.session_id)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
                
                return True
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Cargar sesión desde archivo JSON."""
        try:
            async with self._lock:
                file_path = self._get_session_path(session_id)
                if not file_path.exists():
                    return None
                
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Reconstruir sesión
                session = ChatSession(
                    session_id=data["session_id"],
                    user_id=data.get("user_id"),
                    state=ChatState(data["state"]),
                    is_paused=data.get("is_paused", False),
                    pause_reason=data.get("pause_reason"),
                    auto_continue=data.get("auto_continue", True),
                    max_messages=data.get("max_messages", 1000),
                    auto_respond_interval=data.get("auto_respond_interval", 2.0),
                )
                
                session.created_at = datetime.fromisoformat(data["created_at"])
                session.updated_at = datetime.fromisoformat(data["updated_at"])
                
                # Reconstruir mensajes
                session.messages = [
                    ChatMessage(
                        id=msg_data["id"],
                        role=msg_data["role"],
                        content=msg_data["content"],
                        timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                        metadata=msg_data.get("metadata", {}),
                    )
                    for msg_data in data.get("messages", [])
                ]
                
                return session
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Eliminar sesión."""
        try:
            async with self._lock:
                file_path = self._get_session_path(session_id)
                if file_path.exists():
                    file_path.unlink()
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """Listar IDs de sesiones."""
        try:
            async with self._lock:
                session_ids = []
                for file_path in self.storage_path.glob("*.json"):
                    session_id = file_path.stem
                    if user_id:
                        # Cargar sesión para verificar user_id
                        session = await self.load_session(session_id)
                        if session and session.user_id == user_id:
                            session_ids.append(session_id)
                    else:
                        session_ids.append(session_id)
                return session_ids
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []


class RedisSessionStorage(SessionStorage):
    """Almacenamiento en Redis."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 86400):
        self.redis_url = redis_url
        self.ttl = ttl  # Time to live en segundos
        self._redis = None
        self._lock = asyncio.Lock()
    
    async def _get_redis(self):
        """Obtener cliente Redis (lazy initialization)."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url, decode_responses=False)
            except ImportError:
                logger.error("Redis library not installed. Install with: pip install redis")
                raise
        return self._redis
    
    async def save_session(self, session: ChatSession) -> bool:
        """Guardar sesión en Redis."""
        try:
            redis = await self._get_redis()
            async with self._lock:
                # Serializar sesión
                session_data = {
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "state": session.state.value,
                    "is_paused": session.is_paused,
                    "pause_reason": session.pause_reason,
                    "auto_continue": session.auto_continue,
                    "max_messages": session.max_messages,
                    "auto_respond_interval": session.auto_respond_interval,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "messages": [
                        {
                            "id": msg.id,
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat(),
                            "metadata": msg.metadata,
                        }
                        for msg in session.messages
                    ],
                }
                
                key = f"chat_session:{session.session_id}"
                value = json.dumps(session_data, ensure_ascii=False).encode("utf-8")
                
                await redis.setex(key, self.ttl, value)
                
                # También guardar índice por user_id
                if session.user_id:
                    await redis.sadd(f"user_sessions:{session.user_id}", session.session_id)
                    await redis.expire(f"user_sessions:{session.user_id}", self.ttl)
                
                return True
        except Exception as e:
            logger.error(f"Error saving session {session.session_id} to Redis: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Cargar sesión desde Redis."""
        try:
            redis = await self._get_redis()
            async with self._lock:
                key = f"chat_session:{session_id}"
                value = await redis.get(key)
                
                if not value:
                    return None
                
                data = json.loads(value.decode("utf-8"))
                
                # Reconstruir sesión
                session = ChatSession(
                    session_id=data["session_id"],
                    user_id=data.get("user_id"),
                    state=ChatState(data["state"]),
                    is_paused=data.get("is_paused", False),
                    pause_reason=data.get("pause_reason"),
                    auto_continue=data.get("auto_continue", True),
                    max_messages=data.get("max_messages", 1000),
                    auto_respond_interval=data.get("auto_respond_interval", 2.0),
                )
                
                session.created_at = datetime.fromisoformat(data["created_at"])
                session.updated_at = datetime.fromisoformat(data["updated_at"])
                
                # Reconstruir mensajes
                session.messages = [
                    ChatMessage(
                        id=msg_data["id"],
                        role=msg_data["role"],
                        content=msg_data["content"],
                        timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                        metadata=msg_data.get("metadata", {}),
                    )
                    for msg_data in data.get("messages", [])
                ]
                
                return session
        except Exception as e:
            logger.error(f"Error loading session {session_id} from Redis: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Eliminar sesión de Redis."""
        try:
            redis = await self._get_redis()
            async with self._lock:
                key = f"chat_session:{session_id}"
                value = await redis.get(key)
                
                if value:
                    data = json.loads(value.decode("utf-8"))
                    user_id = data.get("user_id")
                    
                    # Eliminar sesión
                    await redis.delete(key)
                    
                    # Eliminar de índice de usuario
                    if user_id:
                        await redis.srem(f"user_sessions:{user_id}", session_id)
                    
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting session {session_id} from Redis: {e}")
            return False
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """Listar IDs de sesiones."""
        try:
            redis = await self._get_redis()
            async with self._lock:
                if user_id:
                    key = f"user_sessions:{user_id}"
                    session_ids = await redis.smembers(key)
                    return [sid.decode("utf-8") if isinstance(sid, bytes) else sid for sid in session_ids]
                else:
                    # Buscar todas las sesiones (menos eficiente)
                    keys = await redis.keys("chat_session:*")
                    return [key.decode("utf-8").replace("chat_session:", "") for key in keys]
        except Exception as e:
            logger.error(f"Error listing sessions from Redis: {e}")
            return []
































