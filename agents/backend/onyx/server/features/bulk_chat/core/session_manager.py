"""
Session Manager - Gestor Avanzado de Sesiones
=============================================

Sistema avanzado de gestión de sesiones con tracking, análisis y optimización.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Estado de sesión."""
    ACTIVE = "active"
    IDLE = "idle"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"
    TERMINATED = "terminated"


@dataclass
class SessionMetrics:
    """Métricas de sesión."""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    total_messages: int = 0
    total_responses: int = 0
    avg_response_time: float = 0.0
    total_duration: float = 0.0
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionAnalytics:
    """Analíticas de sesión."""
    session_id: str
    peak_concurrent: int = 0
    avg_session_duration: float = 0.0
    user_engagement_score: float = 0.0
    completion_rate: float = 0.0
    retention_rate: float = 0.0


class SessionManager:
    """Gestor avanzado de sesiones."""
    
    def __init__(self, max_sessions: int = 100000):
        self.max_sessions = max_sessions
        self.sessions: Dict[str, SessionMetrics] = {}
        self.session_analytics: Dict[str, SessionAnalytics] = {}
        self.session_history: deque = deque(maxlen=max_sessions)
        self._lock = asyncio.Lock()
    
    def create_session(
        self,
        session_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear sesión."""
        session = SessionMetrics(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata=metadata or {},
        )
        
        async def save_session():
            async with self._lock:
                self.sessions[session_id] = session
                self.session_history.append(session_id)
        
        asyncio.create_task(save_session())
        
        logger.info(f"Created session: {session_id} for user {user_id}")
        return session_id
    
    def update_session_activity(
        self,
        session_id: str,
        message_count: int = 0,
        response_time: Optional[float] = None,
    ):
        """Actualizar actividad de sesión."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        session.last_activity = datetime.now()
        session.total_messages += message_count
        
        if response_time is not None:
            # Actualizar tiempo promedio de respuesta
            if session.total_responses > 0:
                session.avg_response_time = (
                    (session.avg_response_time * session.total_responses + response_time)
                    / (session.total_responses + 1)
                )
            else:
                session.avg_response_time = response_time
            session.total_responses += 1
    
    def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
    ):
        """Actualizar estado de sesión."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        session.status = status
        
        if status in [SessionStatus.COMPLETED, SessionStatus.TERMINATED, SessionStatus.EXPIRED]:
            # Calcular duración total
            session.total_duration = (datetime.now() - session.created_at).total_seconds()
            
            # Generar analíticas
            asyncio.create_task(self._generate_analytics(session_id))
    
    async def _generate_analytics(self, session_id: str):
        """Generar analíticas de sesión."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        async with self._lock:
            analytics = SessionAnalytics(session_id=session_id)
            
            analytics.avg_session_duration = session.total_duration
            analytics.user_engagement_score = min(
                (session.total_messages / max(session.total_duration / 60, 1)) * 10,
                100.0
            )
            
            if session.status == SessionStatus.COMPLETED:
                analytics.completion_rate = 1.0
            else:
                analytics.completion_rate = 0.0
            
            self.session_analytics[session_id] = analytics
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtener sesión."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "total_messages": session.total_messages,
            "total_responses": session.total_responses,
            "avg_response_time": session.avg_response_time,
            "total_duration": session.total_duration,
            "status": session.status.value,
            "metadata": session.metadata,
        }
    
    def get_session_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtener analíticas de sesión."""
        analytics = self.session_analytics.get(session_id)
        if not analytics:
            return None
        
        return {
            "session_id": analytics.session_id,
            "peak_concurrent": analytics.peak_concurrent,
            "avg_session_duration": analytics.avg_session_duration,
            "user_engagement_score": analytics.user_engagement_score,
            "completion_rate": analytics.completion_rate,
            "retention_rate": analytics.retention_rate,
        }
    
    def get_active_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener sesiones activas."""
        active = [
            self.get_session(sid)
            for sid, session in self.sessions.items()
            if session.status == SessionStatus.ACTIVE
        ]
        
        active.sort(key=lambda s: s["last_activity"] if s else "", reverse=True)
        return [s for s in active if s][:limit]
    
    def cleanup_expired_sessions(self, max_idle_seconds: int = 3600):
        """Limpiar sesiones expiradas."""
        now = datetime.now()
        expired = []
        
        for session_id, session in self.sessions.items():
            idle_time = (now - session.last_activity).total_seconds()
            if idle_time > max_idle_seconds and session.status == SessionStatus.ACTIVE:
                session.status = SessionStatus.EXPIRED
                expired.append(session_id)
        
        logger.info(f"Cleaned up {len(expired)} expired sessions")
        return expired
    
    def get_session_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_status: Dict[str, int] = defaultdict(int)
        total_duration = 0.0
        total_messages = 0
        
        for session in self.sessions.values():
            by_status[session.status.value] += 1
            total_duration += session.total_duration
            total_messages += session.total_messages
        
        return {
            "total_sessions": len(self.sessions),
            "sessions_by_status": dict(by_status),
            "active_sessions": len([s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]),
            "total_messages": total_messages,
            "avg_session_duration": total_duration / len(self.sessions) if self.sessions else 0.0,
            "total_analytics": len(self.session_analytics),
        }














