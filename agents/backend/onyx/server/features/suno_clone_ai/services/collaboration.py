"""
Sistema de Colaboración en Tiempo Real

Proporciona:
- Colaboración en tiempo real entre usuarios
- Edición colaborativa de canciones
- Chat en tiempo real
- Compartir proyectos
- Control de versiones colaborativo
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class CollaborationSession:
    """Sesión de colaboración"""
    session_id: str
    project_id: str
    owner_id: str
    participants: Set[str] = field(default_factory=set)
    permissions: Dict[str, str] = field(default_factory=dict)  # user_id -> permission
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True


@dataclass
class CollaborationEvent:
    """Evento de colaboración"""
    event_type: str
    user_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class CollaborationService:
    """Servicio de colaboración en tiempo real"""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> session_ids
        self.event_history: Dict[str, List[CollaborationEvent]] = defaultdict(list)
        self.websocket_connections: Dict[str, Set[str]] = defaultdict(set)  # session_id -> user_ids
        logger.info("CollaborationService initialized")
    
    def create_session(
        self,
        session_id: str,
        project_id: str,
        owner_id: str
    ) -> CollaborationSession:
        """
        Crea una sesión de colaboración
        
        Args:
            session_id: ID único de la sesión
            project_id: ID del proyecto
            owner_id: ID del propietario
        
        Returns:
            CollaborationSession
        """
        session = CollaborationSession(
            session_id=session_id,
            project_id=project_id,
            owner_id=owner_id
        )
        session.participants.add(owner_id)
        session.permissions[owner_id] = "owner"
        
        self.sessions[session_id] = session
        self.user_sessions[owner_id].add(session_id)
        
        logger.info(f"Collaboration session created: {session_id}")
        return session
    
    def join_session(
        self,
        session_id: str,
        user_id: str,
        permission: str = "editor"
    ) -> bool:
        """
        Une un usuario a una sesión
        
        Args:
            session_id: ID de la sesión
            user_id: ID del usuario
            permission: Permiso (viewer, editor, owner)
        
        Returns:
            True si se unió exitosamente
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if not session.active:
            return False
        
        session.participants.add(user_id)
        session.permissions[user_id] = permission
        self.user_sessions[user_id].add(session_id)
        
        # Registrar evento
        self._add_event(session_id, "user_joined", user_id, {"permission": permission})
        
        logger.info(f"User {user_id} joined session {session_id}")
        return True
    
    def leave_session(self, session_id: str, user_id: str):
        """Saca un usuario de una sesión"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session.participants.discard(user_id)
        session.permissions.pop(user_id, None)
        self.user_sessions[user_id].discard(session_id)
        
        # Registrar evento
        self._add_event(session_id, "user_left", user_id, {})
        
        logger.info(f"User {user_id} left session {session_id}")
    
    def add_event(
        self,
        session_id: str,
        user_id: str,
        event_type: str,
        data: Dict[str, Any]
    ):
        """
        Agrega un evento de colaboración
        
        Args:
            session_id: ID de la sesión
            user_id: ID del usuario
            event_type: Tipo de evento
            data: Datos del evento
        """
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        if user_id not in session.participants:
            return
        
        self._add_event(session_id, event_type, user_id, data)
    
    def _add_event(
        self,
        session_id: str,
        event_type: str,
        user_id: str,
        data: Dict[str, Any]
    ):
        """Agrega evento internamente"""
        event = CollaborationEvent(
            event_type=event_type,
            user_id=user_id,
            data=data
        )
        self.event_history[session_id].append(event)
        
        # Mantener solo últimos 1000 eventos por sesión
        if len(self.event_history[session_id]) > 1000:
            self.event_history[session_id] = self.event_history[session_id][-1000:]
    
    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Obtiene una sesión"""
        return self.sessions.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[CollaborationSession]:
        """Obtiene sesiones de un usuario"""
        session_ids = self.user_sessions.get(user_id, set())
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    def get_events(
        self,
        session_id: str,
        since: Optional[datetime] = None
    ) -> List[CollaborationEvent]:
        """
        Obtiene eventos de una sesión
        
        Args:
            session_id: ID de la sesión
            since: Filtrar eventos desde esta fecha
        
        Returns:
            Lista de eventos
        """
        events = self.event_history.get(session_id, [])
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events
    
    def broadcast_to_session(
        self,
        session_id: str,
        event: Dict[str, Any],
        exclude_user: Optional[str] = None
    ):
        """
        Transmite un evento a todos los participantes de una sesión
        
        Args:
            session_id: ID de la sesión
            event: Evento a transmitir
            exclude_user: Usuario a excluir (opcional)
        """
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        participants = session.participants.copy()
        
        if exclude_user:
            participants.discard(exclude_user)
        
        # En producción, esto enviaría a través de WebSocket
        logger.info(f"Broadcasting to {len(participants)} participants in session {session_id}")
    
    def update_permission(
        self,
        session_id: str,
        owner_id: str,
        user_id: str,
        permission: str
    ) -> bool:
        """
        Actualiza permisos de un usuario
        
        Args:
            session_id: ID de la sesión
            owner_id: ID del propietario
            user_id: ID del usuario
            permission: Nuevo permiso
        
        Returns:
            True si se actualizó exitosamente
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if session.owner_id != owner_id:
            return False
        
        if user_id not in session.participants:
            return False
        
        session.permissions[user_id] = permission
        
        # Registrar evento
        self._add_event(session_id, "permission_updated", owner_id, {
            "user_id": user_id,
            "permission": permission
        })
        
        logger.info(f"Permission updated for user {user_id} in session {session_id}")
        return True


# Instancia global
_collaboration_service: Optional[CollaborationService] = None


def get_collaboration_service() -> CollaborationService:
    """Obtiene la instancia global del servicio de colaboración"""
    global _collaboration_service
    if _collaboration_service is None:
        _collaboration_service = CollaborationService()
    return _collaboration_service

