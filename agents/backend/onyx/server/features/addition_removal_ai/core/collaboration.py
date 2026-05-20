"""
Collaboration - Sistema de colaboración en tiempo real
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class CollaborationSession:
    """Sesión de colaboración"""
    id: str
    content_id: str
    participants: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    locked: bool = False
    locked_by: Optional[str] = None


@dataclass
class Comment:
    """Comentario en contenido"""
    id: str
    content_id: str
    user_id: str
    text: str
    position: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    replies: List['Comment'] = field(default_factory=list)


class CollaborationManager:
    """Gestor de colaboración"""

    def __init__(self):
        """Inicializar gestor de colaboración"""
        self.sessions: Dict[str, CollaborationSession] = {}
        self.comments: Dict[str, List[Comment]] = {}
        self.active_users: Dict[str, Dict[str, Any]] = {}

    def create_session(self, content_id: str, user_id: str) -> CollaborationSession:
        """
        Crear sesión de colaboración.

        Args:
            content_id: ID del contenido
            user_id: ID del usuario creador

        Returns:
            Sesión creada
        """
        session_id = str(uuid.uuid4())
        session = CollaborationSession(
            id=session_id,
            content_id=content_id,
            participants=[user_id]
        )
        self.sessions[session_id] = session
        logger.info(f"Sesión de colaboración creada: {session_id}")
        return session

    def join_session(self, session_id: str, user_id: str) -> bool:
        """
        Unirse a una sesión.

        Args:
            session_id: ID de la sesión
            user_id: ID del usuario

        Returns:
            True si se unió exitosamente
        """
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        if user_id not in session.participants:
            session.participants.append(user_id)
            session.last_activity = datetime.utcnow()
            logger.info(f"Usuario {user_id} se unió a sesión {session_id}")
        
        return True

    def leave_session(self, session_id: str, user_id: str):
        """
        Salir de una sesión.

        Args:
            session_id: ID de la sesión
            user_id: ID del usuario
        """
        session = self.sessions.get(session_id)
        if session and user_id in session.participants:
            session.participants.remove(user_id)
            session.last_activity = datetime.utcnow()

    def lock_content(self, content_id: str, user_id: str) -> bool:
        """
        Bloquear contenido para edición.

        Args:
            content_id: ID del contenido
            user_id: ID del usuario

        Returns:
            True si se bloqueó exitosamente
        """
        # Buscar sesión por content_id
        session = next((s for s in self.sessions.values() if s.content_id == content_id), None)
        if not session:
            return False
        
        if session.locked and session.locked_by != user_id:
            return False
        
        session.locked = True
        session.locked_by = user_id
        session.last_activity = datetime.utcnow()
        return True

    def unlock_content(self, content_id: str, user_id: str):
        """
        Desbloquear contenido.

        Args:
            content_id: ID del contenido
            user_id: ID del usuario
        """
        session = next((s for s in self.sessions.values() if s.content_id == content_id), None)
        if session and session.locked_by == user_id:
            session.locked = False
            session.locked_by = None
            session.last_activity = datetime.utcnow()

    def add_comment(
        self,
        content_id: str,
        user_id: str,
        text: str,
        position: Optional[int] = None,
        parent_comment_id: Optional[str] = None
    ) -> Comment:
        """
        Agregar comentario.

        Args:
            content_id: ID del contenido
            user_id: ID del usuario
            text: Texto del comentario
            position: Posición en el contenido
            parent_comment_id: ID del comentario padre (para respuestas)

        Returns:
            Comentario creado
        """
        comment_id = str(uuid.uuid4())
        comment = Comment(
            id=comment_id,
            content_id=content_id,
            user_id=user_id,
            text=text,
            position=position
        )
        
        if content_id not in self.comments:
            self.comments[content_id] = []
        
        if parent_comment_id:
            # Buscar comentario padre y agregar como respuesta
            for c in self.comments[content_id]:
                if c.id == parent_comment_id:
                    c.replies.append(comment)
                    break
        else:
            self.comments[content_id].append(comment)
        
        logger.info(f"Comentario agregado: {comment_id}")
        return comment

    def get_comments(self, content_id: str) -> List[Dict[str, Any]]:
        """
        Obtener comentarios de un contenido.

        Args:
            content_id: ID del contenido

        Returns:
            Lista de comentarios
        """
        comments = self.comments.get(content_id, [])
        return [self._comment_to_dict(c) for c in comments]

    def _comment_to_dict(self, comment: Comment) -> Dict[str, Any]:
        """Convertir comentario a diccionario"""
        return {
            "id": comment.id,
            "content_id": comment.content_id,
            "user_id": comment.user_id,
            "text": comment.text,
            "position": comment.position,
            "created_at": comment.created_at.isoformat(),
            "resolved": comment.resolved,
            "replies": [self._comment_to_dict(r) for r in comment.replies]
        }

    def resolve_comment(self, comment_id: str, content_id: str):
        """
        Marcar comentario como resuelto.

        Args:
            comment_id: ID del comentario
            content_id: ID del contenido
        """
        comments = self.comments.get(content_id, [])
        for comment in comments:
            if comment.id == comment_id:
                comment.resolved = True
                break






