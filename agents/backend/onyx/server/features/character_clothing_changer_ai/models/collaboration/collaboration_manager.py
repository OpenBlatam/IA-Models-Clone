"""
Collaboration System
===================
Sistema de colaboración y compartición de resultados
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid


class SharePermission(Enum):
    """Permisos de compartición"""
    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    FULL = "full"


@dataclass
class ShareLink:
    """Link de compartición"""
    id: str
    result_id: str
    permission: SharePermission
    expires_at: Optional[float]
    created_at: float
    access_count: int
    password: Optional[str] = None
    allowed_users: Optional[List[str]] = None


@dataclass
class Comment:
    """Comentario en resultado compartido"""
    id: str
    result_id: str
    user_id: str
    user_name: str
    text: str
    created_at: float
    replies: List['Comment'] = None
    
    def __post_init__(self):
        if self.replies is None:
            self.replies = []


@dataclass
class CollaborationSession:
    """Sesión de colaboración"""
    id: str
    result_id: str
    owner_id: str
    participants: Set[str]
    created_at: float
    active: bool = True


class CollaborationManager:
    """
    Gestor de colaboración y compartición
    """
    
    def __init__(self):
        self.share_links: Dict[str, ShareLink] = {}
        self.comments: Dict[str, List[Comment]] = {}
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}
        self.result_access: Dict[str, Set[str]] = {}  # result_id -> user_ids
    
    def create_share_link(
        self,
        result_id: str,
        permission: SharePermission = SharePermission.VIEW,
        expires_in_hours: Optional[int] = None,
        password: Optional[str] = None,
        allowed_users: Optional[List[str]] = None
    ) -> ShareLink:
        """
        Crear link de compartición
        
        Args:
            result_id: ID del resultado
            permission: Permiso de acceso
            expires_in_hours: Horas hasta expiración
            password: Contraseña opcional
            allowed_users: Lista de usuarios permitidos
        """
        link_id = hashlib.sha256(
            f"{result_id}{time.time()}{uuid.uuid4()}".encode()
        ).hexdigest()[:16]
        
        expires_at = None
        if expires_in_hours:
            expires_at = time.time() + (expires_in_hours * 3600)
        
        share_link = ShareLink(
            id=link_id,
            result_id=result_id,
            permission=permission,
            expires_at=expires_at,
            created_at=time.time(),
            access_count=0,
            password=password,
            allowed_users=allowed_users
        )
        
        self.share_links[link_id] = share_link
        return share_link
    
    def access_share_link(
        self,
        link_id: str,
        password: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Acceder a link compartido
        
        Args:
            link_id: ID del link
            password: Contraseña si es requerida
            user_id: ID del usuario accediendo
        """
        if link_id not in self.share_links:
            return None
        
        share_link = self.share_links[link_id]
        
        # Verificar expiración
        if share_link.expires_at and time.time() > share_link.expires_at:
            return None
        
        # Verificar contraseña
        if share_link.password and share_link.password != password:
            return None
        
        # Verificar usuarios permitidos
        if share_link.allowed_users and user_id not in share_link.allowed_users:
            return None
        
        # Incrementar contador
        share_link.access_count += 1
        
        # Registrar acceso
        if share_link.result_id not in self.result_access:
            self.result_access[share_link.result_id] = set()
        if user_id:
            self.result_access[share_link.result_id].add(user_id)
        
        return {
            'result_id': share_link.result_id,
            'permission': share_link.permission.value,
            'access_count': share_link.access_count
        }
    
    def add_comment(
        self,
        result_id: str,
        user_id: str,
        user_name: str,
        text: str,
        parent_comment_id: Optional[str] = None
    ) -> Comment:
        """
        Agregar comentario
        
        Args:
            result_id: ID del resultado
            user_id: ID del usuario
            user_name: Nombre del usuario
            text: Texto del comentario
            parent_comment_id: ID del comentario padre (si es respuesta)
        """
        comment = Comment(
            id=str(uuid.uuid4()),
            result_id=result_id,
            user_id=user_id,
            user_name=user_name,
            text=text,
            created_at=time.time()
        )
        
        if result_id not in self.comments:
            self.comments[result_id] = []
        
        if parent_comment_id:
            # Buscar comentario padre y agregar como respuesta
            for parent in self.comments[result_id]:
                if parent.id == parent_comment_id:
                    parent.replies.append(comment)
                    return comment
        
        # Comentario principal
        self.comments[result_id].append(comment)
        return comment
    
    def get_comments(self, result_id: str) -> List[Dict]:
        """Obtener comentarios de un resultado"""
        if result_id not in self.comments:
            return []
        
        def comment_to_dict(comment: Comment) -> Dict:
            return {
                'id': comment.id,
                'user_id': comment.user_id,
                'user_name': comment.user_name,
                'text': comment.text,
                'created_at': comment.created_at,
                'replies': [comment_to_dict(reply) for reply in comment.replies]
            }
        
        return [comment_to_dict(c) for c in self.comments[result_id]]
    
    def create_collaboration_session(
        self,
        result_id: str,
        owner_id: str,
        participants: List[str]
    ) -> CollaborationSession:
        """
        Crear sesión de colaboración
        
        Args:
            result_id: ID del resultado
            owner_id: ID del propietario
            participants: Lista de participantes
        """
        session_id = str(uuid.uuid4())
        
        session = CollaborationSession(
            id=session_id,
            result_id=result_id,
            owner_id=owner_id,
            participants=set(participants),
            created_at=time.time()
        )
        
        self.collaboration_sessions[session_id] = session
        return session
    
    def add_participant(self, session_id: str, user_id: str):
        """Agregar participante a sesión"""
        if session_id in self.collaboration_sessions:
            self.collaboration_sessions[session_id].participants.add(user_id)
    
    def remove_participant(self, session_id: str, user_id: str):
        """Remover participante de sesión"""
        if session_id in self.collaboration_sessions:
            self.collaboration_sessions[session_id].participants.discard(user_id)
    
    def get_share_statistics(self, result_id: str) -> Dict:
        """Obtener estadísticas de compartición"""
        links = [link for link in self.share_links.values() if link.result_id == result_id]
        comments = self.get_comments(result_id)
        
        total_access = sum(link.access_count for link in links)
        unique_access = len(self.result_access.get(result_id, set()))
        
        return {
            'share_links_count': len(links),
            'total_access_count': total_access,
            'unique_access_count': unique_access,
            'comments_count': len(comments),
            'active_links': len([l for l in links if not l.expires_at or time.time() < l.expires_at])
        }
    
    def revoke_share_link(self, link_id: str):
        """Revocar link de compartición"""
        if link_id in self.share_links:
            del self.share_links[link_id]


# Instancia global
collaboration_manager = CollaborationManager()

