"""
Sistema de colaboración y multi-usuario
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, JSON, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from enum import Enum

from ..db.base import Base, get_db_session

logger = logging.getLogger(__name__)


class PermissionLevel(str, Enum):
    """Niveles de permiso"""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class IdentityShareModel(Base):
    """Modelo de compartir identidad"""
    __tablename__ = "identity_shares"
    
    id = Column(String(64), primary_key=True, index=True)
    identity_profile_id = Column(String(64), ForeignKey("identity_profiles.id"), nullable=False, index=True)
    shared_with_user_id = Column(String(255), nullable=False, index=True)
    permission_level = Column(String(20), nullable=False, default=PermissionLevel.VIEWER.value)
    shared_by_user_id = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class CollaborationService:
    """Servicio de colaboración"""
    
    def __init__(self):
        self._init_table()
    
    def _init_table(self):
        """Inicializa tablas de colaboración"""
        from ..db.base import init_db
        init_db()
    
    def share_identity(
        self,
        identity_id: str,
        shared_with_user_id: str,
        permission_level: PermissionLevel,
        shared_by_user_id: str
    ) -> str:
        """
        Comparte una identidad con otro usuario
        
        Args:
            identity_id: ID de la identidad
            shared_with_user_id: ID del usuario con quien compartir
            permission_level: Nivel de permiso
            shared_by_user_id: ID del usuario que comparte
            
        Returns:
            ID del share
        """
        share_id = str(uuid.uuid4())
        
        with get_db_session() as db:
            # Verificar si ya existe
            existing = db.query(IdentityShareModel).filter_by(
                identity_profile_id=identity_id,
                shared_with_user_id=shared_with_user_id
            ).first()
            
            if existing:
                # Actualizar permiso
                existing.permission_level = permission_level.value
                existing.updated_at = datetime.utcnow()
                share_id = existing.id
            else:
                # Crear nuevo
                share = IdentityShareModel(
                    id=share_id,
                    identity_profile_id=identity_id,
                    shared_with_user_id=shared_with_user_id,
                    permission_level=permission_level.value,
                    shared_by_user_id=shared_by_user_id
                )
                db.add(share)
            
            db.commit()
        
        logger.info(f"Identidad {identity_id} compartida con {shared_with_user_id} ({permission_level.value})")
        return share_id
    
    def get_shared_identities(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtiene identidades compartidas con un usuario"""
        with get_db_session() as db:
            shares = db.query(IdentityShareModel).filter_by(
                shared_with_user_id=user_id
            ).all()
            
            return [
                {
                    "share_id": s.id,
                    "identity_id": s.identity_profile_id,
                    "permission_level": s.permission_level,
                    "shared_by": s.shared_by_user_id,
                    "created_at": s.created_at.isoformat()
                }
                for s in shares
            ]
    
    def check_permission(
        self,
        identity_id: str,
        user_id: str,
        required_permission: PermissionLevel
    ) -> bool:
        """Verifica si usuario tiene permiso"""
        with get_db_session() as db:
            share = db.query(IdentityShareModel).filter_by(
                identity_profile_id=identity_id,
                shared_with_user_id=user_id
            ).first()
            
            if not share:
                return False
            
            # Verificar nivel de permiso
            permission_hierarchy = {
                PermissionLevel.VIEWER: 1,
                PermissionLevel.EDITOR: 2,
                PermissionLevel.ADMIN: 3,
                PermissionLevel.OWNER: 4
            }
            
            user_level = permission_hierarchy.get(PermissionLevel(share.permission_level), 0)
            required_level = permission_hierarchy.get(required_permission, 0)
            
            return user_level >= required_level
    
    def revoke_share(self, share_id: str) -> bool:
        """Revoca un share"""
        with get_db_session() as db:
            share = db.query(IdentityShareModel).filter_by(id=share_id).first()
            if not share:
                return False
            
            db.delete(share)
            db.commit()
            return True




