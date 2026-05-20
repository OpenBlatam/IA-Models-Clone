"""
Collaboration System - Sistema de colaboración y compartir
==========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from uuid import uuid4
from enum import Enum

logger = logging.getLogger(__name__)


class SharePermission(str, Enum):
    """Permisos de compartir"""
    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    FULL = "full"


class CollaborationSystem:
    """Sistema de colaboración"""
    
    def __init__(self):
        self.shared_prototypes: Dict[str, Dict[str, Any]] = {}
        self.comments: Dict[str, List[Dict[str, Any]]] = {}
        self.permissions: Dict[str, Dict[str, SharePermission]] = {}
    
    def share_prototype(self, prototype_id: str, owner_id: str,
                       share_with: List[str], permission: SharePermission = SharePermission.VIEW,
                       expires_in_days: Optional[int] = None) -> str:
        """
        Comparte un prototipo con otros usuarios
        
        Returns:
            Share token único
        """
        share_token = str(uuid4())
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        share_record = {
            "token": share_token,
            "prototype_id": prototype_id,
            "owner_id": owner_id,
            "shared_with": share_with,
            "permission": permission.value,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None,
            "active": True
        }
        
        self.shared_prototypes[share_token] = share_record
        
        # Establecer permisos
        for user_id in share_with:
            if prototype_id not in self.permissions:
                self.permissions[prototype_id] = {}
            self.permissions[prototype_id][user_id] = permission
        
        logger.info(f"Prototipo {prototype_id} compartido con token {share_token}")
        return share_token
    
    def get_shared_prototype(self, share_token: str) -> Optional[Dict[str, Any]]:
        """Obtiene un prototipo compartido por token"""
        share_record = self.shared_prototypes.get(share_token)
        
        if not share_record or not share_record.get("active"):
            return None
        
        # Verificar expiración
        if share_record.get("expires_at"):
            expires_at = datetime.fromisoformat(share_record["expires_at"])
            if datetime.now() > expires_at:
                share_record["active"] = False
                return None
        
        return share_record
    
    def revoke_share(self, share_token: str, owner_id: str) -> bool:
        """Revoca el compartir de un prototipo"""
        share_record = self.shared_prototypes.get(share_token)
        
        if not share_record or share_record["owner_id"] != owner_id:
            return False
        
        share_record["active"] = False
        return True
    
    def add_comment(self, prototype_id: str, user_id: str, comment: str,
                   parent_comment_id: Optional[str] = None) -> str:
        """Agrega un comentario a un prototipo"""
        comment_id = str(uuid4())
        
        comment_record = {
            "id": comment_id,
            "prototype_id": prototype_id,
            "user_id": user_id,
            "comment": comment,
            "parent_comment_id": parent_comment_id,
            "created_at": datetime.now().isoformat(),
            "edited": False
        }
        
        if prototype_id not in self.comments:
            self.comments[prototype_id] = []
        
        self.comments[prototype_id].append(comment_record)
        
        logger.info(f"Comentario agregado a prototipo {prototype_id}")
        return comment_id
    
    def get_comments(self, prototype_id: str) -> List[Dict[str, Any]]:
        """Obtiene comentarios de un prototipo"""
        return self.comments.get(prototype_id, [])
    
    def edit_comment(self, comment_id: str, user_id: str, new_comment: str) -> bool:
        """Edita un comentario"""
        for comments in self.comments.values():
            for comment in comments:
                if comment["id"] == comment_id and comment["user_id"] == user_id:
                    comment["comment"] = new_comment
                    comment["edited"] = True
                    comment["edited_at"] = datetime.now().isoformat()
                    return True
        return False
    
    def delete_comment(self, comment_id: str, user_id: str) -> bool:
        """Elimina un comentario"""
        for comments in self.comments.values():
            for i, comment in enumerate(comments):
                if comment["id"] == comment_id and comment["user_id"] == user_id:
                    comments.pop(i)
                    return True
        return False
    
    def check_permission(self, prototype_id: str, user_id: str, 
                        required_permission: SharePermission) -> bool:
        """Verifica si un usuario tiene un permiso"""
        prototype_perms = self.permissions.get(prototype_id, {})
        user_permission = prototype_perms.get(user_id)
        
        if not user_permission:
            return False
        
        permission_levels = {
            SharePermission.VIEW: 1,
            SharePermission.COMMENT: 2,
            SharePermission.EDIT: 3,
            SharePermission.FULL: 4
        }
        
        user_level = permission_levels.get(user_permission, 0)
        required_level = permission_levels.get(required_permission, 0)
        
        return user_level >= required_level
    
    def get_shared_with_me(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtiene prototipos compartidos conmigo"""
        shared = []
        
        for share_token, share_record in self.shared_prototypes.items():
            if (share_record.get("active") and 
                user_id in share_record.get("shared_with", [])):
                shared.append({
                    "share_token": share_token,
                    "prototype_id": share_record["prototype_id"],
                    "owner_id": share_record["owner_id"],
                    "permission": share_record["permission"],
                    "created_at": share_record["created_at"]
                })
        
        return shared
    
    def get_my_shared(self, owner_id: str) -> List[Dict[str, Any]]:
        """Obtiene prototipos que he compartido"""
        shared = []
        
        for share_token, share_record in self.shared_prototypes.items():
            if share_record.get("owner_id") == owner_id and share_record.get("active"):
                shared.append({
                    "share_token": share_token,
                    "prototype_id": share_record["prototype_id"],
                    "shared_with": share_record["shared_with"],
                    "permission": share_record["permission"],
                    "created_at": share_record["created_at"],
                    "expires_at": share_record.get("expires_at")
                })
        
        return shared




