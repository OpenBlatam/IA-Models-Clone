"""
Collaboration Service - Sistema de colaboración y compartir diseños
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class SharePermission(str, Enum):
    """Permisos de compartir"""
    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    FULL = "full"


class CollaborationService:
    """Servicio para colaboración y compartir"""
    
    def __init__(self):
        self.shared_designs: Dict[str, Dict[str, Any]] = {}
        self.comments: Dict[str, List[Dict[str, Any]]] = {}
    
    def share_design(
        self,
        store_id: str,
        shared_by: str,
        permission: SharePermission = SharePermission.VIEW,
        expires_in_days: Optional[int] = None,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compartir diseño"""
        
        share_id = str(uuid.uuid4())
        expires_at = None
        
        if expires_in_days:
            expires_at = (datetime.now() + timedelta(days=expires_in_days)).isoformat()
        
        share_info = {
            "share_id": share_id,
            "store_id": store_id,
            "shared_by": shared_by,
            "permission": permission.value,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at,
            "password": password,
            "access_count": 0,
            "is_active": True
        }
        
        self.shared_designs[share_id] = share_info
        
        logger.info(f"Diseño {store_id} compartido con ID {share_id}")
        return share_info
    
    def get_shared_design(self, share_id: str, password: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Obtener diseño compartido"""
        share_info = self.shared_designs.get(share_id)
        
        if not share_info or not share_info.get("is_active"):
            return None
        
        # Verificar expiración
        if share_info.get("expires_at"):
            expires_at = datetime.fromisoformat(share_info["expires_at"])
            if datetime.now() > expires_at:
                share_info["is_active"] = False
                return None
        
        # Verificar contraseña
        if share_info.get("password"):
            if password != share_info["password"]:
                return {"error": "Contraseña incorrecta"}
        
        # Incrementar contador de acceso
        share_info["access_count"] = share_info.get("access_count", 0) + 1
        
        return share_info
    
    def revoke_share(self, share_id: str, revoked_by: str) -> bool:
        """Revocar compartir"""
        share_info = self.shared_designs.get(share_id)
        
        if not share_info:
            return False
        
        if share_info["shared_by"] != revoked_by:
            return False
        
        share_info["is_active"] = False
        share_info["revoked_at"] = datetime.now().isoformat()
        share_info["revoked_by"] = revoked_by
        
        return True
    
    def add_comment(
        self,
        store_id: str,
        commenter: str,
        content: str,
        section: Optional[str] = None
    ) -> Dict[str, Any]:
        """Agregar comentario a diseño"""
        
        comment = {
            "id": str(uuid.uuid4()),
            "store_id": store_id,
            "commenter": commenter,
            "content": content,
            "section": section,  # "layout", "decoration", "marketing", "financial", etc.
            "created_at": datetime.now().isoformat(),
            "replies": []
        }
        
        if store_id not in self.comments:
            self.comments[store_id] = []
        
        self.comments[store_id].append(comment)
        
        return comment
    
    def get_comments(self, store_id: str) -> List[Dict[str, Any]]:
        """Obtener comentarios de diseño"""
        return self.comments.get(store_id, [])
    
    def reply_to_comment(
        self,
        store_id: str,
        comment_id: str,
        replier: str,
        content: str
    ) -> Optional[Dict[str, Any]]:
        """Responder a comentario"""
        comments = self.comments.get(store_id, [])
        
        for comment in comments:
            if comment["id"] == comment_id:
                reply = {
                    "id": str(uuid.uuid4()),
                    "replier": replier,
                    "content": content,
                    "created_at": datetime.now().isoformat()
                }
                comment["replies"].append(reply)
                return reply
        
        return None
    
    def get_shared_designs_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener diseños compartidos por usuario"""
        return [
            share for share in self.shared_designs.values()
            if share["shared_by"] == user_id and share.get("is_active", True)
        ]
    
    def get_shared_designs_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener diseños compartidos con usuario (placeholder)"""
        # En producción, esto buscaría en base de datos
        return []




