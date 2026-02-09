"""
Sistema de colaboración y compartir
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import uuid


class SharePermission(str, Enum):
    """Permisos de compartir"""
    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    FULL = "full"


@dataclass
class SharedResource:
    """Recurso compartido"""
    id: str
    resource_type: str  # "analysis", "report", etc.
    resource_id: str
    owner_id: str
    shared_with: List[str]  # Lista de user_ids
    permission: SharePermission
    created_at: str = None
    expires_at: Optional[str] = None
    public_link: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "owner_id": self.owner_id,
            "shared_with": self.shared_with,
            "permission": self.permission.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "public_link": self.public_link
        }


class CollaborationService:
    """Servicio de colaboración"""
    
    def __init__(self):
        """Inicializa el servicio"""
        self.shared_resources: Dict[str, SharedResource] = {}
        self.comments: Dict[str, List[Dict]] = {}  # resource_id -> comments
    
    def share_resource(self, resource_type: str, resource_id: str,
                      owner_id: str, shared_with: List[str],
                      permission: SharePermission,
                      expires_at: Optional[str] = None,
                      public: bool = False) -> str:
        """
        Comparte un recurso
        
        Args:
            resource_type: Tipo de recurso
            resource_id: ID del recurso
            owner_id: ID del propietario
            shared_with: Lista de usuarios con quienes compartir
            permission: Permiso
            expires_at: Fecha de expiración
            public: Si es público
            
        Returns:
            ID del recurso compartido
        """
        share_id = str(uuid.uuid4())
        
        public_link = None
        if public:
            public_link = self._generate_public_link(share_id)
        
        shared_resource = SharedResource(
            id=share_id,
            resource_type=resource_type,
            resource_id=resource_id,
            owner_id=owner_id,
            shared_with=shared_with if not public else [],
            permission=permission,
            expires_at=expires_at,
            public_link=public_link
        )
        
        self.shared_resources[share_id] = shared_resource
        return share_id
    
    def get_shared_resource(self, share_id: str) -> Optional[SharedResource]:
        """Obtiene un recurso compartido"""
        return self.shared_resources.get(share_id)
    
    def get_user_shared_resources(self, user_id: str) -> List[SharedResource]:
        """Obtiene recursos compartidos con un usuario"""
        resources = []
        
        for resource in self.shared_resources.values():
            if user_id in resource.shared_with or resource.public_link:
                resources.append(resource)
        
        return resources
    
    def revoke_share(self, share_id: str, owner_id: str) -> bool:
        """
        Revoca un recurso compartido
        
        Args:
            share_id: ID del recurso compartido
            owner_id: ID del propietario
            
        Returns:
            True si se revocó correctamente
        """
        resource = self.shared_resources.get(share_id)
        
        if not resource or resource.owner_id != owner_id:
            return False
        
        del self.shared_resources[share_id]
        return True
    
    def add_comment(self, resource_id: str, user_id: str, comment: str) -> str:
        """
        Agrega un comentario
        
        Args:
            resource_id: ID del recurso
            user_id: ID del usuario
            comment: Comentario
            
        Returns:
            ID del comentario
        """
        comment_id = str(uuid.uuid4())
        
        comment_data = {
            "id": comment_id,
            "user_id": user_id,
            "comment": comment,
            "created_at": datetime.now().isoformat()
        }
        
        if resource_id not in self.comments:
            self.comments[resource_id] = []
        
        self.comments[resource_id].append(comment_data)
        return comment_id
    
    def get_comments(self, resource_id: str) -> List[Dict]:
        """Obtiene comentarios de un recurso"""
        return self.comments.get(resource_id, [])
    
    def _generate_public_link(self, share_id: str) -> str:
        """Genera link público"""
        token = hashlib.md5(f"{share_id}{datetime.now().isoformat()}".encode()).hexdigest()
        return f"public/{token}"






