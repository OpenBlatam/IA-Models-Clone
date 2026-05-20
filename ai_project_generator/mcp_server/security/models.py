"""
Security Models - Modelos de seguridad MCP
===========================================
"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class Scope(str, Enum):
    """Scopes de acceso MCP"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"


class AccessPolicy(BaseModel):
    """
    Política de acceso para un recurso
    
    Define qué scopes son requeridos para acceder a un recurso.
    """
    
    resource_id: str = Field(..., description="ID del recurso")
    required_scopes: List[Scope] = Field(
        default_factory=list,
        description="Scopes requeridos"
    )
    allowed_users: Optional[List[str]] = Field(
        None,
        description="Lista de usuarios permitidos (None = todos)"
    )
    denied_users: Optional[List[str]] = Field(
        None,
        description="Lista de usuarios denegados"
    )
    rate_limit: Optional[Dict[str, Any]] = Field(
        None,
        description="Límites de rate por usuario"
    )
    
    def allows(self, user_id: str, scope: Scope) -> bool:
        """
        Verifica si un usuario tiene acceso con un scope
        
        Args:
            user_id: ID del usuario
            scope: Scope requerido
            
        Returns:
            True si tiene acceso
        """
        # Verificar usuarios denegados
        if self.denied_users and user_id in self.denied_users:
            return False
        
        # Verificar usuarios permitidos
        if self.allowed_users and user_id not in self.allowed_users:
            return False
        
        # Verificar scope
        if scope not in self.required_scopes:
            return False
        
        return True


class AccessLog(BaseModel):
    """
    Log de acceso a recursos MCP
    
    Registra todas las operaciones para auditoría.
    """
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str = Field(..., description="ID del usuario")
    resource_id: str = Field(..., description="ID del recurso accedido")
    operation: str = Field(..., description="Operación realizada")
    scope: Scope = Field(..., description="Scope utilizado")
    success: bool = Field(..., description="Si la operación fue exitosa")
    error: Optional[str] = Field(None, description="Error si hubo fallo")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata adicional"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte log a diccionario"""
        return self.dict(exclude_none=True)

