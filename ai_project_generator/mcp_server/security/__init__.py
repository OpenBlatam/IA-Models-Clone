"""
MCP Security - Seguridad y autenticación para MCP
==================================================

Implementa:
- OAuth2/JWT para endpoints MCP
- Registro de access logs
- Políticas de scopes por recurso
- Auditoría de acceso
"""

from .manager import MCPSecurityManager
from .models import Scope, AccessPolicy, AccessLog
from .oauth2 import MCPOAuth2Provider

__all__ = [
    "MCPSecurityManager",
    "Scope",
    "AccessPolicy",
    "AccessLog",
    "MCPOAuth2Provider",
]

