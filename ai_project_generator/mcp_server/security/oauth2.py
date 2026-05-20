"""
MCP OAuth2 Provider - Proveedor OAuth2 para MCP
===============================================
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from jose import jwt

logger = logging.getLogger(__name__)


class MCPOAuth2Provider:
    """
    Proveedor OAuth2 simplificado para MCP
    
    Soporta:
    - Client credentials flow
    - Token generation
    - Token validation
    """
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Inicializa el proveedor OAuth2
        
        Args:
            secret_key: Clave secreta para JWT
            algorithm: Algoritmo de JWT
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(
        self,
        client_id: str,
        scopes: list[str],
        expires_in: int = 3600,
        **extra_claims
    ) -> Dict[str, str]:
        """
        Crea token de acceso OAuth2
        
        Args:
            client_id: ID del cliente
            scopes: Lista de scopes
            expires_in: Segundos hasta expiración
            **extra_claims: Claims adicionales
            
        Returns:
            Diccionario con access_token y token_type
        """
        now = datetime.utcnow()
        expire = now + timedelta(seconds=expires_in)
        
        payload = {
            "sub": client_id,
            "scopes": scopes,
            "iat": now,
            "exp": expire,
            "token_type": "Bearer",
            **extra_claims,
        }
        
        access_token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "scope": " ".join(scopes),
        }
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """
        Valida token OAuth2
        
        Args:
            token: Token a validar
            
        Returns:
            Payload del token o None si es inválido
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return None

