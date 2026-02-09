"""
JWT Handler - Manejo de tokens JWT
"""

from typing import Dict, Any, Optional
import jwt
from datetime import datetime, timedelta
from configs.settings import Settings


class JWTHandler:
    """Manejador de tokens JWT"""

    def __init__(self, settings: Optional[Settings] = None):
        """Inicializa el manejador JWT"""
        self.settings = settings or Settings()
        self.secret = self.settings.jwt_secret or self.settings.secret_key
        self.algorithm = self.settings.jwt_algorithm

    def encode(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Codifica un payload en JWT"""
        payload["exp"] = datetime.utcnow() + timedelta(seconds=expires_in)
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def decode(self, token: str) -> Optional[Dict[str, Any]]:
        """Decodifica un token JWT"""
        try:
            return jwt.decode(token, self.secret, algorithms=[self.algorithm])
        except jwt.InvalidTokenError:
            return None

