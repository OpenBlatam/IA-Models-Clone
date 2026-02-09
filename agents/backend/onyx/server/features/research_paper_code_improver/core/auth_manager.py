"""
Auth Manager - Sistema de autenticación y autorización
=======================================================
"""

import logging
import jwt
import hashlib
import secrets
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Gestiona autenticación JWT y API keys.
    """
    
    def __init__(self, secret_key: Optional[str] = None, tokens_dir: str = "data/tokens"):
        """
        Inicializar gestor de autenticación.
        
        Args:
            secret_key: Clave secreta para JWT (opcional, se genera si no se proporciona)
            tokens_dir: Directorio para almacenar tokens
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.tokens_dir = Path(tokens_dir)
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self._load_api_keys()
    
    def generate_token(
        self,
        user_id: str,
        email: Optional[str] = None,
        expires_in_hours: int = 24
    ) -> str:
        """
        Genera un token JWT.
        
        Args:
            user_id: ID del usuario
            email: Email del usuario (opcional)
            expires_in_hours: Horas hasta expiración
            
        Returns:
            Token JWT
        """
        payload = {
            "user_id": user_id,
            "email": email,
            "exp": datetime.utcnow() + timedelta(hours=expires_in_hours),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        logger.info(f"Token generado para usuario: {user_id}")
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verifica un token JWT.
        
        Args:
            token: Token a verificar
            
        Returns:
            Payload del token o None si es inválido
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expirado")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token inválido: {e}")
            return None
    
    def generate_api_key(
        self,
        user_id: str,
        name: str,
        permissions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Genera una API key.
        
        Args:
            user_id: ID del usuario
            name: Nombre de la API key
            permissions: Permisos (opcional)
            
        Returns:
            Información de la API key
        """
        import uuid
        
        api_key = secrets.token_urlsafe(32)
        key_id = str(uuid.uuid4())
        
        # Hash de la API key para almacenamiento
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        key_info = {
            "id": key_id,
            "user_id": user_id,
            "name": name,
            "key_hash": key_hash,
            "permissions": permissions or ["read", "write"],
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "active": True
        }
        
        self.api_keys[key_id] = key_info
        self._save_api_keys()
        
        logger.info(f"API key generada: {key_id}")
        
        return {
            "id": key_id,
            "api_key": api_key,  # Solo se muestra una vez
            "name": name,
            "created_at": key_info["created_at"]
        }
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Verifica una API key.
        
        Args:
            api_key: API key a verificar
            
        Returns:
            Información de la API key o None si es inválida
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for key_id, key_info in self.api_keys.items():
            if key_info["key_hash"] == key_hash and key_info.get("active", True):
                # Actualizar último uso
                key_info["last_used"] = datetime.now().isoformat()
                self._save_api_keys()
                
                return key_info
        
        return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoca una API key.
        
        Args:
            key_id: ID de la API key
            
        Returns:
            True si se revocó exitosamente
        """
        if key_id in self.api_keys:
            self.api_keys[key_id]["active"] = False
            self._save_api_keys()
            logger.info(f"API key revocada: {key_id}")
            return True
        
        return False
    
    def list_api_keys(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lista API keys.
        
        Args:
            user_id: ID del usuario (opcional, filtra por usuario)
            
        Returns:
            Lista de API keys
        """
        keys = list(self.api_keys.values())
        
        if user_id:
            keys = [k for k in keys if k["user_id"] == user_id]
        
        # No incluir hash en la respuesta
        return [
            {
                "id": k["id"],
                "name": k["name"],
                "user_id": k["user_id"],
                "permissions": k["permissions"],
                "created_at": k["created_at"],
                "last_used": k["last_used"],
                "active": k.get("active", True)
            }
            for k in keys
        ]
    
    def _load_api_keys(self):
        """Carga API keys desde disco"""
        keys_file = self.tokens_dir / "api_keys.json"
        
        if keys_file.exists():
            try:
                with open(keys_file, "r", encoding="utf-8") as f:
                    self.api_keys = json.load(f)
                logger.info(f"API keys cargadas: {len(self.api_keys)}")
            except Exception as e:
                logger.error(f"Error cargando API keys: {e}")
    
    def _save_api_keys(self):
        """Guarda API keys en disco"""
        keys_file = self.tokens_dir / "api_keys.json"
        
        try:
            with open(keys_file, "w", encoding="utf-8") as f:
                json.dump(self.api_keys, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando API keys: {e}")




