"""
Gestión de secretos para Robot Movement AI v2.0
Almacenamiento seguro de credenciales y secretos
"""

import os
import base64
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None


class SecretsManager:
    """Gestor de secretos con encriptación"""
    
    def __init__(self, key: Optional[bytes] = None, secrets_file: Optional[str] = None):
        """
        Inicializar gestor de secretos
        
        Args:
            key: Clave de encriptación (opcional, se genera si no se proporciona)
            secrets_file: Archivo para almacenar secretos (opcional)
        """
        self.secrets_file = secrets_file or os.getenv("SECRETS_FILE", ".secrets")
        self.secrets: Dict[str, str] = {}
        self.fernet: Optional[Fernet] = None
        
        if CRYPTOGRAPHY_AVAILABLE:
            if key:
                self.fernet = Fernet(key)
            else:
                # Generar clave desde password o usar default
                password = os.getenv("SECRETS_PASSWORD", "default-password-change-in-production")
                key = self._derive_key(password)
                self.fernet = Fernet(key)
            
            # Cargar secretos existentes
            self._load_secrets()
    
    def _derive_key(self, password: str) -> bytes:
        """Derivar clave desde password"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography package is required")
        
        password_bytes = password.encode()
        salt = b'robot_movement_ai_salt'  # En producción, usar salt único
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def _load_secrets(self):
        """Cargar secretos desde archivo"""
        if not Path(self.secrets_file).exists():
            return
        
        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            if self.fernet and encrypted_data:
                decrypted_data = self.fernet.decrypt(encrypted_data)
                import json
                self.secrets = json.loads(decrypted_data.decode())
        except Exception as e:
            print(f"Warning: Could not load secrets: {e}")
    
    def _save_secrets(self):
        """Guardar secretos en archivo"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return
        
        try:
            import json
            data = json.dumps(self.secrets).encode()
            
            if self.fernet:
                encrypted_data = self.fernet.encrypt(data)
            else:
                encrypted_data = data
            
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Establecer permisos restrictivos
            os.chmod(self.secrets_file, 0o600)
        except Exception as e:
            print(f"Warning: Could not save secrets: {e}")
    
    def set_secret(self, key: str, value: str):
        """Establecer secreto"""
        self.secrets[key] = value
        self._save_secrets()
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Obtener secreto"""
        # Primero intentar desde variables de entorno
        env_value = os.getenv(key)
        if env_value:
            return env_value
        
        # Luego desde archivo de secretos
        return self.secrets.get(key, default)
    
    def delete_secret(self, key: str):
        """Eliminar secreto"""
        if key in self.secrets:
            del self.secrets[key]
            self._save_secrets()
    
    def list_secrets(self) -> list:
        """Listar todas las claves de secretos"""
        return list(self.secrets.keys())


# Instancia global
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Obtener instancia global del gestor de secretos"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Helper para obtener secreto"""
    manager = get_secrets_manager()
    return manager.get_secret(key, default)


def set_secret(key: str, value: str):
    """Helper para establecer secreto"""
    manager = get_secrets_manager()
    manager.set_secret(key, value)




