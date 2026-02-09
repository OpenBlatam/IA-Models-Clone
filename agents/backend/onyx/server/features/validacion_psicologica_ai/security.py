"""
Seguridad Avanzada para Validación Psicológica AI
=================================================
Encriptación avanzada y gestión de secretos
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import structlog
import hashlib
import secrets
import base64

logger = structlog.get_logger()

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("cryptography not available, using basic encryption")


class AdvancedTokenEncryption:
    """Encriptación avanzada de tokens"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Inicializar encriptador
        
        Args:
            encryption_key: Clave de encriptación (opcional)
        """
        self.encryption_key = encryption_key or self._generate_key()
        
        if CRYPTOGRAPHY_AVAILABLE:
            self.cipher = self._create_cipher()
        else:
            self.cipher = None
            logger.warning("Using basic encryption (cryptography not available)")
    
    def _generate_key(self) -> str:
        """Generar clave de encriptación"""
        return secrets.token_urlsafe(32)
    
    def _create_cipher(self):
        """Crear cipher de Fernet"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return None
        
        # Derivar clave usando PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'psychological_validation_salt',  # En producción, usar salt único
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key.encode()))
        return Fernet(key)
    
    def encrypt(self, token: str) -> str:
        """
        Encriptar token
        
        Args:
            token: Token a encriptar
            
        Returns:
            Token encriptado
        """
        if self.cipher:
            # Usar Fernet para encriptación fuerte
            encrypted = self.cipher.encrypt(token.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        else:
            # Fallback a encriptación básica
            combined = f"{token}{self.encryption_key}"
            return hashlib.sha256(combined.encode()).hexdigest()
    
    def decrypt(self, encrypted_token: str) -> Optional[str]:
        """
        Desencriptar token
        
        Args:
            encrypted_token: Token encriptado
            
        Returns:
            Token desencriptado o None si falla
        """
        if not self.cipher:
            logger.warning("Decryption not available without cryptography library")
            return None
        
        try:
            decoded = base64.urlsafe_b64decode(encrypted_token.encode())
            decrypted = self.cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error("Error decrypting token", error=str(e))
            return None
    
    def verify_token(self, encrypted_token: str, original_token: str) -> bool:
        """
        Verificar token sin desencriptar
        
        Args:
            encrypted_token: Token encriptado
            original_token: Token original
            
        Returns:
            True si coinciden
        """
        if self.cipher:
            # Encriptar original y comparar
            encrypted_original = self.encrypt(original_token)
            return encrypted_token == encrypted_original
        else:
            # Comparación de hash
            combined = f"{original_token}{self.encryption_key}"
            expected_hash = hashlib.sha256(combined.encode()).hexdigest()
            return encrypted_token == expected_hash


class TokenManager:
    """Gestor de tokens con expiración y renovación"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Inicializar gestor
        
        Args:
            encryption_key: Clave de encriptación
        """
        self.encryptor = AdvancedTokenEncryption(encryption_key)
        self._tokens: Dict[str, Dict[str, Any]] = {}
        logger.info("TokenManager initialized")
    
    def store_token(
        self,
        token_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_in: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Almacenar token de forma segura
        
        Args:
            token_id: ID único del token
            access_token: Token de acceso
            refresh_token: Token de refresco (opcional)
            expires_in: Tiempo de expiración en segundos (opcional)
            
        Returns:
            Información del token almacenado
        """
        encrypted_access = self.encryptor.encrypt(access_token)
        encrypted_refresh = (
            self.encryptor.encrypt(refresh_token)
            if refresh_token else None
        )
        
        expires_at = (
            datetime.utcnow() + timedelta(seconds=expires_in)
            if expires_in else None
        )
        
        token_data = {
            "id": token_id,
            "encrypted_access_token": encrypted_access,
            "encrypted_refresh_token": encrypted_refresh,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None
        }
        
        self._tokens[token_id] = token_data
        
        logger.info("Token stored", token_id=token_id)
        
        return token_data
    
    def get_token(self, token_id: str) -> Optional[str]:
        """
        Obtener token desencriptado
        
        Args:
            token_id: ID del token
            
        Returns:
            Token desencriptado o None
        """
        if token_id not in self._tokens:
            return None
        
        token_data = self._tokens[token_id]
        
        # Verificar expiración
        if token_data["expires_at"]:
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            if datetime.utcnow() > expires_at:
                logger.warning("Token expired", token_id=token_id)
                return None
        
        # Actualizar último uso
        token_data["last_used"] = datetime.utcnow().isoformat()
        
        # Desencriptar
        encrypted_token = token_data["encrypted_access_token"]
        return self.encryptor.decrypt(encrypted_token)
    
    def refresh_token(self, token_id: str) -> Optional[str]:
        """
        Renovar token usando refresh token
        
        Args:
            token_id: ID del token
            
        Returns:
            Nuevo token de acceso o None
        """
        if token_id not in self._tokens:
            return None
        
        token_data = self._tokens[token_id]
        
        if not token_data["encrypted_refresh_token"]:
            logger.warning("No refresh token available", token_id=token_id)
            return None
        
        # En producción, usar refresh token para obtener nuevo access token
        # Por ahora, retornar el token actual
        return self.get_token(token_id)
    
    def is_token_valid(self, token_id: str) -> bool:
        """
        Verificar si token es válido
        
        Args:
            token_id: ID del token
            
        Returns:
            True si es válido
        """
        if token_id not in self._tokens:
            return False
        
        token_data = self._tokens[token_id]
        
        # Verificar expiración
        if token_data["expires_at"]:
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            if datetime.utcnow() > expires_at:
                return False
        
        return True
    
    def delete_token(self, token_id: str) -> bool:
        """
        Eliminar token
        
        Args:
            token_id: ID del token
            
        Returns:
            True si se eliminó exitosamente
        """
        if token_id in self._tokens:
            del self._tokens[token_id]
            logger.info("Token deleted", token_id=token_id)
            return True
        return False


class SecurityAuditor:
    """Auditor de seguridad"""
    
    def __init__(self):
        """Inicializar auditor"""
        self._audit_log: List[Dict[str, Any]] = []
        logger.info("SecurityAuditor initialized")
    
    def log_access(
        self,
        resource: str,
        user_id: Optional[str] = None,
        action: str = "access",
        success: bool = True
    ) -> None:
        """
        Registrar acceso
        
        Args:
            resource: Recurso accedido
            user_id: ID del usuario
            action: Acción realizada
            success: Si fue exitoso
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "resource": resource,
            "user_id": user_id,
            "action": action,
            "success": success
        }
        
        self._audit_log.append(log_entry)
        
        if not success:
            logger.warning(
                "Security event",
                resource=resource,
                user_id=user_id,
                action=action
            )
    
    def get_audit_log(
        self,
        limit: int = 100,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener log de auditoría
        
        Args:
            limit: Límite de resultados
            user_id: Filtrar por usuario
            
        Returns:
            Lista de entradas de log
        """
        log = self._audit_log
        
        if user_id:
            log = [entry for entry in log if entry.get("user_id") == user_id]
        
        log.sort(key=lambda x: x["timestamp"], reverse=True)
        return log[:limit]


# Instancias globales
token_manager = TokenManager()
security_auditor = SecurityAuditor()

