"""
MCP Security Manager - Gestor de seguridad MCP
===============================================
"""

import logging
import threading
from typing import Dict, Optional, List, Any, Set
from datetime import datetime, timedelta, timezone
from collections import deque

from jose import JWTError, jwt
from passlib.context import CryptContext

from .models import Scope, AccessPolicy, AccessLog
from .oauth2 import MCPOAuth2Provider

logger = logging.getLogger(__name__)


class MCPSecurityManager:
    """
    Gestor de seguridad para MCP
    
    Maneja:
    - Autenticación JWT
    - Autorización por scopes
    - Políticas de acceso
    - Auditoría de acceso
    """
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
    ):
        """
        Inicializa el gestor de seguridad
        
        Args:
            secret_key: Clave secreta para JWT (debe ser string no vacío, mínimo 32 caracteres recomendado)
            algorithm: Algoritmo de JWT (default: HS256)
            access_token_expire_minutes: Minutos de expiración del token (debe ser > 0)
            
        Raises:
            ValueError: Si secret_key está vacío, algorithm no es soportado, o expire_minutes es inválido
            TypeError: Si los tipos de parámetros son incorrectos
        """
        # Validar secret_key
        if not isinstance(secret_key, str):
            raise TypeError(f"secret_key must be a string, got {type(secret_key)}")
        if not secret_key or not secret_key.strip():
            raise ValueError("secret_key cannot be empty or whitespace")
        secret_key = secret_key.strip()
        if len(secret_key) < 32:
            logger.warning("secret_key is shorter than 32 characters, consider using a longer key for security")
        
        # Validar algorithm
        if not isinstance(algorithm, str):
            raise TypeError(f"algorithm must be a string, got {type(algorithm)}")
        algorithm = algorithm.strip().upper()
        supported_algorithms = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]
        if algorithm not in supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {', '.join(supported_algorithms)}")
        
        # Validar access_token_expire_minutes
        if not isinstance(access_token_expire_minutes, int):
            raise TypeError(f"access_token_expire_minutes must be an integer, got {type(access_token_expire_minutes)}")
        if access_token_expire_minutes <= 0:
            raise ValueError(f"access_token_expire_minutes must be > 0, got {access_token_expire_minutes}")
        if access_token_expire_minutes > 525600:  # 1 año en minutos
            logger.warning(f"access_token_expire_minutes is very large ({access_token_expire_minutes}), consider shorter expiration")
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_provider = MCPOAuth2Provider(secret_key, algorithm)
        
        # Políticas de acceso por recurso (thread-safe)
        self._policies: Dict[str, AccessPolicy] = {}
        self._policies_lock = threading.RLock()
        
        # Logs de acceso (en producción usar DB o sistema de logging)
        # Usar deque para mejor performance en operaciones FIFO
        self._access_logs: deque = deque(maxlen=10000)
        self._max_logs = 10000  # Límite de logs en memoria
        self._logs_lock = threading.RLock()
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """
        Crea token de acceso JWT
        
        Args:
            data: Datos a incluir en el token (debe incluir 'sub' con user_id)
            
        Returns:
            Token JWT
            
        Raises:
            ValueError: Si data está vacío, falta 'sub', o hay error al crear el token
            TypeError: Si data no es un diccionario
        """
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dictionary, got {type(data)}")
        if not data:
            raise ValueError("data cannot be empty")
        
        # Validar que tenga 'sub'
        if "sub" not in data:
            raise ValueError("data must include 'sub' key with user_id")
        if not data.get("sub") or not isinstance(data.get("sub"), str):
            raise ValueError("'sub' must be a non-empty string")
        
        try:
            to_encode = data.copy()
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
            to_encode.update({"exp": expire})
            
            return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        except Exception as e:
            logger.error(f"Error creating access token: {e}", exc_info=True)
            raise ValueError(f"Failed to create access token: {e}") from e
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verifica y decodifica token JWT
        
        Args:
            token: Token JWT (debe ser string no vacío)
            
        Returns:
            Payload del token (incluye 'sub' con user_id)
            
        Raises:
            ValueError: Si el token es inválido, está vacío, o falta 'sub'
            TypeError: Si token no es string
        """
        if not isinstance(token, str):
            raise TypeError(f"token must be a string, got {type(token)}")
        if not token or not token.strip():
            raise ValueError("token cannot be empty or whitespace")
        
        token = token.strip()
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Validar que tenga 'sub'
            user_id: str = payload.get("sub")
            if user_id is None:
                raise ValueError("Token missing 'sub' claim")
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("Token 'sub' claim must be a non-empty string")
            
            return payload
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise ValueError(f"Invalid token: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error verifying token: {e}", exc_info=True)
            raise ValueError(f"Failed to verify token: {e}") from e
    
    def set_policy(self, resource_id: str, policy: AccessPolicy) -> None:
        """
        Establece política de acceso para un recurso (thread-safe).
        
        Args:
            resource_id: ID del recurso
            policy: Política de acceso
        """
        if not resource_id or not isinstance(resource_id, str):
            raise ValueError("resource_id must be a non-empty string")
        if not isinstance(policy, AccessPolicy):
            raise TypeError("policy must be an AccessPolicy instance")
        
        with self._policies_lock:
            self._policies[resource_id] = policy
        logger.info(f"Set access policy for resource {resource_id}")
    
    def get_policy(self, resource_id: str) -> Optional[AccessPolicy]:
        """
        Obtiene política de acceso para un recurso (thread-safe).
        
        Args:
            resource_id: ID del recurso
            
        Returns:
            Política o None si no existe
        """
        if not resource_id:
            return None
        
        with self._policies_lock:
            return self._policies.get(resource_id)
    
    def has_access(
        self,
        user_payload: Dict[str, Any],
        resource_id: str,
        required_scope: Scope,
    ) -> bool:
        """
        Verifica si un usuario tiene acceso a un recurso
        
        Args:
            user_payload: Payload del token del usuario (debe ser dict no vacío)
            resource_id: ID del recurso (debe ser string no vacío)
            required_scope: Scope requerido (debe ser instancia de Scope)
            
        Returns:
            True si tiene acceso, False en caso contrario
            
        Raises:
            ValueError: Si user_payload está vacío, resource_id está vacío, o required_scope es inválido
            TypeError: Si los tipos de parámetros son incorrectos
        """
        # Validar user_payload
        if not isinstance(user_payload, dict):
            raise TypeError(f"user_payload must be a dictionary, got {type(user_payload)}")
        if not user_payload:
            raise ValueError("user_payload cannot be empty")
        
        # Validar resource_id
        if not isinstance(resource_id, str):
            raise TypeError(f"resource_id must be a string, got {type(resource_id)}")
        if not resource_id or not resource_id.strip():
            raise ValueError("resource_id cannot be empty or whitespace")
        
        # Validar required_scope
        if not isinstance(required_scope, Scope):
            raise TypeError(f"required_scope must be a Scope instance, got {type(required_scope)}")
        
        user_id = user_payload.get("sub")
        if not user_id:
            return False
        
        # Obtener scopes del token
        token_scopes_raw = user_payload.get("scopes", [])
        if isinstance(token_scopes_raw, str):
            token_scopes: Set[str] = {s.strip() for s in token_scopes_raw.split(",") if s.strip()}
        elif isinstance(token_scopes_raw, list):
            token_scopes = {str(s).strip() for s in token_scopes_raw if s}
        else:
            token_scopes = set()
        
        # Verificar scope en token
        required_scope_value = required_scope.value if isinstance(required_scope, Scope) else str(required_scope)
        if required_scope_value not in token_scopes and "admin" not in token_scopes:
            self._log_access(user_id, resource_id, "check", required_scope, False, "Missing scope")
            return False
        
        # Verificar política específica
        policy = self.get_policy(resource_id)
        if policy:
            if not policy.allows(user_id, required_scope):
                self._log_access(user_id, resource_id, "check", required_scope, False, "Policy denied")
                return False
        
        self._log_access(user_id, resource_id, "check", required_scope, True)
        return True
    
    def _log_access(
        self,
        user_id: str,
        resource_id: str,
        operation: str,
        scope: Scope,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Registra acceso para auditoría (thread-safe).
        
        Args:
            user_id: ID del usuario
            resource_id: ID del recurso
            operation: Operación realizada
            scope: Scope utilizado
            success: Si fue exitoso
            error: Error si hubo fallo
            metadata: Metadata adicional
        """
        log = AccessLog(
            user_id=str(user_id),
            resource_id=str(resource_id),
            operation=str(operation),
            scope=scope,
            success=bool(success),
            error=str(error) if error else None,
            metadata=metadata or {},
        )
        
        with self._logs_lock:
            # deque maneja automáticamente el límite con maxlen
            self._access_logs.append(log)
        
        if not success:
            logger.warning(
                f"Access denied: user={user_id}, resource={resource_id}, "
                f"operation={operation}, scope={scope}, error={error}"
            )
    
    def get_access_logs(
        self,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AccessLog]:
        """
        Obtiene logs de acceso (thread-safe, optimizado).
        
        Args:
            user_id: Filtrar por usuario (opcional, debe ser string no vacío si se proporciona)
            resource_id: Filtrar por recurso (opcional, debe ser string no vacío si se proporciona)
            limit: Límite de resultados (máximo 1000, mínimo 1)
            
        Returns:
            Lista de logs de acceso
            
        Raises:
            ValueError: Si user_id o resource_id están vacíos cuando se proporcionan, o limit es inválido
            TypeError: Si los tipos de parámetros son incorrectos
        """
        # Validar user_id
        if user_id is not None:
            if not isinstance(user_id, str):
                raise TypeError(f"user_id must be a string or None, got {type(user_id)}")
            if not user_id or not user_id.strip():
                raise ValueError("user_id cannot be empty or whitespace if provided")
            user_id = user_id.strip()
        
        # Validar resource_id
        if resource_id is not None:
            if not isinstance(resource_id, str):
                raise TypeError(f"resource_id must be a string or None, got {type(resource_id)}")
            if not resource_id or not resource_id.strip():
                raise ValueError("resource_id cannot be empty or whitespace if provided")
            resource_id = resource_id.strip()
        
        # Validar limit
        if not isinstance(limit, int):
            raise TypeError(f"limit must be an integer, got {type(limit)}")
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        
        # Limitar el límite para evitar problemas de memoria
        limit = min(limit, 1000)
        
        with self._logs_lock:
            # Convertir deque a lista para filtrado
            logs = list(self._access_logs)
        
        # Filtrar eficientemente
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        
        if resource_id:
            logs = [log for log in logs if log.resource_id == resource_id]
        
        # Retornar los últimos N logs
        return logs[-limit:] if len(logs) > limit else logs
    
    def hash_password(self, password: str) -> str:
        """
        Hash de contraseña usando bcrypt.
        
        Args:
            password: Contraseña a hashear (debe ser string no vacío)
            
        Returns:
            Hash de la contraseña
            
        Raises:
            ValueError: Si password está vacío
            TypeError: Si password no es string
        """
        if not isinstance(password, str):
            raise TypeError(f"password must be a string, got {type(password)}")
        if not password:
            raise ValueError("password cannot be empty")
        
        try:
            return self.pwd_context.hash(password)
        except Exception as e:
            logger.error(f"Error hashing password: {e}", exc_info=True)
            raise ValueError(f"Failed to hash password: {e}") from e
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verifica contraseña contra hash.
        
        Args:
            plain_password: Contraseña en texto plano (debe ser string no vacío)
            hashed_password: Hash de la contraseña (debe ser string no vacío)
            
        Returns:
            True si la contraseña coincide, False en caso contrario
            
        Raises:
            ValueError: Si plain_password o hashed_password están vacíos
            TypeError: Si los parámetros no son strings
        """
        if not isinstance(plain_password, str):
            raise TypeError(f"plain_password must be a string, got {type(plain_password)}")
        if not plain_password:
            raise ValueError("plain_password cannot be empty")
        
        if not isinstance(hashed_password, str):
            raise TypeError(f"hashed_password must be a string, got {type(hashed_password)}")
        if not hashed_password:
            raise ValueError("hashed_password cannot be empty")
        
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.warning(f"Error verifying password: {e}")
            return False

