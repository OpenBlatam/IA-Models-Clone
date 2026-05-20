"""
Servicio de Autenticación y Autorización.
"""

import hashlib
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class Permission(str, Enum):
    """Permisos del sistema."""
    READ_TASKS = "read:tasks"
    WRITE_TASKS = "write:tasks"
    DELETE_TASKS = "delete:tasks"
    READ_AGENT = "read:agent"
    CONTROL_AGENT = "control:agent"
    READ_AUDIT = "read:audit"
    READ_MONITORING = "read:monitoring"
    ADMIN = "admin"


class Role(str, Enum):
    """Roles del sistema."""
    USER = "user"
    ADMIN = "admin"
    VIEWER = "viewer"


class User:
    """
    Representa un usuario con validaciones.
    
    Attributes:
        user_id: ID único del usuario
        username: Nombre de usuario
        email: Email (opcional)
        role: Rol del usuario
        permissions: Permisos adicionales
        created_at: Fecha de creación
        last_login: Última fecha de login
    """
    
    def __init__(
        self,
        user_id: str,
        username: str,
        email: Optional[str] = None,
        role: Role = Role.USER,
        permissions: Optional[List[Permission]] = None
    ):
        """
        Inicializar usuario con validaciones.
        
        Args:
            user_id: ID único del usuario (debe ser string no vacío)
            username: Nombre de usuario (debe ser string no vacío)
            email: Email (opcional, debe ser string válido si se proporciona)
            role: Rol del usuario (debe ser Role)
            permissions: Permisos adicionales (opcional, debe ser lista de Permission)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"user_id debe ser un string no vacío, recibido: {user_id}")
        
        if not username or not isinstance(username, str) or not username.strip():
            raise ValueError(f"username debe ser un string no vacío, recibido: {username}")
        
        if email is not None:
            if not isinstance(email, str) or not email.strip():
                raise ValueError(f"email debe ser un string no vacío si se proporciona, recibido: {email}")
            # Validación básica de formato de email
            if "@" not in email or "." not in email.split("@")[-1]:
                raise ValueError(f"email debe tener un formato válido, recibido: {email}")
        
        if not isinstance(role, Role):
            raise ValueError(f"role debe ser un Role, recibido: {type(role)}")
        
        if permissions is not None:
            if not isinstance(permissions, list):
                raise ValueError(f"permissions debe ser una lista, recibido: {type(permissions)}")
            for perm in permissions:
                if not isinstance(perm, Permission):
                    raise ValueError(f"Todos los permisos deben ser Permission, recibido: {type(perm)}")
        
        self.user_id = user_id.strip()
        self.username = username.strip()
        self.email = email.strip() if email else None
        self.role = role
        self.permissions = permissions or []
        self.created_at = datetime.now()
        self.last_login: Optional[datetime] = None
        
        logger.debug(f"Usuario creado: {self.user_id} ({self.username}, role: {self.role.value})")
    
    def has_permission(self, permission: Permission) -> bool:
        """
        Verificar si el usuario tiene un permiso.
        
        Args:
            permission: Permiso a verificar
            
        Returns:
            True si tiene el permiso
        """
        if self.role == Role.ADMIN or Permission.ADMIN in self.permissions:
            return True
        
        return permission in self.permissions or self._role_has_permission(permission)
    
    def _role_has_permission(self, permission: Permission) -> bool:
        """Verificar si el rol tiene el permiso."""
        role_permissions = {
            Role.USER: [
                Permission.READ_TASKS,
                Permission.WRITE_TASKS,
                Permission.READ_AGENT
            ],
            Role.VIEWER: [
                Permission.READ_TASKS,
                Permission.READ_AGENT
            ],
            Role.ADMIN: list(Permission)
        }
        return permission in role_permissions.get(self.role, [])


class APIKey:
    """Representa una API key."""
    
    def __init__(
        self,
        key_id: str,
        key_hash: str,
        user_id: str,
        name: str,
        permissions: List[Permission],
        expires_at: Optional[datetime] = None
    ):
        """
        Inicializar API key.
        
        Args:
            key_id: ID único de la key
            key_hash: Hash de la key
            user_id: ID del usuario propietario
            name: Nombre descriptivo
            permissions: Permisos de la key
            expires_at: Fecha de expiración (opcional)
        """
        self.key_id = key_id
        self.key_hash = key_hash
        self.user_id = user_id
        self.name = name
        self.permissions = permissions
        self.expires_at = expires_at
        self.created_at = datetime.now()
        self.last_used: Optional[datetime] = None
        self.usage_count = 0
    
    def is_expired(self) -> bool:
        """Verificar si la key está expirada."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def has_permission(self, permission: Permission) -> bool:
        """Verificar si la key tiene un permiso."""
        return permission in self.permissions or Permission.ADMIN in self.permissions


class AuthService:
    """
    Servicio de autenticación y autorización con mejoras.
    
    Attributes:
        users: Diccionario de usuarios por ID
        api_keys: Diccionario de API keys por ID
        key_to_id: Mapeo de hash de key a ID de key
    """
    
    def __init__(self):
        """Inicializar servicio con logging."""
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.key_to_id: Dict[str, str] = {}  # key_hash -> key_id
        
        logger.info("✅ AuthService inicializado")
    
    def create_user(
        self,
        username: str,
        email: Optional[str] = None,
        role: Role = Role.USER
    ) -> User:
        """
        Crear nuevo usuario con validaciones.
        
        Args:
            username: Nombre de usuario (debe ser string no vacío)
            email: Email (opcional)
            role: Rol del usuario (debe ser Role)
            
        Returns:
            Usuario creado
            
        Raises:
            ValueError: Si los parámetros son inválidos o el usuario ya existe
        """
        # Validaciones básicas (User.__init__ también valida)
        if not username or not isinstance(username, str) or not username.strip():
            raise ValueError(f"username debe ser un string no vacío, recibido: {username}")
        
        if not isinstance(role, Role):
            raise ValueError(f"role debe ser un Role, recibido: {type(role)}")
        
        username = username.strip()
        
        # Verificar si el usuario ya existe (por username)
        for existing_user in self.users.values():
            if existing_user.username == username:
                raise ValueError(f"Usuario con username '{username}' ya existe")
        
        try:
            user_id = f"user_{secrets.token_urlsafe(16)}"
            user = User(user_id, username, email, role)
            self.users[user_id] = user
            logger.info(
                f"✅ Usuario creado: {user_id} ({username}, role: {role.value}, "
                f"email: {email or 'N/A'})"
            )
            return user
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error al crear usuario: {e}", exc_info=True)
            raise ValueError(f"Error al crear usuario: {e}") from e
    
    def get_user(self, user_id: str) -> Optional[User]:
        """
        Obtener usuario por ID.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Usuario o None
        """
        return self.users.get(user_id)
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: List[Permission],
        expires_in_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """
        Crear nueva API key con validaciones.
        
        Args:
            user_id: ID del usuario propietario (debe ser string no vacío)
            name: Nombre descriptivo (debe ser string no vacío)
            permissions: Permisos de la key (debe ser lista no vacía de Permission)
            expires_in_days: Días hasta expiración (opcional, debe ser entero positivo)
            
        Returns:
            Tupla (key_plain, APIKey)
            
        Raises:
            ValueError: Si los parámetros son inválidos o el usuario no existe
        """
        # Validaciones
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"user_id debe ser un string no vacío, recibido: {user_id}")
        
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(f"name debe ser un string no vacío, recibido: {name}")
        
        if not permissions or not isinstance(permissions, list) or len(permissions) == 0:
            raise ValueError("permissions debe ser una lista no vacía de Permission")
        
        for perm in permissions:
            if not isinstance(perm, Permission):
                raise ValueError(f"Todos los permisos deben ser Permission, recibido: {type(perm)}")
        
        if expires_in_days is not None:
            if not isinstance(expires_in_days, int) or expires_in_days < 1:
                raise ValueError(f"expires_in_days debe ser un entero positivo, recibido: {expires_in_days}")
        
        user_id = user_id.strip()
        name = name.strip()
        
        # Verificar que el usuario existe
        if user_id not in self.users:
            raise ValueError(f"Usuario con ID '{user_id}' no existe")
        
        try:
            # Generar key
            key_plain = f"sk_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(key_plain.encode()).hexdigest()
            
            key_id = f"key_{secrets.token_urlsafe(16)}"
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)
            
            api_key = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                user_id=user_id,
                name=name,
                permissions=permissions,
                expires_at=expires_at
            )
            
            self.api_keys[key_id] = api_key
            self.key_to_id[key_hash] = key_id
            
            logger.info(
                f"✅ API key creada: {key_id} para usuario {user_id} "
                f"({name}, {len(permissions)} permisos, "
                f"expira: {expires_at.isoformat() if expires_at else 'nunca'})"
            )
            return key_plain, api_key
        except Exception as e:
            logger.error(f"Error al crear API key: {e}", exc_info=True)
            raise ValueError(f"Error al crear API key: {e}") from e
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """
        Validar API key con validaciones.
        
        Args:
            api_key: API key a validar (debe ser string no vacío)
        
        Returns:
            APIKey si es válida, None si no
            
        Raises:
            ValueError: Si api_key es inválido
        """
        # Validación
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(f"api_key debe ser un string no vacío, recibido: {api_key}")
        
        api_key = api_key.strip()
        
        # Validar formato básico (debe empezar con sk_)
        if not api_key.startswith("sk_"):
            logger.warning(f"API key con formato inválido (no empieza con 'sk_'): {api_key[:10]}...")
            return None
        
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            key_id = self.key_to_id.get(key_hash)
            
            if not key_id:
                logger.debug(f"API key no encontrada (hash: {key_hash[:16]}...)")
                return None
            
            api_key_obj = self.api_keys.get(key_id)
            if not api_key_obj:
                logger.warning(f"API key object no encontrado para key_id: {key_id}")
                return None
            
            if api_key_obj.is_expired():
                logger.warning(f"API key expirada: {key_id} (expired_at: {api_key_obj.expires_at})")
                return None
            
            # Actualizar uso
            api_key_obj.last_used = datetime.now()
            api_key_obj.usage_count += 1
            
            logger.debug(
                f"✅ API key validada: {key_id} (usos: {api_key_obj.usage_count})"
            )
            
            return api_key_obj
        except Exception as e:
            logger.error(f"Error al validar API key: {e}", exc_info=True)
            return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revocar API key.
        
        Args:
            key_id: ID de la key a revocar
            
        Returns:
            True si se revocó exitosamente
        """
        if key_id in self.api_keys:
            api_key = self.api_keys[key_id]
            del self.api_keys[key_id]
            del self.key_to_id[api_key.key_hash]
            logger.info(f"API key revocada: {key_id}")
            return True
        return False
    
    def check_permission(
        self,
        api_key: Optional[APIKey],
        permission: Permission
    ) -> bool:
        """
        Verificar permiso.
        
        Args:
            api_key: API key (opcional)
            permission: Permiso a verificar
            
        Returns:
            True si tiene el permiso
        """
        if not api_key:
            return False
        
        return api_key.has_permission(permission)
    
    def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """
        Obtener API keys de un usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Lista de API keys
        """
        return [
            key for key in self.api_keys.values()
            if key.user_id == user_id
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "total_users": len(self.users),
            "total_api_keys": len(self.api_keys),
            "active_api_keys": len([k for k in self.api_keys.values() if not k.is_expired()]),
            "expired_api_keys": len([k for k in self.api_keys.values() if k.is_expired()])
        }
