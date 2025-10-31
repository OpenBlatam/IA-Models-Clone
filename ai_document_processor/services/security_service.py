"""
Servicio de Seguridad
====================

Servicio para manejo de seguridad, autenticación y autorización.
"""

import asyncio
import logging
import hashlib
import secrets
import jwt
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class SecurityLevel(str, Enum):
    """Niveles de seguridad"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class UserRole(str, Enum):
    """Roles de usuario"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    API_USER = "api_user"

@dataclass
class User:
    """Usuario del sistema"""
    id: str
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class SecurityEvent:
    """Evento de seguridad"""
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    timestamp: datetime
    severity: SecurityLevel

@dataclass
class FileSecurityCheck:
    """Verificación de seguridad de archivo"""
    filename: str
    file_size: int
    file_type: str
    is_safe: bool
    threats_detected: List[str]
    security_score: float
    recommendations: List[str]

class SecurityService:
    """Servicio de seguridad"""
    
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
        
        # Configuración de seguridad
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "52428800"))  # 50MB
        self.allowed_extensions = os.getenv("ALLOWED_EXTENSIONS", ".md,.pdf,.docx,.doc,.txt").split(",")
        self.forbidden_extensions = [".exe", ".bat", ".cmd", ".scr", ".pif", ".com", ".vbs", ".js", ".jar"]
        
        # Usuarios en memoria (en producción usar base de datos)
        self.users: Dict[str, User] = {}
        self.security_events: List[SecurityEvent] = []
        
        # Configuración de rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
        
    async def initialize(self):
        """Inicializa el servicio de seguridad"""
        logger.info("Inicializando servicio de seguridad...")
        
        # Crear usuario admin por defecto
        await self._create_default_admin()
        
        # Configurar políticas de seguridad
        await self._setup_security_policies()
        
        logger.info("Servicio de seguridad inicializado")
    
    async def _create_default_admin(self):
        """Crea usuario admin por defecto"""
        try:
            admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
            admin_user = User(
                id="admin",
                username="admin",
                email="admin@example.com",
                role=UserRole.ADMIN,
                is_active=True,
                created_at=datetime.now()
            )
            
            # Hash de la contraseña
            password_hash = self._hash_password(admin_password)
            
            # En un sistema real, guardar en base de datos
            self.users["admin"] = admin_user
            
            logger.info("Usuario admin creado por defecto")
            
        except Exception as e:
            logger.error(f"Error creando usuario admin: {e}")
    
    async def _setup_security_policies(self):
        """Configura políticas de seguridad"""
        try:
            # Políticas de archivos
            self.file_policies = {
                "max_size": self.max_file_size,
                "allowed_extensions": self.allowed_extensions,
                "forbidden_extensions": self.forbidden_extensions,
                "scan_content": True,
                "quarantine_suspicious": True
            }
            
            # Políticas de autenticación
            self.auth_policies = {
                "max_login_attempts": 5,
                "lockout_duration_minutes": 30,
                "password_min_length": 8,
                "require_special_chars": True,
                "session_timeout_minutes": self.jwt_expiration
            }
            
            logger.info("Políticas de seguridad configuradas")
            
        except Exception as e:
            logger.error(f"Error configurando políticas de seguridad: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Genera hash de contraseña"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verifica contraseña"""
        try:
            salt, hash_hex = password_hash.split(':')
            password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_hash_check.hex() == hash_hex
        except Exception:
            return False
    
    async def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> Tuple[bool, Optional[str], Optional[User]]:
        """Autentica un usuario"""
        try:
            # Buscar usuario
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user:
                await self._log_security_event(
                    "failed_login",
                    None,
                    ip_address,
                    user_agent,
                    {"username": username, "reason": "user_not_found"},
                    SecurityLevel.MEDIUM
                )
                return False, "Usuario no encontrado", None
            
            # Verificar si el usuario está bloqueado
            if user.locked_until and user.locked_until > datetime.now():
                await self._log_security_event(
                    "failed_login",
                    user.id,
                    ip_address,
                    user_agent,
                    {"reason": "account_locked"},
                    SecurityLevel.HIGH
                )
                return False, "Cuenta bloqueada temporalmente", None
            
            # Verificar si el usuario está activo
            if not user.is_active:
                await self._log_security_event(
                    "failed_login",
                    user.id,
                    ip_address,
                    user_agent,
                    {"reason": "account_inactive"},
                    SecurityLevel.MEDIUM
                )
                return False, "Cuenta inactiva", None
            
            # Verificar contraseña (simplificado para demo)
            if password != "admin123" and user.username == "admin":
                user.failed_login_attempts += 1
                
                # Bloquear cuenta si excede intentos
                if user.failed_login_attempts >= self.auth_policies["max_login_attempts"]:
                    user.locked_until = datetime.now() + timedelta(minutes=self.auth_policies["lockout_duration_minutes"])
                    await self._log_security_event(
                        "account_locked",
                        user.id,
                        ip_address,
                        user_agent,
                        {"failed_attempts": user.failed_login_attempts},
                        SecurityLevel.HIGH
                    )
                    return False, "Cuenta bloqueada por múltiples intentos fallidos", None
                
                await self._log_security_event(
                    "failed_login",
                    user.id,
                    ip_address,
                    user_agent,
                    {"failed_attempts": user.failed_login_attempts},
                    SecurityLevel.MEDIUM
                )
                return False, "Contraseña incorrecta", None
            
            # Login exitoso
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now()
            
            # Generar token JWT
            token = self._generate_jwt_token(user)
            
            await self._log_security_event(
                "successful_login",
                user.id,
                ip_address,
                user_agent,
                {"login_time": datetime.now().isoformat()},
                SecurityLevel.LOW
            )
            
            return True, token, user
            
        except Exception as e:
            logger.error(f"Error autenticando usuario: {e}")
            return False, "Error interno de autenticación", None
    
    def _generate_jwt_token(self, user: User) -> str:
        """Genera token JWT"""
        try:
            payload = {
                "user_id": user.id,
                "username": user.username,
                "role": user.role.value,
                "exp": datetime.utcnow() + timedelta(minutes=self.jwt_expiration),
                "iat": datetime.utcnow()
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            return token
            
        except Exception as e:
            logger.error(f"Error generando token JWT: {e}")
            return ""
    
    async def verify_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verifica token JWT"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Verificar si el usuario existe y está activo
            user_id = payload.get("user_id")
            if user_id in self.users:
                user = self.users[user_id]
                if user.is_active:
                    return True, payload
            
            return False, None
            
        except jwt.ExpiredSignatureError:
            return False, None
        except jwt.InvalidTokenError:
            return False, None
        except Exception as e:
            logger.error(f"Error verificando token JWT: {e}")
            return False, None
    
    async def check_rate_limit(self, ip_address: str) -> Tuple[bool, int]:
        """Verifica rate limiting"""
        try:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            
            # Limpiar requests antiguos
            if ip_address in self.rate_limits:
                self.rate_limits[ip_address] = [
                    req_time for req_time in self.rate_limits[ip_address]
                    if req_time > minute_ago
                ]
            else:
                self.rate_limits[ip_address] = []
            
            # Verificar límite
            current_requests = len(self.rate_limits[ip_address])
            if current_requests >= self.max_requests_per_minute:
                await self._log_security_event(
                    "rate_limit_exceeded",
                    None,
                    ip_address,
                    "",
                    {"requests": current_requests, "limit": self.max_requests_per_minute},
                    SecurityLevel.MEDIUM
                )
                return False, current_requests
            
            # Agregar request actual
            self.rate_limits[ip_address].append(now)
            return True, current_requests + 1
            
        except Exception as e:
            logger.error(f"Error verificando rate limit: {e}")
            return True, 0
    
    async def scan_file_security(self, filename: str, file_size: int, file_content: bytes = None) -> FileSecurityCheck:
        """Escanea archivo por amenazas de seguridad"""
        try:
            threats_detected = []
            recommendations = []
            security_score = 100.0
            
            # Verificar extensión
            file_extension = Path(filename).suffix.lower()
            
            if file_extension in self.forbidden_extensions:
                threats_detected.append(f"Extensión prohibida: {file_extension}")
                security_score -= 50.0
                recommendations.append("No se permiten archivos ejecutables")
            
            if file_extension not in self.allowed_extensions:
                threats_detected.append(f"Extensión no permitida: {file_extension}")
                security_score -= 30.0
                recommendations.append("Usar solo extensiones permitidas")
            
            # Verificar tamaño
            if file_size > self.max_file_size:
                threats_detected.append(f"Archivo demasiado grande: {file_size} bytes")
                security_score -= 40.0
                recommendations.append(f"Reducir tamaño del archivo (máximo {self.max_file_size} bytes)")
            
            # Verificar contenido si está disponible
            if file_content:
                content_str = file_content.decode('utf-8', errors='ignore').lower()
                
                # Patrones sospechosos
                suspicious_patterns = [
                    r'<script[^>]*>.*?</script>',
                    r'javascript:',
                    r'vbscript:',
                    r'onload\s*=',
                    r'onerror\s*=',
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'system\s*\(',
                    r'cmd\s*\/',
                    r'powershell',
                    r'wget\s+',
                    r'curl\s+'
                ]
                
                for pattern in suspicious_patterns:
                    if re.search(pattern, content_str):
                        threats_detected.append(f"Patrón sospechoso detectado: {pattern}")
                        security_score -= 20.0
                        recommendations.append("Revisar contenido del archivo")
                
                # Verificar archivos binarios en archivos de texto
                if file_extension in ['.txt', '.md'] and len(file_content) > 0:
                    null_bytes = file_content.count(b'\x00')
                    if null_bytes > 10:
                        threats_detected.append("Contenido binario en archivo de texto")
                        security_score -= 15.0
                        recommendations.append("Verificar que el archivo sea realmente de texto")
            
            # Determinar si es seguro
            is_safe = security_score >= 70.0 and len(threats_detected) == 0
            
            return FileSecurityCheck(
                filename=filename,
                file_size=file_size,
                file_type=file_extension,
                is_safe=is_safe,
                threats_detected=threats_detected,
                security_score=max(0.0, security_score),
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error escaneando seguridad de archivo: {e}")
            return FileSecurityCheck(
                filename=filename,
                file_size=file_size,
                file_type="unknown",
                is_safe=False,
                threats_detected=["Error en escaneo de seguridad"],
                security_score=0.0,
                recommendations=["Contactar administrador"]
            )
    
    async def _log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        ip_address: str,
        user_agent: str,
        details: Dict[str, Any],
        severity: SecurityLevel
    ):
        """Registra evento de seguridad"""
        try:
            event = SecurityEvent(
                event_type=event_type,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
                timestamp=datetime.now(),
                severity=severity
            )
            
            self.security_events.append(event)
            
            # Mantener solo los últimos 1000 eventos
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]
            
            # Log según severidad
            if severity == SecurityLevel.CRITICAL:
                logger.critical(f"SECURITY EVENT [{severity.value}]: {event_type} - {details}")
            elif severity == SecurityLevel.HIGH:
                logger.warning(f"SECURITY EVENT [{severity.value}]: {event_type} - {details}")
            else:
                logger.info(f"SECURITY EVENT [{severity.value}]: {event_type} - {details}")
            
        except Exception as e:
            logger.error(f"Error registrando evento de seguridad: {e}")
    
    async def get_security_events(self, limit: int = 100, severity: Optional[SecurityLevel] = None) -> List[SecurityEvent]:
        """Obtiene eventos de seguridad"""
        try:
            events = self.security_events.copy()
            
            # Filtrar por severidad si se especifica
            if severity:
                events = [e for e in events if e.severity == severity]
            
            # Ordenar por timestamp (más recientes primero)
            events.sort(key=lambda x: x.timestamp, reverse=True)
            
            return events[:limit]
            
        except Exception as e:
            logger.error(f"Error obteniendo eventos de seguridad: {e}")
            return []
    
    async def get_security_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de seguridad"""
        try:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            last_7d = now - timedelta(days=7)
            
            # Filtrar eventos por período
            events_24h = [e for e in self.security_events if e.timestamp > last_24h]
            events_7d = [e for e in self.security_events if e.timestamp > last_7d]
            
            # Contar por severidad
            def count_by_severity(events):
                counts = {level.value: 0 for level in SecurityLevel}
                for event in events:
                    counts[event.severity.value] += 1
                return counts
            
            # Contar por tipo de evento
            def count_by_type(events):
                counts = {}
                for event in events:
                    counts[event.event_type] = counts.get(event.event_type, 0) + 1
                return counts
            
            return {
                "total_events": len(self.security_events),
                "events_24h": len(events_24h),
                "events_7d": len(events_7d),
                "severity_counts_24h": count_by_severity(events_24h),
                "severity_counts_7d": count_by_severity(events_7d),
                "event_types_24h": count_by_type(events_24h),
                "event_types_7d": count_by_type(events_7d),
                "active_users": len([u for u in self.users.values() if u.is_active]),
                "locked_users": len([u for u in self.users.values() if u.locked_until and u.locked_until > now]),
                "rate_limited_ips": len([ip for ip, times in self.rate_limits.items() if len(times) >= self.max_requests_per_minute]),
                "last_updated": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen de seguridad: {e}")
            return {"error": str(e)}
    
    async def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> Tuple[bool, str]:
        """Crea un nuevo usuario"""
        try:
            # Verificar si el usuario ya existe
            for user in self.users.values():
                if user.username == username or user.email == email:
                    return False, "Usuario o email ya existe"
            
            # Validar contraseña
            if len(password) < self.auth_policies["password_min_length"]:
                return False, f"Contraseña debe tener al menos {self.auth_policies['password_min_length']} caracteres"
            
            # Crear usuario
            user_id = secrets.token_hex(8)
            user = User(
                id=user_id,
                username=username,
                email=email,
                role=role,
                is_active=True,
                created_at=datetime.now()
            )
            
            self.users[user_id] = user
            
            await self._log_security_event(
                "user_created",
                user_id,
                "system",
                "system",
                {"username": username, "role": role.value},
                SecurityLevel.LOW
            )
            
            return True, "Usuario creado exitosamente"
            
        except Exception as e:
            logger.error(f"Error creando usuario: {e}")
            return False, "Error interno creando usuario"
    
    async def update_user_role(self, user_id: str, new_role: UserRole, admin_user_id: str) -> Tuple[bool, str]:
        """Actualiza el rol de un usuario"""
        try:
            if user_id not in self.users:
                return False, "Usuario no encontrado"
            
            old_role = self.users[user_id].role
            self.users[user_id].role = new_role
            
            await self._log_security_event(
                "role_updated",
                admin_user_id,
                "system",
                "system",
                {"target_user": user_id, "old_role": old_role.value, "new_role": new_role.value},
                SecurityLevel.MEDIUM
            )
            
            return True, "Rol actualizado exitosamente"
            
        except Exception as e:
            logger.error(f"Error actualizando rol: {e}")
            return False, "Error interno actualizando rol"


