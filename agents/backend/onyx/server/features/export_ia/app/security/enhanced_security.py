"""
Enhanced Security - Sistema de seguridad avanzado
"""

import asyncio
import logging
import hashlib
import secrets
import jwt
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Niveles de seguridad."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ENTERPRISE = "enterprise"


class ThreatType(Enum):
    """Tipos de amenazas."""
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


@dataclass
class SecurityEvent:
    """Evento de seguridad."""
    event_id: str
    threat_type: ThreatType
    severity: str
    source_ip: str
    user_agent: str
    endpoint: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False


@dataclass
class UserSession:
    """Sesión de usuario."""
    session_id: str
    user_id: str
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    is_active: bool = True
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedSecurity:
    """
    Sistema de seguridad avanzado.
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """Inicializar sistema de seguridad."""
        self.security_level = security_level
        self.blocked_ips: set = set()
        self.suspicious_ips: Dict[str, List[datetime]] = defaultdict(list)
        self.user_sessions: Dict[str, UserSession] = {}
        self.security_events: List[SecurityEvent] = []
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque())
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Configuración de seguridad
        self.jwt_secret = secrets.token_urlsafe(32)
        self.jwt_algorithm = "HS256"
        self.session_timeout = timedelta(hours=24)
        self.max_failed_attempts = 5
        self.rate_limit_window = 60  # segundos
        self.max_requests_per_window = 100
        
        # Configuración por nivel
        self._configure_by_level()
        
        logger.info(f"EnhancedSecurity inicializado con nivel {security_level.value}")
    
    def _configure_by_level(self):
        """Configurar seguridad según el nivel."""
        if self.security_level == SecurityLevel.LOW:
            self.max_failed_attempts = 10
            self.max_requests_per_window = 200
            self.session_timeout = timedelta(hours=48)
        elif self.security_level == SecurityLevel.MEDIUM:
            self.max_failed_attempts = 5
            self.max_requests_per_window = 100
            self.session_timeout = timedelta(hours=24)
        elif self.security_level == SecurityLevel.HIGH:
            self.max_failed_attempts = 3
            self.max_requests_per_window = 50
            self.session_timeout = timedelta(hours=12)
        elif self.security_level == SecurityLevel.ENTERPRISE:
            self.max_failed_attempts = 3
            self.max_requests_per_window = 30
            self.session_timeout = timedelta(hours=8)
    
    async def initialize(self):
        """Inicializar el sistema de seguridad."""
        try:
            # Limpiar sesiones expiradas
            await self._cleanup_expired_sessions()
            
            # Configurar limpieza automática
            asyncio.create_task(self._periodic_cleanup())
            
            logger.info("EnhancedSecurity inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar EnhancedSecurity: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el sistema de seguridad."""
        try:
            # Limpiar todas las sesiones
            self.user_sessions.clear()
            
            logger.info("EnhancedSecurity cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar EnhancedSecurity: {e}")
    
    async def _periodic_cleanup(self):
        """Limpieza periódica de datos de seguridad."""
        while True:
            try:
                await self._cleanup_expired_sessions()
                await self._cleanup_old_events()
                await self._cleanup_rate_limits()
                await asyncio.sleep(300)  # Cada 5 minutos
            except Exception as e:
                logger.error(f"Error en limpieza periódica: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_expired_sessions(self):
        """Limpiar sesiones expiradas."""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.user_sessions.items():
            if now - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.user_sessions[session_id]
    
    async def _cleanup_old_events(self):
        """Limpiar eventos de seguridad antiguos."""
        cutoff_time = datetime.now() - timedelta(days=30)
        self.security_events = [
            event for event in self.security_events
            if event.timestamp > cutoff_time
        ]
    
    async def _cleanup_rate_limits(self):
        """Limpiar límites de tasa antiguos."""
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self.rate_limit_window)
        
        for ip, requests in self.rate_limits.items():
            while requests and requests[0] < cutoff_time:
                requests.popleft()
    
    async def check_rate_limit(self, ip_address: str) -> bool:
        """Verificar límite de tasa."""
        try:
            now = datetime.now()
            cutoff_time = now - timedelta(seconds=self.rate_limit_window)
            
            # Limpiar requests antiguos
            requests = self.rate_limits[ip_address]
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Verificar límite
            if len(requests) >= self.max_requests_per_window:
                await self._log_security_event(
                    threat_type=ThreatType.DDoS,
                    severity="high",
                    source_ip=ip_address,
                    endpoint="rate_limit",
                    details={"requests_count": len(requests)}
                )
                return False
            
            # Agregar request actual
            requests.append(now)
            return True
            
        except Exception as e:
            logger.error(f"Error al verificar límite de tasa: {e}")
            return False
    
    async def check_brute_force(self, ip_address: str, success: bool) -> bool:
        """Verificar intentos de fuerza bruta."""
        try:
            now = datetime.now()
            cutoff_time = now - timedelta(minutes=15)
            
            if success:
                # Limpiar intentos fallidos en caso de éxito
                self.failed_attempts[ip_address] = [
                    attempt for attempt in self.failed_attempts[ip_address]
                    if attempt > cutoff_time
                ]
                return True
            
            # Agregar intento fallido
            self.failed_attempts[ip_address].append(now)
            
            # Limpiar intentos antiguos
            self.failed_attempts[ip_address] = [
                attempt for attempt in self.failed_attempts[ip_address]
                if attempt > cutoff_time
            ]
            
            # Verificar si excede el límite
            if len(self.failed_attempts[ip_address]) >= self.max_failed_attempts:
                await self._log_security_event(
                    threat_type=ThreatType.BRUTE_FORCE,
                    severity="high",
                    source_ip=ip_address,
                    endpoint="authentication",
                    details={"failed_attempts": len(self.failed_attempts[ip_address])}
                )
                
                # Bloquear IP temporalmente
                await self.block_ip(ip_address, duration_minutes=30)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error al verificar fuerza bruta: {e}")
            return True
    
    async def block_ip(self, ip_address: str, duration_minutes: int = 60):
        """Bloquear dirección IP."""
        try:
            self.blocked_ips.add(ip_address)
            
            # Programar desbloqueo
            asyncio.create_task(self._unblock_ip_after(ip_address, duration_minutes))
            
            await self._log_security_event(
                threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                severity="medium",
                source_ip=ip_address,
                endpoint="ip_blocking",
                details={"duration_minutes": duration_minutes}
            )
            
            logger.warning(f"IP {ip_address} bloqueada por {duration_minutes} minutos")
            
        except Exception as e:
            logger.error(f"Error al bloquear IP: {e}")
    
    async def _unblock_ip_after(self, ip_address: str, duration_minutes: int):
        """Desbloquear IP después de un tiempo."""
        await asyncio.sleep(duration_minutes * 60)
        self.blocked_ips.discard(ip_address)
        logger.info(f"IP {ip_address} desbloqueada")
    
    async def is_ip_blocked(self, ip_address: str) -> bool:
        """Verificar si una IP está bloqueada."""
        return ip_address in self.blocked_ips
    
    async def validate_input(self, input_data: str, input_type: str = "general") -> bool:
        """Validar entrada de usuario."""
        try:
            # Patrones de inyección SQL
            sql_patterns = [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
                r"(--|\#|\/\*|\*\/)",
                r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\b)"
            ]
            
            # Patrones XSS
            xss_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>"
            ]
            
            # Verificar patrones maliciosos
            import re
            
            for pattern in sql_patterns + xss_patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    return False
            
            # Verificar longitud
            if len(input_data) > 10000:  # Límite de caracteres
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error al validar entrada: {e}")
            return False
    
    async def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        permissions: List[str] = None
    ) -> str:
        """Crear sesión de usuario."""
        try:
            session_id = str(uuid.uuid4())
            now = datetime.now()
            
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=now,
                last_activity=now,
                permissions=permissions or []
            )
            
            self.user_sessions[session_id] = session
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error al crear sesión: {e}")
            return ""
    
    async def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validar sesión de usuario."""
        try:
            if session_id not in self.user_sessions:
                return None
            
            session = self.user_sessions[session_id]
            
            # Verificar si la sesión ha expirado
            if datetime.now() - session.last_activity > self.session_timeout:
                del self.user_sessions[session_id]
                return None
            
            # Actualizar última actividad
            session.last_activity = datetime.now()
            
            return session
            
        except Exception as e:
            logger.error(f"Error al validar sesión: {e}")
            return None
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidar sesión de usuario."""
        try:
            if session_id in self.user_sessions:
                del self.user_sessions[session_id]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error al invalidar sesión: {e}")
            return False
    
    async def generate_jwt_token(
        self,
        user_id: str,
        permissions: List[str] = None,
        expires_in_hours: int = 24
    ) -> str:
        """Generar token JWT."""
        try:
            now = datetime.utcnow()
            payload = {
                "user_id": user_id,
                "permissions": permissions or [],
                "iat": now,
                "exp": now + timedelta(hours=expires_in_hours),
                "jti": str(uuid.uuid4())
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            return token
            
        except Exception as e:
            logger.error(f"Error al generar token JWT: {e}")
            return ""
    
    async def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validar token JWT."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token JWT expirado")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token JWT inválido")
            return None
        except Exception as e:
            logger.error(f"Error al validar token JWT: {e}")
            return None
    
    async def hash_password(self, password: str) -> str:
        """Hashear contraseña."""
        try:
            salt = secrets.token_hex(16)
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return f"{salt}:{password_hash.hex()}"
            
        except Exception as e:
            logger.error(f"Error al hashear contraseña: {e}")
            return ""
    
    async def verify_password(self, password: str, password_hash: str) -> bool:
        """Verificar contraseña."""
        try:
            if ':' not in password_hash:
                return False
            
            salt, hash_part = password_hash.split(':', 1)
            password_hash_verify = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            
            return password_hash_verify.hex() == hash_part
            
        except Exception as e:
            logger.error(f"Error al verificar contraseña: {e}")
            return False
    
    async def _log_security_event(
        self,
        threat_type: ThreatType,
        severity: str,
        source_ip: str,
        endpoint: str,
        details: Dict[str, Any] = None
    ):
        """Registrar evento de seguridad."""
        try:
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                threat_type=threat_type,
                severity=severity,
                source_ip=source_ip,
                user_agent="",  # Se puede agregar si está disponible
                endpoint=endpoint,
                timestamp=datetime.now(),
                details=details or {}
            )
            
            self.security_events.append(event)
            
            # Log del evento
            logger.warning(
                f"SECURITY EVENT [{severity.upper()}] {threat_type.value}: "
                f"IP {source_ip} - {endpoint} - {details}"
            )
            
        except Exception as e:
            logger.error(f"Error al registrar evento de seguridad: {e}")
    
    async def get_security_events(
        self,
        threat_type: Optional[ThreatType] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener eventos de seguridad."""
        try:
            events = self.security_events
            
            # Filtrar por tipo de amenaza
            if threat_type:
                events = [e for e in events if e.threat_type == threat_type]
            
            # Filtrar por severidad
            if severity:
                events = [e for e in events if e.severity == severity]
            
            # Ordenar por timestamp (más recientes primero)
            events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            events = events[:limit]
            
            return [
                {
                    "event_id": e.event_id,
                    "threat_type": e.threat_type.value,
                    "severity": e.severity,
                    "source_ip": e.source_ip,
                    "endpoint": e.endpoint,
                    "timestamp": e.timestamp.isoformat(),
                    "details": e.details,
                    "blocked": e.blocked
                }
                for e in events
            ]
            
        except Exception as e:
            logger.error(f"Error al obtener eventos de seguridad: {e}")
            return []
    
    async def get_security_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de seguridad."""
        try:
            # Contar eventos por tipo
            events_by_type = defaultdict(int)
            events_by_severity = defaultdict(int)
            
            for event in self.security_events:
                events_by_type[event.threat_type.value] += 1
                events_by_severity[event.severity] += 1
            
            # Estadísticas de sesiones
            active_sessions = len([s for s in self.user_sessions.values() if s.is_active])
            
            # Estadísticas de IPs bloqueadas
            blocked_ips_count = len(self.blocked_ips)
            
            return {
                "security_level": self.security_level.value,
                "blocked_ips": blocked_ips_count,
                "active_sessions": active_sessions,
                "total_sessions": len(self.user_sessions),
                "security_events": {
                    "total": len(self.security_events),
                    "by_type": dict(events_by_type),
                    "by_severity": dict(events_by_severity)
                },
                "rate_limits": {
                    "tracked_ips": len(self.rate_limits),
                    "max_requests_per_window": self.max_requests_per_window,
                    "window_seconds": self.rate_limit_window
                },
                "failed_attempts": {
                    "tracked_ips": len(self.failed_attempts),
                    "max_attempts": self.max_failed_attempts
                },
                "session_config": {
                    "timeout_hours": self.session_timeout.total_seconds() / 3600,
                    "max_sessions": len(self.user_sessions)
                }
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estadísticas de seguridad: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del sistema de seguridad."""
        try:
            return {
                "status": "healthy",
                "security_level": self.security_level.value,
                "blocked_ips": len(self.blocked_ips),
                "active_sessions": len(self.user_sessions),
                "security_events": len(self.security_events),
                "rate_limits_tracked": len(self.rate_limits),
                "failed_attempts_tracked": len(self.failed_attempts),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de seguridad: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




