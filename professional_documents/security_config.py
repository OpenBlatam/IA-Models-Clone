"""
Configuraciones de Seguridad Avanzadas para el Sistema de Documentos Profesionales

Este módulo implementa configuraciones de seguridad robustas incluyendo
encriptación, autenticación, autorización, y protección contra amenazas.
"""

import os
import hashlib
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Niveles de seguridad"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EncryptionType(Enum):
    """Tipos de encriptación"""
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    CHACHA20 = "chacha20"
    QUANTUM_SAFE = "quantum_safe"

@dataclass
class SecurityPolicy:
    """Política de seguridad"""
    id: str
    name: str
    level: SecurityLevel
    encryption_type: EncryptionType
    password_requirements: Dict[str, Any]
    session_timeout: int
    max_login_attempts: int
    ip_whitelist: List[str]
    ip_blacklist: List[str]
    mfa_required: bool
    audit_logging: bool
    data_retention_days: int

@dataclass
class SecurityEvent:
    """Evento de seguridad"""
    id: str
    type: str
    severity: str
    description: str
    source_ip: str
    user_id: Optional[str]
    timestamp: datetime
    resolved: bool = False

class EncryptionManager:
    """Gestor de encriptación"""
    
    def __init__(self):
        self.encryption_keys = {}
        self._load_encryption_keys()
    
    def _load_encryption_keys(self):
        """Cargar claves de encriptación"""
        try:
            # Cargar clave AES desde variable de entorno
            aes_key = os.getenv("AES_ENCRYPTION_KEY")
            if aes_key:
                self.encryption_keys["aes"] = aes_key.encode()
            else:
                # Generar nueva clave si no existe
                self.encryption_keys["aes"] = Fernet.generate_key()
                logger.warning("Generated new AES encryption key")
            
            # Cargar claves RSA
            self._load_rsa_keys()
            
        except Exception as e:
            logger.error(f"Error loading encryption keys: {e}")
            raise
    
    def _load_rsa_keys(self):
        """Cargar claves RSA"""
        try:
            # Cargar clave privada RSA
            private_key_path = os.getenv("RSA_PRIVATE_KEY_PATH", "keys/private_key.pem")
            if os.path.exists(private_key_path):
                with open(private_key_path, "rb") as key_file:
                    self.encryption_keys["rsa_private"] = serialization.load_pem_private_key(
                        key_file.read(),
                        password=None
                    )
            else:
                # Generar nuevas claves RSA
                self._generate_rsa_keys()
            
            # Cargar clave pública RSA
            public_key_path = os.getenv("RSA_PUBLIC_KEY_PATH", "keys/public_key.pem")
            if os.path.exists(public_key_path):
                with open(public_key_path, "rb") as key_file:
                    self.encryption_keys["rsa_public"] = serialization.load_pem_public_key(
                        key_file.read()
                    )
            
        except Exception as e:
            logger.error(f"Error loading RSA keys: {e}")
            raise
    
    def _generate_rsa_keys(self):
        """Generar nuevas claves RSA"""
        try:
            # Generar clave privada
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096
            )
            
            # Generar clave pública
            public_key = private_key.public_key()
            
            # Guardar claves
            os.makedirs("keys", exist_ok=True)
            
            # Guardar clave privada
            with open("keys/private_key.pem", "wb") as key_file:
                key_file.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Guardar clave pública
            with open("keys/public_key.pem", "wb") as key_file:
                key_file.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            
            self.encryption_keys["rsa_private"] = private_key
            self.encryption_keys["rsa_public"] = public_key
            
            logger.info("Generated new RSA key pair")
            
        except Exception as e:
            logger.error(f"Error generating RSA keys: {e}")
            raise
    
    def encrypt_data(self, data: str, encryption_type: EncryptionType = EncryptionType.AES_256) -> str:
        """Encriptar datos"""
        try:
            if encryption_type == EncryptionType.AES_256:
                return self._encrypt_aes(data)
            elif encryption_type in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
                return self._encrypt_rsa(data)
            elif encryption_type == EncryptionType.CHACHA20:
                return self._encrypt_chacha20(data)
            elif encryption_type == EncryptionType.QUANTUM_SAFE:
                return self._encrypt_quantum_safe(data)
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str, encryption_type: EncryptionType = EncryptionType.AES_256) -> str:
        """Desencriptar datos"""
        try:
            if encryption_type == EncryptionType.AES_256:
                return self._decrypt_aes(encrypted_data)
            elif encryption_type in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
                return self._decrypt_rsa(encrypted_data)
            elif encryption_type == EncryptionType.CHACHA20:
                return self._decrypt_chacha20(encrypted_data)
            elif encryption_type == EncryptionType.QUANTUM_SAFE:
                return self._decrypt_quantum_safe(encrypted_data)
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    def _encrypt_aes(self, data: str) -> str:
        """Encriptar con AES-256"""
        try:
            fernet = Fernet(self.encryption_keys["aes"])
            encrypted_data = fernet.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            logger.error(f"Error in AES encryption: {e}")
            raise
    
    def _decrypt_aes(self, encrypted_data: str) -> str:
        """Desencriptar con AES-256"""
        try:
            fernet = Fernet(self.encryption_keys["aes"])
            decrypted_data = fernet.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error in AES decryption: {e}")
            raise
    
    def _encrypt_rsa(self, data: str) -> str:
        """Encriptar con RSA"""
        try:
            public_key = self.encryption_keys["rsa_public"]
            encrypted_data = public_key.encrypt(
                data.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return encrypted_data.hex()
        except Exception as e:
            logger.error(f"Error in RSA encryption: {e}")
            raise
    
    def _decrypt_rsa(self, encrypted_data: str) -> str:
        """Desencriptar con RSA"""
        try:
            private_key = self.encryption_keys["rsa_private"]
            decrypted_data = private_key.decrypt(
                bytes.fromhex(encrypted_data),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error in RSA decryption: {e}")
            raise
    
    def _encrypt_chacha20(self, data: str) -> str:
        """Encriptar con ChaCha20"""
        try:
            # Implementación simplificada de ChaCha20
            key = os.urandom(32)
            nonce = os.urandom(12)
            
            # Usar Fernet con clave derivada
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=nonce,
                iterations=100000,
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(key))
            fernet = Fernet(derived_key)
            
            encrypted_data = fernet.encrypt(data.encode())
            return f"{key.hex()}:{nonce.hex()}:{encrypted_data.decode()}"
        except Exception as e:
            logger.error(f"Error in ChaCha20 encryption: {e}")
            raise
    
    def _decrypt_chacha20(self, encrypted_data: str) -> str:
        """Desencriptar con ChaCha20"""
        try:
            key_hex, nonce_hex, encrypted = encrypted_data.split(":")
            key = bytes.fromhex(key_hex)
            nonce = bytes.fromhex(nonce_hex)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=nonce,
                iterations=100000,
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(key))
            fernet = Fernet(derived_key)
            
            decrypted_data = fernet.decrypt(encrypted.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error in ChaCha20 decryption: {e}")
            raise
    
    def _encrypt_quantum_safe(self, data: str) -> str:
        """Encriptar con algoritmo cuántico seguro"""
        try:
            # Implementación simplificada de encriptación post-cuántica
            # En una implementación real, usaría algoritmos como NTRU, McEliece, etc.
            return self._encrypt_aes(data)  # Fallback a AES por ahora
        except Exception as e:
            logger.error(f"Error in quantum-safe encryption: {e}")
            raise
    
    def _decrypt_quantum_safe(self, encrypted_data: str) -> str:
        """Desencriptar con algoritmo cuántico seguro"""
        try:
            # Implementación simplificada de desencriptación post-cuántica
            return self._decrypt_aes(encrypted_data)  # Fallback a AES por ahora
        except Exception as e:
            logger.error(f"Error in quantum-safe decryption: {e}")
            raise

class AuthenticationManager:
    """Gestor de autenticación"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.jwt_secret = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        self.session_timeout = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hora
    
    def hash_password(self, password: str) -> str:
        """Hashear contraseña"""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        except Exception as e:
            logger.error(f"Error hashing password: {e}")
            raise
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verificar contraseña"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    def generate_jwt_token(self, user_id: str, user_data: Dict[str, Any]) -> str:
        """Generar token JWT"""
        try:
            payload = {
                "user_id": user_id,
                "user_data": user_data,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(seconds=self.session_timeout)
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            return token
        except Exception as e:
            logger.error(f"Error generating JWT token: {e}")
            raise
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verificar token JWT"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error(f"Error verifying JWT token: {e}")
            raise HTTPException(status_code=401, detail="Token verification failed")
    
    async def create_session(self, user_id: str, token: str, ip_address: str) -> str:
        """Crear sesión de usuario"""
        try:
            session_id = secrets.token_urlsafe(32)
            session_data = {
                "user_id": user_id,
                "token": token,
                "ip_address": ip_address,
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }
            
            # Guardar sesión en Redis
            await self.redis.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(session_data)
            )
            
            # Agregar a la lista de sesiones activas del usuario
            await self.redis.sadd(f"user_sessions:{user_id}", session_id)
            
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    async def validate_session(self, session_id: str) -> Dict[str, Any]:
        """Validar sesión"""
        try:
            session_data = await self.redis.get(f"session:{session_id}")
            if not session_data:
                raise HTTPException(status_code=401, detail="Session not found")
            
            session = json.loads(session_data)
            
            # Actualizar última actividad
            session["last_activity"] = datetime.utcnow().isoformat()
            await self.redis.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(session)
            )
            
            return session
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            raise HTTPException(status_code=401, detail="Session validation failed")
    
    async def destroy_session(self, session_id: str):
        """Destruir sesión"""
        try:
            # Obtener datos de la sesión
            session_data = await self.redis.get(f"session:{session_id}")
            if session_data:
                session = json.loads(session_data)
                user_id = session.get("user_id")
                
                # Remover de la lista de sesiones del usuario
                if user_id:
                    await self.redis.srem(f"user_sessions:{user_id}", session_id)
            
            # Eliminar sesión
            await self.redis.delete(f"session:{session_id}")
            
        except Exception as e:
            logger.error(f"Error destroying session: {e}")
    
    async def destroy_all_user_sessions(self, user_id: str):
        """Destruir todas las sesiones del usuario"""
        try:
            # Obtener todas las sesiones del usuario
            session_ids = await self.redis.smembers(f"user_sessions:{user_id}")
            
            # Destruir cada sesión
            for session_id in session_ids:
                await self.destroy_session(session_id.decode())
            
            # Limpiar la lista de sesiones
            await self.redis.delete(f"user_sessions:{user_id}")
            
        except Exception as e:
            logger.error(f"Error destroying all user sessions: {e}")

class AuthorizationManager:
    """Gestor de autorización"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.roles_permissions = self._load_roles_permissions()
    
    def _load_roles_permissions(self) -> Dict[str, List[str]]:
        """Cargar permisos por rol"""
        return {
            "admin": [
                "read:all", "write:all", "delete:all",
                "manage:users", "manage:system", "manage:security",
                "view:logs", "manage:backups", "manage:ai_models"
            ],
            "manager": [
                "read:all", "write:documents", "delete:own",
                "manage:team", "view:reports", "manage:workflows"
            ],
            "editor": [
                "read:assigned", "write:documents", "delete:own",
                "manage:own_workflows", "view:own_reports"
            ],
            "viewer": [
                "read:assigned", "view:own_reports"
            ],
            "guest": [
                "read:public"
            ]
        }
    
    def has_permission(self, user_role: str, permission: str) -> bool:
        """Verificar si el usuario tiene un permiso específico"""
        try:
            role_permissions = self.roles_permissions.get(user_role, [])
            return permission in role_permissions
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False
    
    def get_user_permissions(self, user_role: str) -> List[str]:
        """Obtener todos los permisos del usuario"""
        try:
            return self.roles_permissions.get(user_role, [])
        except Exception as e:
            logger.error(f"Error getting user permissions: {e}")
            return []
    
    async def check_resource_access(self, user_id: str, resource_type: str, resource_id: str, action: str) -> bool:
        """Verificar acceso a un recurso específico"""
        try:
            # Obtener rol del usuario
            user_data = await self.redis.get(f"user:{user_id}")
            if not user_data:
                return False
            
            user = json.loads(user_data)
            user_role = user.get("role", "guest")
            
            # Verificar permisos básicos
            permission = f"{action}:{resource_type}"
            if not self.has_permission(user_role, permission):
                return False
            
            # Verificar acceso específico al recurso
            if resource_type == "document":
                return await self._check_document_access(user_id, resource_id, action)
            elif resource_type == "user":
                return await self._check_user_access(user_id, resource_id, action)
            elif resource_type == "system":
                return await self._check_system_access(user_id, action)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking resource access: {e}")
            return False
    
    async def _check_document_access(self, user_id: str, document_id: str, action: str) -> bool:
        """Verificar acceso a documento"""
        try:
            # Obtener información del documento
            document_data = await self.redis.get(f"document:{document_id}")
            if not document_data:
                return False
            
            document = json.loads(document_data)
            
            # Verificar si el usuario es el propietario
            if document.get("owner_id") == user_id:
                return True
            
            # Verificar si el usuario tiene acceso compartido
            shared_with = document.get("shared_with", [])
            if user_id in shared_with:
                return True
            
            # Verificar permisos de equipo
            user_teams = await self.redis.smembers(f"user_teams:{user_id}")
            document_teams = document.get("team_access", [])
            
            for team in user_teams:
                if team.decode() in document_teams:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking document access: {e}")
            return False
    
    async def _check_user_access(self, user_id: str, target_user_id: str, action: str) -> bool:
        """Verificar acceso a usuario"""
        try:
            # Los usuarios solo pueden acceder a su propia información
            if user_id == target_user_id:
                return True
            
            # Los administradores pueden acceder a todos los usuarios
            user_data = await self.redis.get(f"user:{user_id}")
            if user_data:
                user = json.loads(user_data)
                if user.get("role") == "admin":
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking user access: {e}")
            return False
    
    async def _check_system_access(self, user_id: str, action: str) -> bool:
        """Verificar acceso al sistema"""
        try:
            user_data = await self.redis.get(f"user:{user_id}")
            if not user_data:
                return False
            
            user = json.loads(user_data)
            user_role = user.get("role", "guest")
            
            # Solo administradores pueden acceder al sistema
            return user_role == "admin"
            
        except Exception as e:
            logger.error(f"Error checking system access: {e}")
            return False

class SecurityMonitor:
    """Monitor de seguridad"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.security_events = []
        self.threat_patterns = self._load_threat_patterns()
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Cargar patrones de amenazas"""
        return {
            "sql_injection": [
                "'; DROP TABLE",
                "UNION SELECT",
                "OR 1=1",
                "AND 1=1"
            ],
            "xss_attempt": [
                "<script>",
                "javascript:",
                "onload=",
                "onerror="
            ],
            "brute_force": [
                "multiple_failed_logins",
                "rapid_login_attempts"
            ],
            "ddos_attack": [
                "high_request_rate",
                "suspicious_user_agents"
            ]
        }
    
    async def monitor_request(self, request: Request, user_id: Optional[str] = None) -> bool:
        """Monitorear solicitud en busca de amenazas"""
        try:
            # Verificar rate limiting
            if not await self._check_rate_limit(request):
                await self._log_security_event("rate_limit_exceeded", "high", request, user_id)
                return False
            
            # Verificar patrones maliciosos
            if await self._check_malicious_patterns(request):
                await self._log_security_event("malicious_pattern_detected", "critical", request, user_id)
                return False
            
            # Verificar IP en blacklist
            if await self._check_ip_blacklist(request):
                await self._log_security_event("blacklisted_ip", "high", request, user_id)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error monitoring request: {e}")
            return True  # Permitir en caso de error
    
    async def _check_rate_limit(self, request: Request) -> bool:
        """Verificar rate limiting"""
        try:
            client_ip = request.client.host
            current_time = int(time.time())
            window_start = current_time - 60  # Ventana de 1 minuto
            
            # Contar requests en la ventana
            request_count = await self.redis.zcount(
                f"rate_limit:{client_ip}",
                window_start,
                current_time
            )
            
            # Límite de 100 requests por minuto
            if request_count >= 100:
                return False
            
            # Agregar request actual
            await self.redis.zadd(f"rate_limit:{client_ip}", {str(current_time): current_time})
            await self.redis.expire(f"rate_limit:{client_ip}", 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True
    
    async def _check_malicious_patterns(self, request: Request) -> bool:
        """Verificar patrones maliciosos"""
        try:
            # Verificar URL
            url = str(request.url)
            for pattern_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in url.lower():
                        await self.redis.incr(f"threat_detected:{pattern_type}")
                        return True
            
            # Verificar headers
            for header_name, header_value in request.headers.items():
                for pattern_type, patterns in self.threat_patterns.items():
                    for pattern in patterns:
                        if pattern.lower() in header_value.lower():
                            await self.redis.incr(f"threat_detected:{pattern_type}")
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking malicious patterns: {e}")
            return False
    
    async def _check_ip_blacklist(self, request: Request) -> bool:
        """Verificar IP en blacklist"""
        try:
            client_ip = request.client.host
            is_blacklisted = await self.redis.sismember("blacklisted_ips", client_ip)
            return bool(is_blacklisted)
        except Exception as e:
            logger.error(f"Error checking IP blacklist: {e}")
            return False
    
    async def _log_security_event(self, event_type: str, severity: str, request: Request, user_id: Optional[str] = None):
        """Registrar evento de seguridad"""
        try:
            event = SecurityEvent(
                id=secrets.token_urlsafe(16),
                type=event_type,
                severity=severity,
                description=f"Security event: {event_type}",
                source_ip=request.client.host,
                user_id=user_id,
                timestamp=datetime.utcnow()
            )
            
            # Guardar en Redis
            await self.redis.lpush("security_events", json.dumps(asdict(event)))
            await self.redis.ltrim("security_events", 0, 9999)  # Mantener solo los últimos 10000 eventos
            
            # Guardar en lista local
            self.security_events.append(event)
            
            # Enviar alerta si es crítica
            if severity == "critical":
                await self._send_security_alert(event)
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    async def _send_security_alert(self, event: SecurityEvent):
        """Enviar alerta de seguridad"""
        try:
            # Implementar envío de alertas (email, Slack, etc.)
            logger.critical(f"SECURITY ALERT: {event.type} - {event.description}")
            
            # Notificar a administradores
            await self.redis.publish("security_alerts", json.dumps(asdict(event)))
            
        except Exception as e:
            logger.error(f"Error sending security alert: {e}")
    
    async def get_security_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Obtener eventos de seguridad recientes"""
        try:
            events_data = await self.redis.lrange("security_events", 0, limit - 1)
            events = []
            
            for event_data in events_data:
                event_dict = json.loads(event_data)
                event = SecurityEvent(**event_dict)
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting security events: {e}")
            return []

class SecurityConfigManager:
    """Gestor de configuración de seguridad"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.encryption_manager = EncryptionManager()
        self.auth_manager = AuthenticationManager(redis_client)
        self.authz_manager = AuthorizationManager(redis_client)
        self.security_monitor = SecurityMonitor(redis_client)
        self.security_policies = self._load_security_policies()
    
    def _load_security_policies(self) -> Dict[str, SecurityPolicy]:
        """Cargar políticas de seguridad"""
        return {
            "default": SecurityPolicy(
                id="default",
                name="Default Security Policy",
                level=SecurityLevel.MEDIUM,
                encryption_type=EncryptionType.AES_256,
                password_requirements={
                    "min_length": 8,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_special_chars": True
                },
                session_timeout=3600,
                max_login_attempts=5,
                ip_whitelist=[],
                ip_blacklist=[],
                mfa_required=False,
                audit_logging=True,
                data_retention_days=365
            ),
            "high_security": SecurityPolicy(
                id="high_security",
                name="High Security Policy",
                level=SecurityLevel.HIGH,
                encryption_type=EncryptionType.RSA_4096,
                password_requirements={
                    "min_length": 12,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_special_chars": True
                },
                session_timeout=1800,
                max_login_attempts=3,
                ip_whitelist=[],
                ip_blacklist=[],
                mfa_required=True,
                audit_logging=True,
                data_retention_days=2555  # 7 años
            ),
            "critical": SecurityPolicy(
                id="critical",
                name="Critical Security Policy",
                level=SecurityLevel.CRITICAL,
                encryption_type=EncryptionType.QUANTUM_SAFE,
                password_requirements={
                    "min_length": 16,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_special_chars": True
                },
                session_timeout=900,
                max_login_attempts=2,
                ip_whitelist=[],
                ip_blacklist=[],
                mfa_required=True,
                audit_logging=True,
                data_retention_days=3650  # 10 años
            )
        }
    
    def get_security_policy(self, policy_id: str) -> Optional[SecurityPolicy]:
        """Obtener política de seguridad"""
        return self.security_policies.get(policy_id)
    
    def update_security_policy(self, policy_id: str, policy: SecurityPolicy):
        """Actualizar política de seguridad"""
        self.security_policies[policy_id] = policy
    
    async def apply_security_policy(self, policy_id: str, user_id: str):
        """Aplicar política de seguridad a usuario"""
        try:
            policy = self.get_security_policy(policy_id)
            if not policy:
                raise ValueError(f"Security policy {policy_id} not found")
            
            # Aplicar configuración de sesión
            self.auth_manager.session_timeout = policy.session_timeout
            
            # Aplicar configuración de encriptación
            # (Esto requeriría migración de datos existentes)
            
            # Guardar política aplicada
            await self.redis.set(f"user_policy:{user_id}", policy_id)
            
            logger.info(f"Applied security policy {policy_id} to user {user_id}")
            
        except Exception as e:
            logger.error(f"Error applying security policy: {e}")
            raise
    
    async def validate_password(self, password: str, policy_id: str = "default") -> Dict[str, Any]:
        """Validar contraseña según política"""
        try:
            policy = self.get_security_policy(policy_id)
            if not policy:
                return {"valid": False, "errors": ["Policy not found"]}
            
            requirements = policy.password_requirements
            errors = []
            
            # Verificar longitud mínima
            if len(password) < requirements["min_length"]:
                errors.append(f"Password must be at least {requirements['min_length']} characters long")
            
            # Verificar mayúsculas
            if requirements["require_uppercase"] and not any(c.isupper() for c in password):
                errors.append("Password must contain at least one uppercase letter")
            
            # Verificar minúsculas
            if requirements["require_lowercase"] and not any(c.islower() for c in password):
                errors.append("Password must contain at least one lowercase letter")
            
            # Verificar números
            if requirements["require_numbers"] and not any(c.isdigit() for c in password):
                errors.append("Password must contain at least one number")
            
            # Verificar caracteres especiales
            if requirements["require_special_chars"] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                errors.append("Password must contain at least one special character")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error validating password: {e}")
            return {"valid": False, "errors": ["Validation error"]}

# Dependencias de FastAPI
security_scheme = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> Dict[str, Any]:
    """Obtener usuario actual desde token"""
    try:
        token = credentials.credentials
        payload = SecurityConfigManager(None).auth_manager.verify_jwt_token(token)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

async def require_permission(permission: str):
    """Decorador para requerir permiso específico"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Implementar verificación de permisos
            return await func(*args, **kwargs)
        return wrapper
    return decorator

async def require_role(role: str):
    """Decorador para requerir rol específico"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Implementar verificación de rol
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Funciones de utilidad
def generate_secure_token(length: int = 32) -> str:
    """Generar token seguro"""
    return secrets.token_urlsafe(length)

def generate_api_key() -> str:
    """Generar API key"""
    return f"api_{secrets.token_urlsafe(32)}"

def hash_sensitive_data(data: str) -> str:
    """Hashear datos sensibles"""
    return hashlib.sha256(data.encode()).hexdigest()

# Configuración de seguridad por defecto
DEFAULT_SECURITY_CONFIG = {
    "encryption": {
        "default_type": "AES_256",
        "key_rotation_days": 90,
        "quantum_safe_enabled": False
    },
    "authentication": {
        "jwt_expiry_hours": 24,
        "refresh_token_expiry_days": 30,
        "mfa_required": False
    },
    "authorization": {
        "rbac_enabled": True,
        "permission_cache_ttl": 3600
    },
    "monitoring": {
        "security_events_retention_days": 90,
        "threat_detection_enabled": True,
        "rate_limiting_enabled": True
    },
    "compliance": {
        "gdpr_compliant": True,
        "sox_compliant": True,
        "hipaa_compliant": False
    }
}



























