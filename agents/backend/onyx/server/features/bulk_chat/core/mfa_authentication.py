"""
MFA Authentication - Autenticación Multi-Factor
==============================================

Sistema de autenticación multi-factor con TOTP, SMS, Email y backup codes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import secrets
import hashlib
import base64

logger = logging.getLogger(__name__)

try:
    import pyotp
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False
    logger.warning("pyotp not available, TOTP features will be limited")


class MFAMethod(Enum):
    """Método de MFA."""
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BACKUP_CODE = "backup_code"


class MFAStatus(Enum):
    """Estado de MFA."""
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class MFASession:
    """Sesión de MFA."""
    session_id: str
    user_id: str
    method: MFAMethod
    status: MFAStatus
    created_at: datetime
    expires_at: datetime
    verified_at: Optional[datetime] = None
    attempts: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MFACredential:
    """Credencial MFA del usuario."""
    user_id: str
    totp_secret: Optional[str] = None
    totp_enabled: bool = False
    sms_phone: Optional[str] = None
    sms_enabled: bool = False
    email: Optional[str] = None
    email_enabled: bool = False
    backup_codes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class MFAAuthentication:
    """Sistema de autenticación multi-factor."""
    
    def __init__(self):
        self.credentials: Dict[str, MFACredential] = {}
        self.sessions: Dict[str, MFASession] = {}
        self._lock = asyncio.Lock()
    
    def generate_totp_secret(self) -> str:
        """Generar secreto TOTP."""
        if TOTP_AVAILABLE:
            return pyotp.random_base32()
        else:
            # Fallback: generar secreto aleatorio
            return base64.b32encode(secrets.token_bytes(20)).decode()
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generar códigos de respaldo."""
        return [secrets.token_hex(4).upper() for _ in range(count)]
    
    def setup_totp(self, user_id: str) -> Dict[str, Any]:
        """Configurar TOTP para usuario."""
        secret = self.generate_totp_secret()
        
        async def save_credential():
            async with self._lock:
                if user_id not in self.credentials:
                    self.credentials[user_id] = MFACredential(user_id=user_id)
                
                credential = self.credentials[user_id]
                credential.totp_secret = secret
                credential.totp_enabled = True
                credential.updated_at = datetime.now()
        
        asyncio.create_task(save_credential())
        
        # Generar QR code URI (en producción, usar librería de QR)
        if TOTP_AVAILABLE:
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=user_id,
                issuer_name="Bulk Chat",
            )
        else:
            provisioning_uri = None
        
        return {
            "secret": secret,
            "provisioning_uri": provisioning_uri,
            "backup_codes": self.generate_backup_codes(),
        }
    
    def setup_sms(self, user_id: str, phone: str) -> bool:
        """Configurar SMS para usuario."""
        async def save_credential():
            async with self._lock:
                if user_id not in self.credentials:
                    self.credentials[user_id] = MFACredential(user_id=user_id)
                
                credential = self.credentials[user_id]
                credential.sms_phone = phone
                credential.sms_enabled = True
                credential.updated_at = datetime.now()
        
        asyncio.create_task(save_credential())
        return True
    
    def setup_email(self, user_id: str, email: str) -> bool:
        """Configurar Email para usuario."""
        async def save_credential():
            async with self._lock:
                if user_id not in self.credentials:
                    self.credentials[user_id] = MFACredential(user_id=user_id)
                
                credential = self.credentials[user_id]
                credential.email = email
                credential.email_enabled = True
                credential.updated_at = datetime.now()
        
        asyncio.create_task(save_credential())
        return True
    
    async def initiate_mfa(
        self,
        user_id: str,
        method: MFAMethod,
    ) -> str:
        """Iniciar proceso de MFA."""
        credential = self.credentials.get(user_id)
        if not credential:
            raise ValueError(f"No MFA credentials found for user {user_id}")
        
        # Verificar que el método está habilitado
        if method == MFAMethod.TOTP and not credential.totp_enabled:
            raise ValueError("TOTP not enabled")
        elif method == MFAMethod.SMS and not credential.sms_enabled:
            raise ValueError("SMS not enabled")
        elif method == MFAMethod.EMAIL and not credential.email_enabled:
            raise ValueError("Email not enabled")
        
        session_id = f"mfa_{user_id}_{datetime.now().timestamp()}"
        
        session = MFASession(
            session_id=session_id,
            user_id=user_id,
            method=method,
            status=MFAStatus.PENDING,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5),
        )
        
        async with self._lock:
            self.sessions[session_id] = session
        
        # Enviar código según método
        if method == MFAMethod.SMS:
            # En producción, enviar SMS
            logger.info(f"Would send SMS code to {credential.sms_phone}")
        elif method == MFAMethod.EMAIL:
            # En producción, enviar email
            logger.info(f"Would send email code to {credential.email}")
        
        return session_id
    
    def verify_totp(self, user_id: str, code: str) -> bool:
        """Verificar código TOTP."""
        credential = self.credentials.get(user_id)
        if not credential or not credential.totp_secret:
            return False
        
        if TOTP_AVAILABLE:
            totp = pyotp.TOTP(credential.totp_secret)
            return totp.verify(code)
        else:
            # Fallback: verificación básica
            return False
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verificar código de respaldo."""
        credential = self.credentials.get(user_id)
        if not credential:
            return False
        
        code_upper = code.upper()
        if code_upper in credential.backup_codes:
            # Remover código usado
            credential.backup_codes.remove(code_upper)
            credential.updated_at = datetime.now()
            return True
        
        return False
    
    async def verify_mfa(
        self,
        session_id: str,
        code: str,
    ) -> bool:
        """Verificar código MFA."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Verificar expiración
        if datetime.now() > session.expires_at:
            session.status = MFAStatus.EXPIRED
            return False
        
        # Verificar intentos
        if session.attempts >= session.max_attempts:
            session.status = MFAStatus.FAILED
            return False
        
        session.attempts += 1
        
        # Verificar según método
        if session.method == MFAMethod.TOTP:
            verified = self.verify_totp(session.user_id, code)
        elif session.method == MFAMethod.BACKUP_CODE:
            verified = self.verify_backup_code(session.user_id, code)
        else:
            # SMS y Email: verificación básica (en producción, validar código enviado)
            verified = len(code) == 6 and code.isdigit()
        
        if verified:
            session.status = MFAStatus.VERIFIED
            session.verified_at = datetime.now()
            return True
        else:
            if session.attempts >= session.max_attempts:
                session.status = MFAStatus.FAILED
        
        return False
    
    def get_user_mfa_status(self, user_id: str) -> Dict[str, Any]:
        """Obtener estado MFA del usuario."""
        credential = self.credentials.get(user_id)
        if not credential:
            return {
                "mfa_enabled": False,
                "methods": [],
            }
        
        methods = []
        if credential.totp_enabled:
            methods.append("totp")
        if credential.sms_enabled:
            methods.append("sms")
        if credential.email_enabled:
            methods.append("email")
        
        return {
            "mfa_enabled": len(methods) > 0,
            "methods": methods,
            "backup_codes_count": len(credential.backup_codes),
        }














