"""
Advanced Security - Seguridad avanzada (2FA, SSO, OAuth)
=========================================================
"""

import logging
import hashlib
import secrets
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import pyotp
import qrcode
import io
import base64

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Métodos de autenticación"""
    PASSWORD = "password"
    TWO_FACTOR = "two_factor"
    SSO = "sso"
    OAUTH = "oauth"
    API_KEY = "api_key"


class SecurityLevel(Enum):
    """Niveles de seguridad"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TwoFactorAuth:
    """Sistema de autenticación de dos factores"""
    
    def __init__(self):
        self.user_secrets: Dict[str, str] = {}  # user_id -> secret
        self.backup_codes: Dict[str, List[str]] = {}  # user_id -> backup codes
    
    def generate_secret(self, user_id: str) -> str:
        """Genera un secreto para 2FA"""
        secret = pyotp.random_base32()
        self.user_secrets[user_id] = secret
        return secret
    
    def generate_qr_code(self, user_id: str, email: str, issuer: str = "Code Improver") -> str:
        """Genera un código QR para 2FA"""
        if user_id not in self.user_secrets:
            self.generate_secret(user_id)
        
        secret = self.user_secrets[user_id]
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=email,
            issuer_name=issuer
        )
        
        # Generar QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def verify_token(self, user_id: str, token: str) -> bool:
        """Verifica un token 2FA"""
        if user_id not in self.user_secrets:
            return False
        
        secret = self.user_secrets[user_id]
        totp = pyotp.TOTP(secret)
        
        # Verificar token actual y los 2 anteriores/posteriores (ventana de tiempo)
        return totp.verify(token, valid_window=2)
    
    def generate_backup_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Genera códigos de respaldo"""
        codes = [secrets.token_hex(4).upper() for _ in range(count)]
        self.backup_codes[user_id] = codes
        return codes
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verifica un código de respaldo"""
        if user_id not in self.backup_codes:
            return False
        
        codes = self.backup_codes[user_id]
        if code.upper() in codes:
            codes.remove(code.upper())
            return True
        return False


class SSOProvider:
    """Proveedor SSO"""
    
    def __init__(self, provider_id: str, config: Dict[str, Any]):
        self.provider_id = provider_id
        self.config = config
        self.enabled = config.get("enabled", False)
    
    def get_auth_url(self, redirect_uri: str, state: str) -> str:
        """Genera URL de autenticación SSO"""
        # Implementación específica según proveedor (OAuth 2.0, SAML, etc.)
        return f"{self.config.get('auth_url')}?redirect_uri={redirect_uri}&state={state}"
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica un token SSO y retorna información del usuario"""
        # Implementación específica según proveedor
        return None


class OAuthManager:
    """Gestor de OAuth"""
    
    def __init__(self):
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.authorization_codes: Dict[str, Dict[str, Any]] = {}  # code -> data
        self.access_tokens: Dict[str, Dict[str, Any]] = {}  # token -> data
    
    def register_provider(
        self,
        provider_id: str,
        client_id: str,
        client_secret: str,
        auth_url: str,
        token_url: str,
        user_info_url: str
    ):
        """Registra un proveedor OAuth"""
        self.providers[provider_id] = {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_url": auth_url,
            "token_url": token_url,
            "user_info_url": user_info_url
        }
    
    def generate_authorization_url(
        self,
        provider_id: str,
        redirect_uri: str,
        scopes: List[str],
        state: str
    ) -> str:
        """Genera URL de autorización OAuth"""
        if provider_id not in self.providers:
            raise ValueError(f"Proveedor {provider_id} no registrado")
        
        provider = self.providers[provider_id]
        params = {
            "client_id": provider["client_id"],
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "response_type": "code",
            "state": state
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{provider['auth_url']}?{query_string}"
    
    def exchange_code_for_token(
        self,
        provider_id: str,
        code: str,
        redirect_uri: str
    ) -> Dict[str, Any]:
        """Intercambia código de autorización por token"""
        if provider_id not in self.providers:
            raise ValueError(f"Proveedor {provider_id} no registrado")
        
        # En producción, hacer request HTTP real
        # Por ahora, simulamos
        token = secrets.token_urlsafe(32)
        expires_in = 3600
        
        self.access_tokens[token] = {
            "provider_id": provider_id,
            "expires_at": datetime.now() + timedelta(seconds=expires_in),
            "scopes": []
        }
        
        return {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": expires_in
        }
    
    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica un token de acceso"""
        if token not in self.access_tokens:
            return None
        
        token_data = self.access_tokens[token]
        if token_data["expires_at"] < datetime.now():
            del self.access_tokens[token]
            return None
        
        return token_data


class AdvancedSecurity:
    """Sistema de seguridad avanzado"""
    
    def __init__(self):
        self.two_factor = TwoFactorAuth()
        self.sso_providers: Dict[str, SSOProvider] = {}
        self.oauth_manager = OAuthManager()
        self.failed_attempts: Dict[str, List[datetime]] = {}  # user_id -> attempts
        self.locked_accounts: Dict[str, datetime] = {}  # user_id -> lock_until
        self.password_history: Dict[str, List[str]] = {}  # user_id -> password hashes
        self.session_tokens: Dict[str, Dict[str, Any]] = {}  # token -> session data
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hashea una contraseña"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        return password_hash, salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verifica una contraseña"""
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == password_hash
    
    def check_password_strength(self, password: str) -> Dict[str, Any]:
        """Verifica la fortaleza de una contraseña"""
        score = 0
        feedback = []
        
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("La contraseña debe tener al menos 8 caracteres")
        
        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("Agregar letras mayúsculas")
        
        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("Agregar letras minúsculas")
        
        if any(c.isdigit() for c in password):
            score += 1
        else:
            feedback.append("Agregar números")
        
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        else:
            feedback.append("Agregar caracteres especiales")
        
        if len(password) >= 12:
            score += 1
        
        strength = "weak" if score < 3 else "medium" if score < 5 else "strong"
        
        return {
            "strength": strength,
            "score": score,
            "max_score": 6,
            "feedback": feedback
        }
    
    def register_sso_provider(
        self,
        provider_id: str,
        config: Dict[str, Any]
    ) -> SSOProvider:
        """Registra un proveedor SSO"""
        provider = SSOProvider(provider_id, config)
        self.sso_providers[provider_id] = provider
        return provider
    
    def get_sso_auth_url(
        self,
        provider_id: str,
        redirect_uri: str,
        state: str
    ) -> str:
        """Obtiene URL de autenticación SSO"""
        if provider_id not in self.sso_providers:
            raise ValueError(f"Proveedor SSO {provider_id} no encontrado")
        
        provider = self.sso_providers[provider_id]
        return provider.get_auth_url(redirect_uri, state)
    
    def record_failed_login(self, user_id: str):
        """Registra un intento de login fallido"""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        self.failed_attempts[user_id].append(datetime.now())
        
        # Limpiar intentos antiguos (últimos 15 minutos)
        cutoff = datetime.now() - timedelta(minutes=15)
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > cutoff
        ]
        
        # Bloquear cuenta después de 5 intentos fallidos
        if len(self.failed_attempts[user_id]) >= 5:
            self.locked_accounts[user_id] = datetime.now() + timedelta(minutes=30)
            logger.warning(f"Cuenta {user_id} bloqueada por múltiples intentos fallidos")
    
    def is_account_locked(self, user_id: str) -> bool:
        """Verifica si una cuenta está bloqueada"""
        if user_id not in self.locked_accounts:
            return False
        
        if self.locked_accounts[user_id] < datetime.now():
            del self.locked_accounts[user_id]
            return False
        
        return True
    
    def create_session(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Crea una nueva sesión"""
        token = secrets.token_urlsafe(32)
        self.session_tokens[token] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24),
            "ip_address": ip_address,
            "user_agent": user_agent,
            "last_activity": datetime.now()
        }
        return token
    
    def verify_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica una sesión"""
        if token not in self.session_tokens:
            return None
        
        session = self.session_tokens[token]
        if session["expires_at"] < datetime.now():
            del self.session_tokens[token]
            return None
        
        session["last_activity"] = datetime.now()
        return session
    
    def revoke_session(self, token: str):
        """Revoca una sesión"""
        if token in self.session_tokens:
            del self.session_tokens[token]




