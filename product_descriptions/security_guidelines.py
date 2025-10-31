from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from fastapi import FastAPI, APIRouter, HTTPException, status, Depends, Request, Response
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional, Union
import hashlib
import secrets
import re
import asyncio
import logging
from datetime import datetime, timedelta
from functools import wraps
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
        import hashlib
        import os
        from logging.handlers import RotatingFileHandler
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Security-Specific Guidelines for Cybersecurity Tool Development
Following Python/Cybersecurity Best Practices
"""


# ============================================================================
# SECURITY GUIDELINES IMPLEMENTATION
# ============================================================================

# 1. Input Validation and Sanitization
class SecureInputValidator:
    """Input validation and sanitization for security tools"""
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate IPv4/IPv6 address format"""
        ipv4_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
        ipv6_pattern = r"^(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}$"
        
        if re.match(ipv4_pattern, ip):
            # Validate IPv4 ranges
            parts = ip.split('.')
            return all(0 <= int(part) <= 255 for part in parts)
        elif re.match(ipv6_pattern, ip, re.IGNORECASE):
            return True
        return False
    
    @staticmethod
    def validate_hostname(hostname: str) -> bool:
        """Validate hostname format"""
        if len(hostname) > 253:
            return False
        if hostname.endswith("."):
            hostname = hostname[:-1]
        allowed = re.compile(r"(?!-)[A-Z\d-]{1,63}(?<!-)$", re.IGNORECASE)
        return all(allowed.match(x) for x in hostname.split("."))
    
    @staticmethod
    def validate_port_range(port: int) -> bool:
        """Validate port number range"""
        return 1 <= port <= 65535
    
    @staticmethod
    def sanitize_command_input(command: str) -> str:
        """Sanitize command input to prevent injection"""
        # Remove dangerous characters
        dangerous_chars = [';', '|', '&', '>', '<', '`', '$', '(', ')']
        for char in dangerous_chars:
            command = command.replace(char, '')
        return command.strip()
    
    @staticmethod
    def validate_file_path(path: str) -> bool:
        """Validate file path for path traversal attacks"""
        # Prevent directory traversal
        if '..' in path or path.startswith('/'):
            return False
        # Allow only alphanumeric, dots, hyphens, underscores
        return bool(re.match(r"^[a-zA-Z0-9._/-]+$", path))

# 2. Authentication and Authorization
class SecurityAuthenticator:
    """Authentication and authorization management"""
    
    def __init__(self, secret_key: str):
        
    """__init__ function."""
self.secret_key = secret_key
        self.token_blacklist = set()
        self.rate_limit_store = {}
    
    def generate_secure_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate JWT token with security claims"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)  # JWT ID for uniqueness
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            if token in self.token_blacklist:
                return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def check_permission(self, token: str, required_permission: str) -> bool:
        """Check if token has required permission"""
        payload = self.verify_token(token)
        if not payload:
            return False
        
        permissions = payload.get('permissions', [])
        return required_permission in permissions
    
    def blacklist_token(self, token: str) -> None:
        """Add token to blacklist"""
        self.token_blacklist.add(token)
    
    def check_rate_limit(self, user_id: str, action: str, limit: int = 10) -> bool:
        """Check rate limiting for user actions"""
        key = f"{user_id}:{action}"
        now = datetime.utcnow()
        
        if key not in self.rate_limit_store:
            self.rate_limit_store[key] = []
        
        # Remove old entries
        self.rate_limit_store[key] = [
            timestamp for timestamp in self.rate_limit_store[key]
            if now - timestamp < timedelta(minutes=1)
        ]
        
        if len(self.rate_limit_store[key]) >= limit:
            return False
        
        self.rate_limit_store[key].append(now)
        return True

# 3. Encryption and Hashing
class SecurityCrypto:
    """Cryptographic operations for security tools"""
    
    def __init__(self) -> Any:
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """Hash password with salt"""
        if not salt:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100k iterations
        )
        
        return {
            'hash': key.hex(),
            'salt': salt
        }
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        result = self.hash_password(password, salt)
        return result['hash'] == stored_hash
    
    def generate_secure_random_string(self, length: int = 32) -> str:
        """Generate cryptographically secure random string"""
        return secrets.token_urlsafe(length)

# 4. Secure Logging
class SecurityLogger:
    """Secure logging for security tools"""
    
    def __init__(self, log_file: str = "security.log"):
        
    """__init__ function."""
self.logger = logging.getLogger("security")
        self.logger.setLevel(logging.INFO)
        
        # File handler with rotation
        handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], user_id: Optional[str] = None) -> None:
        """Log security events with sanitized data"""
        # Sanitize sensitive data
        sanitized_details = self._sanitize_log_data(details)
        
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'details': sanitized_details,
            'ip_address': 'REDACTED'  # Don't log IPs in security logs
        }
        
        self.logger.info(f"SECURITY_EVENT: {log_entry}")
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str) -> None:
        """Log authentication attempts"""
        self.log_security_event(
            'authentication_attempt',
            {
                'success': success,
                'ip_address': ip_address
            },
            user_id
        )
    
    def log_authorization_failure(self, user_id: str, action: str, resource: str) -> None:
        """Log authorization failures"""
        self.log_security_event(
            'authorization_failure',
            {
                'action': action,
                'resource': resource
            },
            user_id
        )
    
    def _sanitize_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data for logging"""
        sensitive_keys = ['password', 'token', 'secret', 'key', 'credential']
        sanitized = {}
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '***REDACTED***'
            else:
                sanitized[key] = value
        
        return sanitized

# 5. Security Headers and CORS
class SecurityHeaders:
    """Security headers management"""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get security headers for responses"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    @staticmethod
    def add_security_headers(response: Response) -> Response:
        """Add security headers to response"""
        headers = SecurityHeaders.get_security_headers()
        for header, value in headers.items():
            response.headers[header] = value
        return response

# 6. Data Models with Security Validation
class SecureScanRequest(BaseModel):
    """Secure scan request model with validation"""
    target: str = Field(..., min_length=1, max_length=255)
    scan_type: str = Field(..., regex="^(port|vulnerability|web|network)$")
    timeout: int = Field(default=30, ge=1, le=300)
    max_ports: int = Field(default=1000, ge=1, le=65535)
    
    @field_validator('target')
    @classmethod
    def validate_target(cls, v: str) -> str:
        """Validate target address"""
        validator = SecureInputValidator()
        if not (validator.validate_ip_address(v) or validator.validate_hostname(v)):
            raise ValueError("Invalid target address")
        return v
    
    @field_validator('scan_type')
    @classmethod
    def validate_scan_type(cls, v: str) -> str:
        """Validate scan type"""
        allowed_types = ['port', 'vulnerability', 'web', 'network']
        if v not in allowed_types:
            raise ValueError(f"Invalid scan type. Allowed: {allowed_types}")
        return v

class SecureScanResponse(BaseModel):
    """Secure scan response model"""
    scan_id: str
    target: str
    scan_type: str
    status: str
    results: Dict[str, Any]
    timestamp: datetime
    duration: float

# 7. Security Middleware
class SecurityMiddleware:
    """Security middleware for FastAPI"""
    
    def __init__(self, authenticator: SecurityAuthenticator, logger: SecurityLogger):
        
    """__init__ function."""
self.authenticator = authenticator
        self.logger = logger
    
    async def __call__(self, request: Request, call_next):
        """Process request with security checks"""
        start_time = datetime.utcnow()
        
        # Log request
        self.logger.log_security_event('request_received', {
            'method': request.method,
            'path': str(request.url.path),
            'user_agent': request.headers.get('user-agent', 'Unknown')
        })
        
        # Check rate limiting
        client_ip = request.client.host
        if not self.authenticator.check_rate_limit(client_ip, 'api_request'):
            self.logger.log_security_event('rate_limit_exceeded', {
                'client_ip': client_ip
            })
            return Response(
                content='{"error": "Rate limit exceeded"}',
                status_code=429,
                media_type='application/json'
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response = SecurityHeaders.add_security_headers(response)
        
        # Log response
        duration = (datetime.utcnow() - start_time).total_seconds()
        self.logger.log_security_event('request_completed', {
            'status_code': response.status_code,
            'duration': duration
        })
        
        return response

# 8. Security Decorators
def require_authentication(permission: Optional[str] = None):
    """Decorator to require authentication"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract token from request
            request = kwargs.get('request')
            if not request:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                raise HTTPException(status_code=401, detail="Invalid authentication header")
            
            token = auth_header.split(' ')[1]
            
            # Verify token
            payload = authenticator.verify_token(token)
            if not payload:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Check permission if required
            if permission and not authenticator.check_permission(token, permission):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def validate_input(validator_func) -> bool:
    """Decorator to validate input"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Validate input using provided validator
            for arg_name, arg_value in kwargs.items():
                if hasattr(validator_func, f'validate_{arg_name}'):
                    validator = getattr(validator_func, f'validate_{arg_name}')
                    if not validator(arg_value):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid {arg_name}"
                        )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# 9. Security Utilities
class SecurityUtils:
    """Security utility functions"""
    
    @staticmethod
    def generate_secure_filename(original_name: str) -> str:
        """Generate secure filename"""
        # Remove dangerous characters
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', original_name)
        # Add random suffix
        suffix = secrets.token_hex(8)
        return f"{safe_name}_{suffix}"
    
    @staticmethod
    async def validate_file_upload(file_content: bytes, max_size: int = 10*1024*1024) -> bool:
        """Validate file upload"""
        if len(file_content) > max_size:
            return False
        
        # Check for dangerous file signatures
        dangerous_signatures = [
            b'\x4D\x5A',  # EXE
            b'\x7F\x45\x4C\x46',  # ELF
            b'\xFE\xED\xFA',  # Mach-O
        ]
        
        for signature in dangerous_signatures:
            if file_content.startswith(signature):
                return False
        
        return True
    
    @staticmethod
    def sanitize_sql_query(query: str) -> str:
        """Basic SQL injection prevention"""
        # Remove dangerous SQL keywords
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER',
            'EXEC', 'EXECUTE', 'UNION', 'SELECT'
        ]
        
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise ValueError(f"Dangerous SQL keyword detected: {keyword}")
        
        return query

# 10. FastAPI Security Router
router = APIRouter(prefix="/security", tags=["Security Operations"])

# Initialize security components
authenticator = SecurityAuthenticator(secrets.token_hex(32))
crypto = SecurityCrypto()
logger = SecurityLogger()
validator = SecureInputValidator()

@router.post("/scan", response_model=SecureScanResponse)
@require_authentication("scan_permission")
async def secure_scan(
    request: SecureScanRequest,
    current_request: Request
) -> SecureScanResponse:
    """Perform secure scan with validation"""
    scan_id = secrets.token_hex(16)
    
    # Log scan attempt
    logger.log_security_event('scan_initiated', {
        'scan_id': scan_id,
        'target': request.target,
        'scan_type': request.scan_type
    })
    
    # Perform scan (simulated)
    await asyncio.sleep(1)
    
    # Log scan completion
    logger.log_security_event('scan_completed', {
        'scan_id': scan_id,
        'target': request.target
    })
    
    return SecureScanResponse(
        scan_id=scan_id,
        target=request.target,
        scan_type=request.scan_type,
        status="completed",
        results={"ports": [80, 443], "services": ["http", "https"]},
        timestamp=datetime.utcnow(),
        duration=1.0
    )

@router.post("/encrypt")
async def encrypt_data(data: str) -> Dict[str, str]:
    """Encrypt sensitive data"""
    encrypted = crypto.encrypt_sensitive_data(data)
    return {"encrypted_data": encrypted}

@router.post("/hash-password")
async def hash_password(password: str) -> Dict[str, str]:
    """Hash password securely"""
    result = crypto.hash_password(password)
    return result

@router.get("/validate-input")
async def validate_input_endpoint(
    ip_address: str,
    hostname: str,
    port: int
) -> Dict[str, bool]:
    """Validate various input types"""
    return {
        "ip_valid": validator.validate_ip_address(ip_address),
        "hostname_valid": validator.validate_hostname(hostname),
        "port_valid": validator.validate_port_range(port)
    }

# 11. Security Configuration
class SecurityConfig:
    """Security configuration settings"""
    
    def __init__(self) -> Any:
        self.max_scan_duration = 300
        self.rate_limit_per_minute = 60
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_file_types = ['.txt', '.log', '.csv', '.json']
        self.session_timeout = 3600  # 1 hour
        self.password_min_length = 12
        self.require_special_chars = True
        self.max_login_attempts = 5
        self.lockout_duration = 900  # 15 minutes

# 12. Security Best Practices Implementation
class SecurityBestPractices:
    """Implementation of security best practices"""
    
    @staticmethod
    def implement_defense_in_depth():
        """Implement defense in depth strategy"""
        return {
            "network_layer": "Firewalls, IDS/IPS",
            "application_layer": "Input validation, authentication",
            "data_layer": "Encryption, access controls",
            "physical_layer": "Physical security, environmental controls"
        }
    
    @staticmethod
    def implement_least_privilege():
        """Implement least privilege principle"""
        return {
            "user_permissions": "Minimum required permissions",
            "service_accounts": "Limited scope and access",
            "network_access": "Restricted network segments",
            "file_permissions": "Read/write as needed only"
        }
    
    @staticmethod
    def implement_secure_by_default():
        """Implement secure by default configuration"""
        return {
            "default_deny": "Deny by default, allow by exception",
            "encryption_at_rest": "All data encrypted by default",
            "encryption_in_transit": "All communications encrypted",
            "secure_defaults": "Secure configuration defaults"
        }

# FastAPI app with security
app = FastAPI(
    title="Security Guidelines Implementation",
    description="Cybersecurity tool with security best practices",
    version="1.0.0"
)

# Add security middleware
app.add_middleware(SecurityMiddleware, authenticator=authenticator, logger=logger)

# Include security router
app.include_router(router)

if __name__ == "__main__":
    print("Security Guidelines Implementation")
    print("Access API at: http://localhost:8000")
    print("API Documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000) 