"""
Gamma App - Advanced Security Service
Comprehensive security features including rate limiting, encryption, and threat detection
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import jwt
from passlib.context import CryptContext
import ipaddress
import re
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import threading
from collections import defaultdict, deque
import requests
import dns.resolver
import whois

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats"""
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    CSRF_ATTACK = "csrf_attack"
    MALICIOUS_FILE = "malicious_file"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ANOMALY_DETECTED = "anomaly_detected"
    MALWARE_DETECTED = "malware_detected"
    PHISHING_ATTEMPT = "phishing_attempt"
    DDoS_ATTACK = "ddos_attack"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SESSION_HIJACKING = "session_hijacking"
    BOT_TRAFFIC = "bot_traffic"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"

@dataclass
class SecurityConfig:
    """Security configuration"""
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    session_timeout: int = 3600  # 1 hour
    password_min_length: int = 8
    require_strong_password: bool = True
    enable_2fa: bool = False
    encryption_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    allowed_ips: Optional[List[str]] = None
    blocked_ips: Optional[List[str]] = None

@dataclass
class SecurityEvent:
    """Security event"""
    id: str
    timestamp: datetime
    event_type: ThreatType
    severity: SecurityLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    metadata: Dict[str, Any]

@dataclass
class RateLimitInfo:
    """Rate limit information"""
    requests: int
    window_start: float
    limit: int
    window_size: int

class AdvancedSecurityService:
    """
    Advanced security service with comprehensive protection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize security service"""
        self.config = config or {}
        self.security_config = SecurityConfig(**self.config.get('security', {}))
        
        # Redis for rate limiting and session management
        self.redis_client = None
        self._init_redis()
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Encryption
        self.encryption_key = self._get_or_generate_key("encryption")
        self.cipher_suite = Fernet(self.encryption_key)
        
        # JWT secret
        self.jwt_secret = self._get_or_generate_key("jwt")
        
        # Security events storage
        self.security_events: List[SecurityEvent] = []
        
        # IP tracking
        self.ip_attempts: Dict[str, List[float]] = {}
        self.blocked_ips: Set[str] = set()
        
        # Advanced threat detection
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.behavioral_profiles: Dict[str, Dict] = {}
        self.threat_intelligence: Dict[str, Any] = {}
        self.geo_anomaly_detector = {}
        
        # Machine learning models
        self.ml_models = {
            'anomaly_detection': None,
            'malware_detection': None,
            'bot_detection': None
        }
        
        # Behavioral analysis
        self.user_behaviors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.ip_behaviors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Threat intelligence feeds
        self.threat_feeds = {
            'malicious_ips': set(),
            'malicious_domains': set(),
            'malicious_hashes': set()
        }
        
        # Load blocked IPs
        if self.security_config.blocked_ips:
            self.blocked_ips.update(self.security_config.blocked_ips)
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Start threat intelligence updates
        asyncio.create_task(self._update_threat_intelligence())
        
        logger.info("Advanced Security Service initialized successfully")

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established for security")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")

    def _get_or_generate_key(self, key_type: str) -> bytes:
        """Get or generate encryption key"""
        key_config = f"{key_type}_key"
        if key_config in self.config and self.config[key_config]:
            return self.config[key_config].encode()
        else:
            # Generate new key
            key = Fernet.generate_key()
            logger.warning(f"Generated new {key_type} key. Store it securely!")
            return key

    async def check_rate_limit(self, identifier: str, 
                             limit: Optional[int] = None,
                             window: Optional[int] = None) -> Tuple[bool, RateLimitInfo]:
        """Check rate limit for identifier"""
        try:
            limit = limit or self.security_config.rate_limit_requests
            window = window or self.security_config.rate_limit_window
            
            current_time = time.time()
            window_start = current_time - window
            
            # Check Redis first
            if self.redis_client:
                key = f"rate_limit:{identifier}"
                pipe = self.redis_client.pipeline()
                
                # Remove expired entries
                pipe.zremrangebyscore(key, 0, window_start)
                
                # Count current requests
                pipe.zcard(key)
                
                # Add current request
                pipe.zadd(key, {str(current_time): current_time})
                
                # Set expiration
                pipe.expire(key, window)
                
                results = pipe.execute()
                current_requests = results[1]
            else:
                # Fallback to in-memory storage
                if identifier not in self.ip_attempts:
                    self.ip_attempts[identifier] = []
                
                # Clean old attempts
                self.ip_attempts[identifier] = [
                    attempt for attempt in self.ip_attempts[identifier]
                    if attempt > window_start
                ]
                
                current_requests = len(self.ip_attempts[identifier])
                self.ip_attempts[identifier].append(current_time)
            
            # Check if limit exceeded
            is_allowed = current_requests < limit
            
            rate_limit_info = RateLimitInfo(
                requests=current_requests,
                window_start=window_start,
                limit=limit,
                window_size=window
            )
            
            if not is_allowed:
                await self._log_security_event(
                    ThreatType.RATE_LIMIT_EXCEEDED,
                    SecurityLevel.MEDIUM,
                    identifier,
                    f"Rate limit exceeded: {current_requests}/{limit} requests"
                )
            
            return is_allowed, rate_limit_info
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True, RateLimitInfo(0, 0, limit or 100, window or 3600)

    async def check_ip_security(self, ip_address: str) -> Tuple[bool, str]:
        """Check IP address security"""
        try:
            # Check if IP is blocked
            if ip_address in self.blocked_ips:
                return False, "IP address is blocked"
            
            # Check if IP is in allowed list (if configured)
            if self.security_config.allowed_ips:
                if ip_address not in self.security_config.allowed_ips:
                    return False, "IP address not in allowed list"
            
            # Check for suspicious patterns
            if self._is_suspicious_ip(ip_address):
                await self._log_security_event(
                    ThreatType.SUSPICIOUS_ACTIVITY,
                    SecurityLevel.MEDIUM,
                    ip_address,
                    f"Suspicious IP address detected: {ip_address}"
                )
                return False, "Suspicious IP address"
            
            return True, "IP address is safe"
            
        except Exception as e:
            logger.error(f"Error checking IP security: {e}")
            return True, "IP check failed"

    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious"""
        try:
            # Check if it's a private IP (might be suspicious in some contexts)
            ip = ipaddress.ip_address(ip_address)
            
            # Check for known malicious IP ranges (simplified)
            suspicious_ranges = [
                "10.0.0.0/8",      # Private range
                "172.16.0.0/12",   # Private range
                "192.168.0.0/16"   # Private range
            ]
            
            for range_str in suspicious_ranges:
                if ip in ipaddress.ip_network(range_str):
                    return True
            
            # Check for patterns that might indicate automated attacks
            if ip_address.endswith('.0') or ip_address.endswith('.255'):
                return True
            
            return False
            
        except Exception:
            return True  # Invalid IP addresses are suspicious

    async def validate_input(self, input_data: str, input_type: str = "general") -> Tuple[bool, str]:
        """Validate input for security threats"""
        try:
            # SQL Injection detection
            sql_patterns = [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
                r"(\b(OR|AND)\s+'.*'\s*=\s*'.*')",
                r"(--|\#|\/\*|\*\/)",
                r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\b)"
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    await self._log_security_event(
                        ThreatType.SQL_INJECTION,
                        SecurityLevel.HIGH,
                        "input_validation",
                        f"SQL injection attempt detected: {input_data[:100]}"
                    )
                    return False, "Potential SQL injection detected"
            
            # XSS detection
            xss_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>"
            ]
            
            for pattern in xss_patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    await self._log_security_event(
                        ThreatType.XSS_ATTACK,
                        SecurityLevel.HIGH,
                        "input_validation",
                        f"XSS attempt detected: {input_data[:100]}"
                    )
                    return False, "Potential XSS attack detected"
            
            # Check for excessive length
            if len(input_data) > 10000:
                return False, "Input too long"
            
            return True, "Input is safe"
            
        except Exception as e:
            logger.error(f"Error validating input: {e}")
            return False, "Input validation failed"

    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength"""
        issues = []
        
        if len(password) < self.security_config.password_min_length:
            issues.append(f"Password must be at least {self.security_config.password_min_length} characters")
        
        if self.security_config.require_strong_password:
            if not re.search(r"[A-Z]", password):
                issues.append("Password must contain at least one uppercase letter")
            
            if not re.search(r"[a-z]", password):
                issues.append("Password must contain at least one lowercase letter")
            
            if not re.search(r"\d", password):
                issues.append("Password must contain at least one digit")
            
            if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
                issues.append("Password must contain at least one special character")
        
        # Check for common passwords
        common_passwords = [
            "password", "123456", "123456789", "qwerty", "abc123",
            "password123", "admin", "letmein", "welcome", "monkey"
        ]
        
        if password.lower() in common_passwords:
            issues.append("Password is too common")
        
        return len(issues) == 0, issues

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise

    def generate_jwt_token(self, payload: Dict[str, Any], 
                          expires_delta: Optional[timedelta] = None) -> str:
        """Generate JWT token"""
        try:
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(seconds=self.security_config.session_timeout)
            
            payload.update({
                "exp": expire,
                "iat": datetime.utcnow(),
                "jti": secrets.token_urlsafe(32)
            })
            
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            return token
            
        except Exception as e:
            logger.error(f"Error generating JWT token: {e}")
            raise

    def verify_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, None
        except jwt.InvalidTokenError:
            return False, None
        except Exception as e:
            logger.error(f"Error verifying JWT token: {e}")
            return False, None

    async def check_login_attempts(self, identifier: str) -> Tuple[bool, int]:
        """Check login attempts for identifier"""
        try:
            key = f"login_attempts:{identifier}"
            
            if self.redis_client:
                attempts = self.redis_client.get(key)
                attempts = int(attempts) if attempts else 0
            else:
                attempts = self.ip_attempts.get(identifier, [])
                attempts = len([a for a in attempts if time.time() - a < self.security_config.lockout_duration])
            
            is_allowed = attempts < self.security_config.max_login_attempts
            return is_allowed, attempts
            
        except Exception as e:
            logger.error(f"Error checking login attempts: {e}")
            return True, 0

    async def record_login_attempt(self, identifier: str, success: bool):
        """Record login attempt"""
        try:
            key = f"login_attempts:{identifier}"
            
            if success:
                # Clear attempts on successful login
                if self.redis_client:
                    self.redis_client.delete(key)
                else:
                    self.ip_attempts.pop(identifier, None)
            else:
                # Increment failed attempts
                if self.redis_client:
                    pipe = self.redis_client.pipeline()
                    pipe.incr(key)
                    pipe.expire(key, self.security_config.lockout_duration)
                    pipe.execute()
                else:
                    if identifier not in self.ip_attempts:
                        self.ip_attempts[identifier] = []
                    self.ip_attempts[identifier].append(time.time())
                
                # Check if should be blocked
                is_allowed, attempts = await self.check_login_attempts(identifier)
                if not is_allowed:
                    await self._log_security_event(
                        ThreatType.BRUTE_FORCE_ATTACK,
                        SecurityLevel.HIGH,
                        identifier,
                        f"Brute force attack detected: {attempts} failed attempts"
                    )
                    
                    # Block IP temporarily
                    await self.block_ip(identifier, self.security_config.lockout_duration)
            
        except Exception as e:
            logger.error(f"Error recording login attempt: {e}")

    async def block_ip(self, ip_address: str, duration: int = 3600):
        """Block IP address temporarily"""
        try:
            self.blocked_ips.add(ip_address)
            
            if self.redis_client:
                key = f"blocked_ip:{ip_address}"
                self.redis_client.setex(key, duration, "blocked")
            
            # Schedule unblock
            asyncio.create_task(self._unblock_ip_after_delay(ip_address, duration))
            
            logger.warning(f"IP address {ip_address} blocked for {duration} seconds")
            
        except Exception as e:
            logger.error(f"Error blocking IP: {e}")

    async def _unblock_ip_after_delay(self, ip_address: str, delay: int):
        """Unblock IP address after delay"""
        await asyncio.sleep(delay)
        self.blocked_ips.discard(ip_address)
        logger.info(f"IP address {ip_address} unblocked")

    async def _log_security_event(self, event_type: ThreatType, severity: SecurityLevel,
                                source: str, description: str, 
                                user_id: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None):
        """Log security event"""
        try:
            event = SecurityEvent(
                id=secrets.token_urlsafe(16),
                timestamp=datetime.now(),
                event_type=event_type,
                severity=severity,
                source_ip=source,
                user_id=user_id,
                description=description,
                metadata=metadata or {}
            )
            
            self.security_events.append(event)
            
            # Store in Redis if available
            if self.redis_client:
                key = f"security_event:{event.id}"
                self.redis_client.setex(key, 86400, json.dumps(asdict(event), default=str))
            
            # Log to file
            logger.warning(f"Security Event: {event_type.value} - {description}")
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")

    async def get_security_events(self, limit: int = 100, 
                                severity: Optional[SecurityLevel] = None) -> List[SecurityEvent]:
        """Get security events"""
        try:
            events = self.security_events.copy()
            
            if severity:
                events = [e for e in events if e.severity == severity]
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.timestamp, reverse=True)
            
            return events[:limit]
            
        except Exception as e:
            logger.error(f"Error getting security events: {e}")
            return []

    async def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        try:
            total_events = len(self.security_events)
            
            # Count by severity
            severity_counts = {}
            for level in SecurityLevel:
                severity_counts[level.value] = sum(
                    1 for event in self.security_events 
                    if event.severity == level
                )
            
            # Count by type
            type_counts = {}
            for threat_type in ThreatType:
                type_counts[threat_type.value] = sum(
                    1 for event in self.security_events 
                    if event.event_type == threat_type
                )
            
            # Recent events (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_events = [
                event for event in self.security_events
                if event.timestamp > recent_cutoff
            ]
            
            return {
                "total_events": total_events,
                "recent_events": len(recent_events),
                "severity_counts": severity_counts,
                "type_counts": type_counts,
                "blocked_ips": len(self.blocked_ips),
                "active_rate_limits": len(self.ip_attempts)
            }
            
        except Exception as e:
            logger.error(f"Error getting security stats: {e}")
            return {}

    def generate_csrf_token(self) -> str:
        """Generate CSRF token"""
        return secrets.token_urlsafe(32)

    def verify_csrf_token(self, token: str, session_token: str) -> bool:
        """Verify CSRF token"""
        return hmac.compare_digest(token, session_token)

    async def cleanup_expired_data(self):
        """Clean up expired security data"""
        try:
            current_time = time.time()
            
            # Clean up old IP attempts
            for ip in list(self.ip_attempts.keys()):
                self.ip_attempts[ip] = [
                    attempt for attempt in self.ip_attempts[ip]
                    if current_time - attempt < self.security_config.lockout_duration
                ]
                
                if not self.ip_attempts[ip]:
                    del self.ip_attempts[ip]
            
            # Clean up old security events (keep last 1000)
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]
            
            logger.info("Security data cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during security cleanup: {e}")

    def _initialize_ml_models(self):
        """Initialize machine learning models for threat detection"""
        try:
            # Anomaly detection model
            self.ml_models['anomaly_detection'] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Initialize with dummy data to fit the model
            dummy_features = np.random.randn(100, 10)
            self.ml_models['anomaly_detection'].fit(dummy_features)
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    async def _update_threat_intelligence(self):
        """Update threat intelligence feeds"""
        while True:
            try:
                # Update malicious IPs (simplified - would use real threat feeds)
                await self._fetch_malicious_ips()
                
                # Update malicious domains
                await self._fetch_malicious_domains()
                
                # Update malicious file hashes
                await self._fetch_malicious_hashes()
                
                # Sleep for 1 hour before next update
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error updating threat intelligence: {e}")
                await asyncio.sleep(3600)
    
    async def _fetch_malicious_ips(self):
        """Fetch malicious IP addresses from threat feeds"""
        try:
            # This would integrate with real threat intelligence feeds
            # For now, we'll use a simplified approach
            malicious_ips = [
                "1.2.3.4",  # Example malicious IP
                "5.6.7.8",  # Example malicious IP
            ]
            
            self.threat_feeds['malicious_ips'].update(malicious_ips)
            logger.info(f"Updated {len(malicious_ips)} malicious IPs")
            
        except Exception as e:
            logger.error(f"Error fetching malicious IPs: {e}")
    
    async def _fetch_malicious_domains(self):
        """Fetch malicious domains from threat feeds"""
        try:
            # This would integrate with real threat intelligence feeds
            malicious_domains = [
                "malicious-site.com",
                "phishing-site.net",
            ]
            
            self.threat_feeds['malicious_domains'].update(malicious_domains)
            logger.info(f"Updated {len(malicious_domains)} malicious domains")
            
        except Exception as e:
            logger.error(f"Error fetching malicious domains: {e}")
    
    async def _fetch_malicious_hashes(self):
        """Fetch malicious file hashes from threat feeds"""
        try:
            # This would integrate with real threat intelligence feeds
            malicious_hashes = [
                "d41d8cd98f00b204e9800998ecf8427e",  # Example hash
                "5d41402abc4b2a76b9719d911017c592",  # Example hash
            ]
            
            self.threat_feeds['malicious_hashes'].update(malicious_hashes)
            logger.info(f"Updated {len(malicious_hashes)} malicious hashes")
            
        except Exception as e:
            logger.error(f"Error fetching malicious hashes: {e}")
    
    async def detect_anomaly(self, user_id: str, ip_address: str, 
                           request_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Detect behavioral anomalies using ML"""
        try:
            # Extract features for anomaly detection
            features = self._extract_behavioral_features(user_id, ip_address, request_data)
            
            if features is None:
                return False, 0.0
            
            # Use ML model to detect anomalies
            anomaly_score = self.ml_models['anomaly_detection'].decision_function([features])[0]
            is_anomaly = self.ml_models['anomaly_detection'].predict([features])[0] == -1
            
            # Update behavioral profile
            self._update_behavioral_profile(user_id, ip_address, features)
            
            if is_anomaly:
                await self._log_security_event(
                    ThreatType.ANOMALY_DETECTED,
                    SecurityLevel.MEDIUM,
                    ip_address,
                    f"Behavioral anomaly detected for user {user_id}",
                    user_id=user_id,
                    metadata={'anomaly_score': float(anomaly_score), 'features': features}
                )
            
            return is_anomaly, float(anomaly_score)
            
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return False, 0.0
    
    def _extract_behavioral_features(self, user_id: str, ip_address: str, 
                                   request_data: Dict[str, Any]) -> Optional[List[float]]:
        """Extract behavioral features for ML analysis"""
        try:
            current_time = time.time()
            
            # Get user behavior history
            user_history = self.user_behaviors[user_id]
            ip_history = self.ip_behaviors[ip_address]
            
            # Extract features
            features = []
            
            # Time-based features
            features.append(current_time % 86400)  # Time of day
            features.append(current_time % 604800)  # Day of week
            
            # Request frequency features
            recent_user_requests = len([t for t in user_history if current_time - t < 3600])
            recent_ip_requests = len([t for t in ip_history if current_time - t < 3600])
            
            features.append(recent_user_requests)
            features.append(recent_ip_requests)
            
            # Request pattern features
            if len(user_history) > 1:
                intervals = [user_history[i] - user_history[i-1] for i in range(1, len(user_history))]
                avg_interval = np.mean(intervals) if intervals else 0
                features.append(avg_interval)
            else:
                features.append(0)
            
            # IP-based features
            features.append(len(ip_history))
            features.append(len(user_history))
            
            # Request data features
            features.append(len(str(request_data)))
            features.append(len(request_data.get('headers', {})))
            
            # Geographic features (simplified)
            features.append(self._get_geo_risk_score(ip_address))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting behavioral features: {e}")
            return None
    
    def _get_geo_risk_score(self, ip_address: str) -> float:
        """Get geographic risk score for IP address"""
        try:
            # This would integrate with real geolocation services
            # For now, return a simplified risk score
            if ip_address.startswith('192.168.') or ip_address.startswith('10.'):
                return 0.1  # Low risk for private IPs
            else:
                return 0.5  # Medium risk for public IPs
                
        except Exception:
            return 0.5
    
    def _update_behavioral_profile(self, user_id: str, ip_address: str, features: List[float]):
        """Update behavioral profile for user and IP"""
        try:
            current_time = time.time()
            
            # Update user behavior
            self.user_behaviors[user_id].append(current_time)
            
            # Update IP behavior
            self.ip_behaviors[ip_address].append(current_time)
            
            # Update behavioral profiles
            if user_id not in self.behavioral_profiles:
                self.behavioral_profiles[user_id] = {
                    'request_count': 0,
                    'avg_features': features.copy(),
                    'last_seen': current_time
                }
            
            profile = self.behavioral_profiles[user_id]
            profile['request_count'] += 1
            profile['last_seen'] = current_time
            
            # Update average features (exponential moving average)
            alpha = 0.1
            for i, feature in enumerate(features):
                profile['avg_features'][i] = (1 - alpha) * profile['avg_features'][i] + alpha * feature
            
        except Exception as e:
            logger.error(f"Error updating behavioral profile: {e}")
    
    async def detect_malware(self, file_data: bytes, file_name: str) -> Tuple[bool, str]:
        """Detect malware in uploaded files"""
        try:
            # Calculate file hash
            file_hash = hashlib.sha256(file_data).hexdigest()
            
            # Check against known malicious hashes
            if file_hash in self.threat_feeds['malicious_hashes']:
                await self._log_security_event(
                    ThreatType.MALWARE_DETECTED,
                    SecurityLevel.CRITICAL,
                    "file_upload",
                    f"Known malware detected: {file_name}",
                    metadata={'file_hash': file_hash, 'file_name': file_name}
                )
                return True, "Known malware detected"
            
            # Check file size (very large files might be suspicious)
            if len(file_data) > 100 * 1024 * 1024:  # 100MB
                await self._log_security_event(
                    ThreatType.SUSPICIOUS_ACTIVITY,
                    SecurityLevel.MEDIUM,
                    "file_upload",
                    f"Large file upload: {file_name}",
                    metadata={'file_size': len(file_data), 'file_name': file_name}
                )
                return False, "Large file detected"
            
            # Check file extension
            suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com']
            if any(file_name.lower().endswith(ext) for ext in suspicious_extensions):
                await self._log_security_event(
                    ThreatType.SUSPICIOUS_ACTIVITY,
                    SecurityLevel.HIGH,
                    "file_upload",
                    f"Suspicious file type: {file_name}",
                    metadata={'file_name': file_name}
                )
                return False, "Suspicious file type"
            
            return False, "File appears safe"
            
        except Exception as e:
            logger.error(f"Error detecting malware: {e}")
            return False, "Malware detection failed"
    
    async def detect_bot_traffic(self, ip_address: str, user_agent: str, 
                               request_pattern: Dict[str, Any]) -> Tuple[bool, float]:
        """Detect bot traffic"""
        try:
            bot_score = 0.0
            
            # Check user agent
            if not user_agent or len(user_agent) < 10:
                bot_score += 0.3
            
            # Check for common bot patterns
            bot_patterns = [
                'bot', 'crawler', 'spider', 'scraper', 'curl', 'wget',
                'python-requests', 'java', 'go-http-client'
            ]
            
            user_agent_lower = user_agent.lower()
            for pattern in bot_patterns:
                if pattern in user_agent_lower:
                    bot_score += 0.2
            
            # Check request frequency
            ip_history = self.ip_behaviors[ip_address]
            if len(ip_history) > 100:  # High request frequency
                bot_score += 0.3
            
            # Check request patterns
            if request_pattern.get('no_referer', False):
                bot_score += 0.1
            
            if request_pattern.get('no_cookies', False):
                bot_score += 0.1
            
            # Check for automated patterns
            if len(ip_history) > 10:
                intervals = [ip_history[i] - ip_history[i-1] for i in range(1, len(ip_history))]
                if intervals:
                    avg_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    
                    # Very regular intervals suggest automation
                    if std_interval < avg_interval * 0.1:
                        bot_score += 0.2
            
            is_bot = bot_score > 0.5
            
            if is_bot:
                await self._log_security_event(
                    ThreatType.BOT_TRAFFIC,
                    SecurityLevel.MEDIUM,
                    ip_address,
                    f"Bot traffic detected: {user_agent}",
                    metadata={'bot_score': bot_score, 'user_agent': user_agent}
                )
            
            return is_bot, bot_score
            
        except Exception as e:
            logger.error(f"Error detecting bot traffic: {e}")
            return False, 0.0
    
    async def detect_geographic_anomaly(self, ip_address: str, user_id: str) -> Tuple[bool, str]:
        """Detect geographic anomalies in user behavior"""
        try:
            # This would integrate with real geolocation services
            # For now, we'll use a simplified approach
            
            if user_id in self.behavioral_profiles:
                profile = self.behavioral_profiles[user_id]
                
                # Check if user has a history of locations
                if 'locations' in profile:
                    # Compare with previous locations
                    current_location = self._get_ip_location(ip_address)
                    
                    if current_location not in profile['locations']:
                        # New location detected
                        profile['locations'].append(current_location)
                        
                        if len(profile['locations']) > 3:
                            await self._log_security_event(
                                ThreatType.GEOGRAPHIC_ANOMALY,
                                SecurityLevel.MEDIUM,
                                ip_address,
                                f"Geographic anomaly detected for user {user_id}",
                                user_id=user_id,
                                metadata={'current_location': current_location, 'previous_locations': profile['locations']}
                            )
                            return True, "Geographic anomaly detected"
                else:
                    # First location for this user
                    profile['locations'] = [self._get_ip_location(ip_address)]
            
            return False, "No geographic anomaly"
            
        except Exception as e:
            logger.error(f"Error detecting geographic anomaly: {e}")
            return False, "Geographic anomaly detection failed"
    
    def _get_ip_location(self, ip_address: str) -> str:
        """Get location for IP address (simplified)"""
        try:
            # This would integrate with real geolocation services
            # For now, return a simplified location
            if ip_address.startswith('192.168.'):
                return 'Private Network'
            elif ip_address.startswith('10.'):
                return 'Private Network'
            else:
                return 'Public Network'
                
        except Exception:
            return 'Unknown'
    
    async def analyze_request_security(self, request_data: Dict[str, Any], 
                                     ip_address: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive security analysis of a request"""
        try:
            analysis_results = {
                'is_safe': True,
                'threats_detected': [],
                'risk_score': 0.0,
                'recommendations': []
            }
            
            # Check IP security
            ip_safe, ip_message = await self.check_ip_security(ip_address)
            if not ip_safe:
                analysis_results['is_safe'] = False
                analysis_results['threats_detected'].append(f"IP Security: {ip_message}")
                analysis_results['risk_score'] += 0.3
            
            # Check rate limiting
            rate_limit_ok, rate_info = await self.check_rate_limit(ip_address)
            if not rate_limit_ok:
                analysis_results['is_safe'] = False
                analysis_results['threats_detected'].append("Rate limit exceeded")
                analysis_results['risk_score'] += 0.2
            
            # Validate input
            if 'data' in request_data:
                input_safe, input_message = await self.validate_input(str(request_data['data']))
                if not input_safe:
                    analysis_results['is_safe'] = False
                    analysis_results['threats_detected'].append(f"Input Validation: {input_message}")
                    analysis_results['risk_score'] += 0.4
            
            # Detect anomalies
            if user_id:
                is_anomaly, anomaly_score = await self.detect_anomaly(user_id, ip_address, request_data)
                if is_anomaly:
                    analysis_results['is_safe'] = False
                    analysis_results['threats_detected'].append("Behavioral anomaly detected")
                    analysis_results['risk_score'] += 0.3
            
            # Detect bot traffic
            user_agent = request_data.get('headers', {}).get('user-agent', '')
            is_bot, bot_score = await self.detect_bot_traffic(ip_address, user_agent, request_data)
            if is_bot:
                analysis_results['is_safe'] = False
                analysis_results['threats_detected'].append("Bot traffic detected")
                analysis_results['risk_score'] += 0.2
            
            # Detect geographic anomalies
            if user_id:
                geo_anomaly, geo_message = await self.detect_geographic_anomaly(ip_address, user_id)
                if geo_anomaly:
                    analysis_results['is_safe'] = False
                    analysis_results['threats_detected'].append(f"Geographic: {geo_message}")
                    analysis_results['risk_score'] += 0.1
            
            # Generate recommendations
            if analysis_results['risk_score'] > 0.5:
                analysis_results['recommendations'].append("Consider blocking this request")
            if analysis_results['risk_score'] > 0.3:
                analysis_results['recommendations'].append("Monitor this user/IP closely")
            if analysis_results['risk_score'] > 0.1:
                analysis_results['recommendations'].append("Log this request for analysis")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing request security: {e}")
            return {
                'is_safe': False,
                'threats_detected': ['Analysis failed'],
                'risk_score': 1.0,
                'recommendations': ['Block request due to analysis failure']
            }
    
    async def get_security_analytics(self) -> Dict[str, Any]:
        """Get comprehensive security analytics"""
        try:
            # Get basic security stats
            basic_stats = await self.get_security_stats()
            
            # Calculate threat trends
            recent_events = [e for e in self.security_events 
                           if e.timestamp > datetime.now() - timedelta(hours=24)]
            
            threat_trends = {}
            for threat_type in ThreatType:
                count = sum(1 for e in recent_events if e.event_type == threat_type)
                threat_trends[threat_type.value] = count
            
            # Calculate risk distribution
            risk_distribution = {
                'low': sum(1 for e in recent_events if e.severity == SecurityLevel.LOW),
                'medium': sum(1 for e in recent_events if e.severity == SecurityLevel.MEDIUM),
                'high': sum(1 for e in recent_events if e.severity == SecurityLevel.HIGH),
                'critical': sum(1 for e in recent_events if e.severity == SecurityLevel.CRITICAL)
            }
            
            # Calculate behavioral insights
            behavioral_insights = {
                'total_users_tracked': len(self.behavioral_profiles),
                'total_ips_tracked': len(self.ip_behaviors),
                'anomalies_detected': sum(1 for e in recent_events if e.event_type == ThreatType.ANOMALY_DETECTED),
                'bots_detected': sum(1 for e in recent_events if e.event_type == ThreatType.BOT_TRAFFIC)
            }
            
            return {
                'basic_stats': basic_stats,
                'threat_trends': threat_trends,
                'risk_distribution': risk_distribution,
                'behavioral_insights': behavioral_insights,
                'threat_intelligence': {
                    'malicious_ips': len(self.threat_feeds['malicious_ips']),
                    'malicious_domains': len(self.threat_feeds['malicious_domains']),
                    'malicious_hashes': len(self.threat_feeds['malicious_hashes'])
                },
                'ml_models_status': {
                    'anomaly_detection': self.ml_models['anomaly_detection'] is not None,
                    'malware_detection': self.ml_models['malware_detection'] is not None,
                    'bot_detection': self.ml_models['bot_detection'] is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting security analytics: {e}")
            return {}
    
    async def close(self):
        """Close security service"""
        try:
            if self.redis_client:
                self.redis_client.close()
            
            logger.info("Security service closed")
            
        except Exception as e:
            logger.error(f"Error closing security service: {e}")

# Global security service instance
security_service = AdvancedSecurityService()

# Convenience functions
async def check_rate_limit(identifier: str) -> Tuple[bool, RateLimitInfo]:
    """Check rate limit"""
    return await security_service.check_rate_limit(identifier)

async def validate_input(input_data: str) -> Tuple[bool, str]:
    """Validate input"""
    return await security_service.validate_input(input_data)

def hash_password(password: str) -> str:
    """Hash password"""
    return security_service.hash_password(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return security_service.verify_password(plain_password, hashed_password)











