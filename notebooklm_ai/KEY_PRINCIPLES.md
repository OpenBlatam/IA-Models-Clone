# Key Principles for Python Cybersecurity & AI/ML Development

## 🛡️ **Security-First Development Principles**

### 1. **Defense in Depth**
```python
# Multiple layers of security
class SecurityLayer:
    def __init__(self):
        self.input_validation = InputValidator()
        self.authentication = AuthManager()
        self.authorization = AccessControl()
        self.encryption = CryptoManager()
        self.monitoring = SecurityMonitor()
```

### 2. **Zero Trust Architecture**
```python
# Never trust, always verify
class ZeroTrustSecurity:
    def verify_request(self, request: Request) -> bool:
        # Verify identity
        if not self.authenticate_user(request):
            return False
        
        # Verify device
        if not self.verify_device(request):
            return False
        
        # Verify network
        if not self.verify_network(request):
            return False
        
        # Verify access
        if not self.verify_permissions(request):
            return False
        
        return True
```

### 3. **Principle of Least Privilege**
```python
# Minimum required permissions
class AccessControl:
    def __init__(self):
        self.roles = {
            'admin': ['read', 'write', 'delete', 'execute'],
            'user': ['read', 'write'],
            'guest': ['read']
        }
    
    def check_permission(self, user_role: str, action: str) -> bool:
        return action in self.roles.get(user_role, [])
```

## 🔒 **Input Validation & Sanitization**

### 1. **Whitelist Validation**
```python
class InputValidator:
    def __init__(self):
        self.allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
        self.max_length = 1000
    
    def sanitize_input(self, input_str: str) -> str:
        # Remove dangerous characters
        sanitized = ''.join(c for c in input_str if c in self.allowed_chars)
        
        # Limit length
        if len(sanitized) > self.max_length:
            sanitized = sanitized[:self.max_length]
        
        return sanitized
```

### 2. **Type Safety**
```python
from typing import Annotated, Union
from pydantic import BaseModel, Field, validator

class SecureRequest(BaseModel):
    user_id: Annotated[int, Field(gt=0)]
    action: Annotated[str, Field(min_length=1, max_length=50)]
    data: Annotated[Union[str, dict], Field()]
    
    @validator('action')
    def validate_action(cls, v):
        allowed_actions = ['read', 'write', 'delete']
        if v not in allowed_actions:
            raise ValueError(f'Action must be one of: {allowed_actions}')
        return v
```

### 3. **SQL Injection Prevention**
```python
class SecureDatabase:
    def __init__(self, connection_string: str):
        self.db = Database(connection_string)
    
    async def safe_query(self, query: str, params: dict) -> List[Dict]:
        # Use parameterized queries
        return await self.db.fetch_all(text(query), params)
    
    async def unsafe_query(self, query: str) -> List[Dict]:
        # NEVER DO THIS - vulnerable to SQL injection
        return await self.db.fetch_all(query)
```

## 🔐 **Authentication & Authorization**

### 1. **Multi-Factor Authentication**
```python
class MFAManager:
    def __init__(self):
        self.totp = pyotp.TOTP('base32secret3232')
        self.sms_provider = SMSProvider()
    
    async def authenticate_user(self, username: str, password: str, mfa_code: str) -> bool:
        # Verify password
        if not await self.verify_password(username, password):
            return False
        
        # Verify MFA
        if not self.totp.verify(mfa_code):
            return False
        
        return True
```

### 2. **JWT Token Security**
```python
class JWTSecurity:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
    
    def create_token(self, user_id: str, expires_in: int = 3600) -> str:
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())  # Unique token ID
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
```

### 3. **Session Management**
```python
class SecureSessionManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.session_timeout = 3600
    
    async def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }
        
        await self.redis.setex(
            f"session:{session_id}",
            self.session_timeout,
            json.dumps(session_data)
        )
        
        return session_id
    
    async def validate_session(self, session_id: str) -> Optional[Dict]:
        session_data = await self.redis.get(f"session:{session_id}")
        if not session_data:
            return None
        
        session = json.loads(session_data)
        session["last_activity"] = datetime.utcnow().isoformat()
        
        # Update last activity
        await self.redis.setex(
            f"session:{session_id}",
            self.session_timeout,
            json.dumps(session)
        )
        
        return session
```

## 🚀 **Performance & Scalability**

### 1. **Async Programming**
```python
class AsyncSecurityScanner:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(100)  # Limit concurrent connections
    
    async def scan_targets(self, targets: List[str]) -> List[Dict]:
        async def scan_single_target(target: str) -> Dict:
            async with self.semaphore:
                return await self.scan_target(target)
        
        tasks = [scan_single_target(target) for target in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if not isinstance(r, Exception)]
```

### 2. **Connection Pooling**
```python
class SecureConnectionPool:
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self._pool = None
    
    async def get_pool(self) -> aiohttp.TCPConnector:
        if self._pool is None:
            self._pool = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30
            )
        return self._pool
```

### 3. **Caching Strategy**
```python
class SecurityCache:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def get_cached_result(self, key: str) -> Optional[Dict]:
        cached = await self.redis.get(f"security:{key}")
        return json.loads(cached) if cached else None
    
    async def cache_result(self, key: str, result: Dict, ttl: int = 3600):
        await self.redis.setex(
            f"security:{key}",
            ttl,
            json.dumps(result)
        )
```

## 📊 **Monitoring & Logging**

### 1. **Structured Logging**
```python
import structlog

class SecurityLogger:
    def __init__(self):
        self.logger = structlog.get_logger()
    
    def log_security_event(self, event_type: str, user_id: str, details: Dict):
        self.logger.info(
            "security_event",
            event_type=event_type,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            ip_address=details.get('ip_address'),
            user_agent=details.get('user_agent'),
            action=details.get('action'),
            result=details.get('result'),
            risk_score=details.get('risk_score', 0)
        )
```

### 2. **Metrics Collection**
```python
from prometheus_client import Counter, Histogram, Gauge

class SecurityMetrics:
    def __init__(self):
        self.login_attempts = Counter('security_login_attempts_total', 'Total login attempts', ['result'])
        self.failed_logins = Counter('security_failed_logins_total', 'Failed login attempts', ['username'])
        self.security_events = Counter('security_events_total', 'Security events', ['type', 'severity'])
        self.response_time = Histogram('security_response_time_seconds', 'Response time', ['endpoint'])
        self.active_sessions = Gauge('security_active_sessions', 'Number of active sessions')
    
    def record_login_attempt(self, username: str, success: bool):
        result = 'success' if success else 'failure'
        self.login_attempts.labels(result=result).inc()
        
        if not success:
            self.failed_logins.labels(username=username).inc()
```

### 3. **Real-time Alerting**
```python
class SecurityAlerting:
    def __init__(self):
        self.alert_thresholds = {
            'failed_logins': 5,
            'suspicious_activity': 3,
            'data_exfiltration': 1
        }
    
    async def check_alerts(self, event_type: str, count: int):
        threshold = self.alert_thresholds.get(event_type, 0)
        
        if count >= threshold:
            await self.send_alert(event_type, count)
    
    async def send_alert(self, event_type: str, count: int):
        alert = {
            "type": "security_alert",
            "event_type": event_type,
            "count": count,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "high" if count > 10 else "medium"
        }
        
        # Send to monitoring system
        await self.send_to_monitoring(alert)
```

## 🔍 **Threat Detection**

### 1. **Anomaly Detection**
```python
class AnomalyDetector:
    def __init__(self):
        self.baseline = {}
        self.threshold = 2.0  # Standard deviations
    
    def detect_anomaly(self, user_id: str, activity: Dict) -> bool:
        if user_id not in self.baseline:
            self.baseline[user_id] = self.calculate_baseline(activity)
            return False
        
        current_score = self.calculate_activity_score(activity)
        baseline_score = self.baseline[user_id]
        
        # Check if activity is anomalous
        deviation = abs(current_score - baseline_score) / baseline_score
        
        return deviation > self.threshold
    
    def calculate_activity_score(self, activity: Dict) -> float:
        # Implement activity scoring algorithm
        score = 0
        score += activity.get('request_count', 0) * 0.1
        score += activity.get('data_volume', 0) * 0.01
        score += activity.get('error_rate', 0) * 10
        return score
```

### 2. **Behavioral Analysis**
```python
class BehavioralAnalyzer:
    def __init__(self):
        self.user_profiles = {}
    
    def analyze_behavior(self, user_id: str, action: str, context: Dict) -> Dict:
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = self.create_user_profile()
        
        profile = self.user_profiles[user_id]
        
        # Analyze action patterns
        risk_score = self.calculate_risk_score(action, context, profile)
        
        # Update profile
        self.update_profile(profile, action, context)
        
        return {
            "risk_score": risk_score,
            "anomaly_detected": risk_score > 0.7,
            "recommendations": self.get_recommendations(risk_score)
        }
```

## 🛠️ **Secure Development Practices**

### 1. **Code Review Checklist**
```python
class SecurityCodeReview:
    def __init__(self):
        self.checklist = [
            "Input validation implemented",
            "SQL injection prevented",
            "XSS protection in place",
            "Authentication required",
            "Authorization checked",
            "Sensitive data encrypted",
            "Logging implemented",
            "Error handling secure",
            "Rate limiting configured",
            "HTTPS enforced"
        ]
    
    def review_code(self, code_file: str) -> Dict:
        results = {}
        for item in self.checklist:
            results[item] = self.check_item(code_file, item)
        return results
```

### 2. **Dependency Management**
```python
class SecureDependencies:
    def __init__(self):
        self.vulnerability_db = VulnerabilityDatabase()
    
    async def scan_dependencies(self, requirements_file: str) -> List[Dict]:
        vulnerabilities = []
        
        with open(requirements_file, 'r') as f:
            dependencies = f.readlines()
        
        for dep in dependencies:
            package_name = dep.split('==')[0].strip()
            version = dep.split('==')[1].strip() if '==' in dep else None
            
            vulns = await self.vulnerability_db.check_package(package_name, version)
            vulnerabilities.extend(vulns)
        
        return vulnerabilities
```

### 3. **Configuration Security**
```python
class SecureConfig:
    def __init__(self):
        self.secret_manager = SecretManager()
    
    def load_config(self) -> Dict:
        config = {
            "database_url": os.getenv("DATABASE_URL"),
            "redis_url": os.getenv("REDIS_URL"),
            "jwt_secret": self.secret_manager.get_secret("jwt_secret"),
            "api_keys": self.secret_manager.get_secret("api_keys"),
            "encryption_key": self.secret_manager.get_secret("encryption_key")
        }
        
        # Validate configuration
        self.validate_config(config)
        
        return config
    
    def validate_config(self, config: Dict):
        required_keys = ["database_url", "jwt_secret", "encryption_key"]
        for key in required_keys:
            if not config.get(key):
                raise ValueError(f"Missing required configuration: {key}")
```

## 🚨 **Incident Response**

### 1. **Automated Response**
```python
class IncidentResponse:
    def __init__(self):
        self.response_actions = {
            'brute_force': self.handle_brute_force,
            'data_exfiltration': self.handle_data_exfiltration,
            'malware_detection': self.handle_malware,
            'privilege_escalation': self.handle_privilege_escalation
        }
    
    async def handle_incident(self, incident_type: str, details: Dict):
        if incident_type in self.response_actions:
            await self.response_actions[incident_type](details)
        
        # Log incident
        await self.log_incident(incident_type, details)
        
        # Notify security team
        await self.notify_security_team(incident_type, details)
    
    async def handle_brute_force(self, details: Dict):
        # Block IP address
        await self.block_ip(details['ip_address'])
        
        # Lock account
        await self.lock_account(details['username'])
        
        # Increase monitoring
        await self.increase_monitoring(details['ip_address'])
```

### 2. **Forensic Analysis**
```python
class ForensicAnalyzer:
    def __init__(self):
        self.evidence_collector = EvidenceCollector()
    
    async def analyze_incident(self, incident_id: str) -> Dict:
        # Collect evidence
        evidence = await self.evidence_collector.collect(incident_id)
        
        # Analyze timeline
        timeline = self.analyze_timeline(evidence)
        
        # Identify root cause
        root_cause = self.identify_root_cause(evidence)
        
        # Generate report
        report = {
            "incident_id": incident_id,
            "timeline": timeline,
            "root_cause": root_cause,
            "evidence": evidence,
            "recommendations": self.generate_recommendations(evidence)
        }
        
        return report
```

## 📋 **Compliance & Governance**

### 1. **Audit Trail**
```python
class AuditTrail:
    def __init__(self, database: Database):
        self.db = database
    
    async def log_audit_event(self, event: Dict):
        query = """
            INSERT INTO audit_log (
                user_id, action, resource, timestamp, ip_address, 
                user_agent, success, details
            ) VALUES (
                :user_id, :action, :resource, :timestamp, :ip_address,
                :user_agent, :success, :details
            )
        """
        
        await self.db.execute(query, event)
    
    async def get_audit_trail(self, user_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        query = """
            SELECT * FROM audit_log 
            WHERE user_id = :user_id 
            AND timestamp BETWEEN :start_date AND :end_date
            ORDER BY timestamp DESC
        """
        
        return await self.db.fetch_all(query, {
            "user_id": user_id,
            "start_date": start_date,
            "end_date": end_date
        })
```

### 2. **Data Protection**
```python
class DataProtection:
    def __init__(self, encryption_key: str):
        self.cipher = AES.new(encryption_key.encode(), AES.MODE_GCM)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        ciphertext, tag = self.cipher.encrypt_and_digest(data.encode())
        return base64.b64encode(ciphertext + tag).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        data = base64.b64decode(encrypted_data.encode())
        ciphertext = data[:-16]
        tag = data[-16:]
        plaintext = self.cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext.decode()
```

## 🎯 **Key Takeaways**

1. **Security is not a feature, it's a fundamental requirement**
2. **Always validate and sanitize inputs**
3. **Implement defense in depth**
4. **Monitor everything, alert on anomalies**
5. **Use async programming for performance**
6. **Implement proper authentication and authorization**
7. **Log security events comprehensively**
8. **Have incident response procedures**
9. **Regular security audits and updates**
10. **Train developers on secure coding practices**

These principles form the foundation of secure, scalable, and maintainable Python applications in cybersecurity and AI/ML domains. 