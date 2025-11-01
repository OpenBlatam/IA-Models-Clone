# 🔒 Guía de Seguridad - Blatam Academy Features

## 🛡️ Mejores Prácticas de Seguridad

### 1. Autenticación y Autorización

#### JWT Tokens
```python
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
```

#### Validación de Tokens
```python
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 2. Rate Limiting

#### Rate Limiting por IP
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/endpoint")
@limiter.limit("100/minute")
async def endpoint(request: Request):
    ...
```

#### Rate Limiting por Usuario
```python
def get_user_id(request: Request):
    # Obtener user_id del token
    token = request.headers.get("Authorization")
    payload = verify_token(token)
    return payload.get("user_id")

limiter = Limiter(key_func=get_user_id)
```

### 3. Input Sanitization

```python
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper

secure_engine = SecureEngineWrapper(
    engine,
    enable_sanitization=True,
    block_sql_injection=True,
    block_xss=True,
    block_path_traversal=True
)

# Sanitización automática de todos los inputs
```

### 4. Secrets Management

#### ❌ Evitar
```python
API_KEY = "sk-1234567890abcdef"  # NUNCA en código
```

#### ✅ Correcto
```python
import os
from azure.keyvault.secrets import SecretClient  # Ejemplo

# Opción 1: Variables de entorno
API_KEY = os.getenv("OPENAI_API_KEY")

# Opción 2: Secret Manager
secret_client = SecretClient(vault_url, credential)
API_KEY = secret_client.get_secret("openai-api-key").value
```

### 5. HTTPS y SSL/TLS

#### Configurar Nginx con SSL
```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    location / {
        proxy_pass http://backend:8000;
    }
}
```

### 6. CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Específico
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600
)
```

### 7. Security Headers

```python
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response
```

### 8. Audit Logging

```python
import logging

audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)

def log_access(user_id, endpoint, action, result):
    audit_logger.info(json.dumps({
        "event": "access",
        "user_id": user_id,
        "endpoint": endpoint,
        "action": action,
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    }))
```

### 9. KV Cache Security

```python
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper

secure_engine = SecureEngineWrapper(
    engine,
    enable_sanitization=True,
    enable_rate_limiting=True,
    rate_limit_per_minute=100,
    enable_access_control=True,
    enable_hmac=True,
    hmac_secret=os.getenv("HMAC_SECRET")
)

# IP Whitelist/Blacklist
secure_engine.add_ip_to_whitelist(['192.168.1.0/24'])
secure_engine.add_ip_to_blacklist(['10.0.0.100'])
```

### 10. Database Security

```python
# Usar parámetros preparados (nunca string formatting)
# ❌ Vulnerable a SQL Injection
query = f"SELECT * FROM users WHERE id = {user_id}"

# ✅ Seguro
query = "SELECT * FROM users WHERE id = :user_id"
result = db.execute(query, {"user_id": user_id})
```

## 🔐 Checklist de Seguridad

### Pre-Deployment
- [ ] Secrets no en código
- [ ] Variables de entorno configuradas
- [ ] SSL/TLS configurado
- [ ] Firewall configurado
- [ ] Rate limiting activado
- [ ] CORS configurado correctamente
- [ ] Security headers configurados

### Post-Deployment
- [ ] Penetration testing realizado
- [ ] Security audit completado
- [ ] Vulnerabilidades verificadas
- [ ] Access logs monitoreados
- [ ] Alertas de seguridad configuradas

## 🚨 Incident Response

### Detectar Breach
1. Monitorear logs de acceso
2. Alertas automáticas
3. Análisis de patrones sospechosos

### Respuesta
1. Aislar sistema afectado
2. Cambiar credenciales
3. Revisar logs
4. Notificar stakeholders
5. Documentar incidente

---

**Recursos:**
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/secrets.html)

