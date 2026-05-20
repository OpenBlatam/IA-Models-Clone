# 🔒 Guía de Seguridad - Music Analyzer AI

Esta guía cubre las mejores prácticas de seguridad para Music Analyzer AI.

## 🛡️ Principios de Seguridad

### 1. Defensa en Profundidad

Implementa múltiples capas de seguridad:
- Validación de inputs
- Autenticación y autorización
- Rate limiting
- Logging y monitoreo
- Encriptación

### 2. Principio de Menor Privilegio

- Usa solo los permisos necesarios
- Limita acceso a recursos
- Valida todas las entradas

### 3. Seguridad por Defecto

- Configuraciones seguras por defecto
- Sin credenciales hardcodeadas
- Validación automática

## 🔐 Gestión de Credenciales

### Variables de Entorno

✅ **Correcto:**
```env
# .env (en .gitignore)
SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret
```

❌ **Incorrecto:**
```python
# main.py
SPOTIFY_CLIENT_ID = "hardcoded_id"  # NUNCA
```

### Secret Managers

Para producción, usa secret managers:

**AWS Secrets Manager:**
```python
import boto3

secrets_client = boto3.client('secretsmanager')
secret = secrets_client.get_secret_value(SecretId='music-analyzer/spotify')
credentials = json.loads(secret['SecretString'])
```

**HashiCorp Vault:**
```python
import hvac

client = hvac.Client(url='https://vault.example.com')
credentials = client.secrets.kv.v2.read_secret_version(path='music-analyzer/spotify')
```

### Rotación de Credenciales

- Rota credenciales regularmente
- Implementa proceso automatizado
- Notifica antes de expiración

## 🚫 Validación de Inputs

### Validación con Pydantic

✅ **Correcto:**
```python
from pydantic import BaseModel, Field, validator

class AnalyzeRequest(BaseModel):
    track_id: str = Field(..., min_length=1, max_length=50, regex="^[a-zA-Z0-9]+$")
    include_coaching: bool = False
    
    @validator('track_id')
    def validate_track_id(cls, v):
        if not v.isalnum():
            raise ValueError('track_id must be alphanumeric')
        return v
```

❌ **Incorrecto:**
```python
@router.post("/analyze")
async def analyze(track_id: str):  # Sin validación
    # Vulnerable a inyección
    pass
```

### Sanitización

```python
import re

def sanitize_query(query: str) -> str:
    # Remover caracteres peligrosos
    query = re.sub(r'[<>"\']', '', query)
    # Limitar longitud
    query = query[:100]
    return query.strip()
```

## 🚦 Rate Limiting

### Configuración

```env
RATE_LIMIT_ENABLED=True
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
```

### Implementación

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/music/analyze")
@limiter.limit("10/minute")
async def analyze_track(request: Request):
    # ...
```

### Rate Limiting por Usuario

```python
def get_user_id(request: Request):
    # Obtener user_id del token
    token = request.headers.get("Authorization")
    user_id = decode_token(token)
    return user_id

@app.post("/music/analyze")
@limiter.limit("50/hour", key_func=get_user_id)
async def analyze_track(request: Request):
    # ...
```

## 🔑 Autenticación y Autorización

### JWT Tokens

```python
import jwt
from datetime import datetime, timedelta

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"

def create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Protección de Endpoints

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Security(security)):
    payload = verify_token(token.credentials)
    return payload["user_id"]

@app.post("/music/analyze")
async def analyze_track(
    request: AnalyzeRequest,
    user_id: str = Depends(get_current_user)
):
    # Solo usuarios autenticados
    # ...
```

## 🔒 Encriptación

### Encriptación de Datos Sensibles

```python
from cryptography.fernet import Fernet

# Generar clave (hacerlo una vez)
key = Fernet.generate_key()
cipher = Fernet(key)

# Encriptar
encrypted_data = cipher.encrypt(b"sensitive data")

# Desencriptar
decrypted_data = cipher.decrypt(encrypted_data)
```

### HTTPS/TLS

Siempre usa HTTPS en producción:

```nginx
# Nginx configuration
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
}
```

## 🛡️ Protección contra Ataques Comunes

### SQL Injection

✅ **Correcto (usando ORM):**
```python
# Usando SQLAlchemy
track = session.query(Track).filter(Track.id == track_id).first()
```

❌ **Incorrecto:**
```python
# Vulnerable a SQL injection
query = f"SELECT * FROM tracks WHERE id = '{track_id}'"
```

### XSS (Cross-Site Scripting)

```python
from markupsafe import escape

# Escapar output
safe_output = escape(user_input)
```

### CSRF (Cross-Site Request Forgery)

```python
from fastapi_csrf_protect import CsrfProtect

@CsrfProtect.load_config
def get_csrf_config():
    return CsrfSettings(secret_key=os.getenv("CSRF_SECRET_KEY"))

@app.post("/music/analyze")
async def analyze_track(
    request: Request,
    csrf_protect: CsrfProtect = Depends()
):
    await csrf_protect.validate_csrf(request)
    # ...
```

### DDoS Protection

```python
# Rate limiting agresivo
@limiter.limit("5/minute")
async def expensive_operation():
    # Operaciones costosas
    pass

# Timeouts
import asyncio

async def with_timeout(coro, timeout=30):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
```

## 📊 Logging y Monitoreo

### Logging Seguro

✅ **Correcto:**
```python
logger.info("track_analyzed", track_id=track_id, user_id=user_id)
```

❌ **Incorrecto:**
```python
logger.info(f"Auth: {client_secret}")  # NUNCA loggear secrets
logger.info(f"User data: {user_data}")  # Cuidado con datos sensibles
```

### Monitoreo de Seguridad

```python
# Detectar intentos de acceso no autorizado
failed_attempts = {}

async def check_brute_force(ip_address: str):
    if ip_address in failed_attempts:
        failed_attempts[ip_address] += 1
        if failed_attempts[ip_address] > 5:
            logger.warning("brute_force_detected", ip=ip_address)
            raise HTTPException(status_code=429, detail="Too many failed attempts")
    else:
        failed_attempts[ip_address] = 1
```

## 🔍 Auditoría

### Logs de Auditoría

```python
def audit_log(action: str, user_id: str, details: dict):
    logger.info("audit", 
        action=action,
        user_id=user_id,
        timestamp=datetime.utcnow().isoformat(),
        details=details
    )

# Uso
audit_log("track_analyzed", user_id, {"track_id": track_id})
```

## 🚨 Incident Response

### Plan de Respuesta

1. **Identificar**: Detectar el incidente
2. **Contener**: Limitar el daño
3. **Eradicar**: Eliminar la amenaza
4. **Recuperar**: Restaurar servicios
5. **Aprender**: Documentar y mejorar

### Checklist de Incidente

- [ ] Identificar tipo de ataque
- [ ] Bloquear IPs maliciosas
- [ ] Revocar credenciales comprometidas
- [ ] Notificar a usuarios afectados
- [ ] Documentar incidente
- [ ] Implementar medidas preventivas

## ✅ Checklist de Seguridad

### Desarrollo

- [ ] Sin credenciales hardcodeadas
- [ ] Validación de todos los inputs
- [ ] Rate limiting implementado
- [ ] Logging seguro
- [ ] Tests de seguridad

### Producción

- [ ] HTTPS habilitado
- [ ] Credenciales en secret manager
- [ ] Firewall configurado
- [ ] Monitoreo activo
- [ ] Backups encriptados
- [ ] Actualizaciones de seguridad
- [ ] Plan de respuesta a incidentes

## 📚 Recursos Adicionales

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Spotify Security Best Practices](https://developer.spotify.com/documentation/general/guides/security/)

---

**Última actualización**: 2025  
**Versión**: 2.21.0






