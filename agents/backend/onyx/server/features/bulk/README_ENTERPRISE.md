# 🏢 BUL API - Enterprise Features

## 🎯 Características Enterprise

### 🔐 Autenticación y Autorización
- ✅ JWT tokens
- ✅ Roles y permisos
- ✅ Password hashing con bcrypt
- ✅ Token expiration

### 📦 Cache Redis
- ✅ Cache distribuido
- ✅ TTL configurable
- ✅ Pattern-based clearing
- ✅ Estadísticas de cache

### ⚙️ Configuración
- ✅ Variables de entorno (.env)
- ✅ Pydantic Settings
- ✅ Configuración centralizada

### 🔄 Circuit Breaker
- ✅ Protección contra fallos en cascada
- ✅ Auto-recovery
- ✅ Estados: CLOSED, OPEN, HALF_OPEN

### 🚀 CI/CD
- ✅ GitHub Actions completo
- ✅ Tests automatizados
- ✅ Security scanning
- ✅ Docker builds
- ✅ Multi-environment deployment

## 📋 Instalación

### 1. Variables de Entorno

```bash
# Copiar template
cp .env.example .env

# Editar configuración
nano .env
```

### 2. Redis (Opcional)

```bash
# Docker
docker run -d -p 6379:6379 redis:alpine

# O instalar localmente
sudo apt-get install redis-server
```

### 3. Dependencias

```bash
pip install -r requirements.txt
python-jose[cryptography] passlib[bcrypt] redis
```

## 🔐 Autenticación

### Login

```python
from auth import authenticate_user, create_access_token
from datetime import timedelta

# Autenticar
user = authenticate_user("admin", "admin123")

# Crear token
access_token = create_access_token(
    data={"sub": user.username},
    expires_delta=timedelta(minutes=30)
)
```

### Usar en Endpoints

```python
from auth import get_current_active_user, require_role
from fastapi import Depends

@app.get("/api/protected")
async def protected_route(
    current_user: User = Depends(get_current_active_user)
):
    return {"message": f"Hello {current_user.username}"}

@app.get("/api/admin")
async def admin_route(
    current_user: User = Depends(require_role("admin"))
):
    return {"message": "Admin only"}
```

### En Cliente

```python
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(url, headers=headers)
```

## 📦 Cache Redis

### Uso Básico

```python
from cache_redis import cache_manager

# Guardar
cache_manager.set("key", "value", ttl=3600)

# Obtener
value = cache_manager.get("key")

# Eliminar
cache_manager.delete("key")
```

### Decorator

```python
from cache_redis import cache_result

@cache_result(ttl=1800, key_prefix="documents")
def get_document(id: str):
    # Función que se cachea automáticamente
    return expensive_operation(id)
```

### Stats

```python
stats = cache_manager.get_stats()
print(f"Cache hits: {stats['keyspace_hits']}")
```

## ⚙️ Configuración

### Settings

```python
from config import settings

print(settings.api_host)
print(settings.redis_host)
print(settings.secret_key)
```

### Variables Importantes

```bash
# Security
SECRET_KEY=change-this-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL=3600

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## 🔄 Circuit Breaker

### Uso

```python
from circuit_breaker import circuit_breaker

@circuit_breaker(failure_threshold=5, recovery_timeout=60)
def risky_external_api_call():
    # Si falla 5 veces, se abre el circuit
    # Después de 60s, intenta recovery
    return external_api.get_data()

# O manualmente
from circuit_breaker import document_generation_breaker

try:
    result = document_generation_breaker.call(
        generate_document,
        query="..."
    )
except Exception as e:
    print("Circuit is open")
```

### Estado

```python
state = document_generation_breaker.get_state()
print(state)
# {
#   "state": "closed",
#   "failure_count": 0,
#   "last_failure_time": None,
#   "success_count": 0
# }
```

## 🚀 CI/CD

### GitHub Actions

El workflow incluye:

1. **Test Suite**
   - Linter (flake8, black, mypy)
   - Unit tests (pytest)
   - API tests
   - Coverage reports

2. **Build**
   - Docker image build
   - Push to registry
   - Multi-tag support

3. **Security**
   - Trivy vulnerability scanning
   - SARIF upload

4. **Deploy**
   - Staging (develop branch)
   - Production (main branch)

### Configurar

1. **Secrets** (GitHub Settings → Secrets):
   - `GITHUB_TOKEN` - Auto-generated
   - Agregar otros según necesidad

2. **Environments**:
   - `staging`
   - `production`

3. **Ejecutar**:
   ```bash
   git push origin main  # Triggers CI/CD
   ```

## 📊 Monitoreo Enterprise

### Métricas Adicionales

- Circuit breaker states
- Cache hit/miss rates
- Auth success/failure rates
- Token usage

### Logging Estructurado

```python
from structured_logging import setup_structured_logging

logger = setup_structured_logging(
    log_level="INFO",
    log_file="logs/app.json",
    console_output=True
)
```

## 🔒 Seguridad

### Best Practices

1. **Cambiar SECRET_KEY** en producción
2. **Usar HTTPS** en producción
3. **Limitar CORS** origins
4. **Rate limiting** configurado
5. **Password hashing** con bcrypt
6. **Token expiration** configurado

### Roles

- `admin` - Acceso completo
- `user` - Acceso limitado
- Custom roles según necesidad

## 📝 Archivos Enterprise

- `auth.py` - Autenticación y autorización
- `cache_redis.py` - Cache Redis
- `config.py` - Configuración
- `circuit_breaker.py` - Circuit breaker
- `.env.example` - Template de variables
- `.github/workflows/ci-cd.yml` - CI/CD pipeline

---

**Estado**: ✅ **Enterprise Ready**
































