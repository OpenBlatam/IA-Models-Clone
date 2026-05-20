# 🚀 Mejores Prácticas para APIs - Guía Completa

## 📋 **Índice**
1. [Estructura y Diseño](#estructura-y-diseño)
2. [Validación y Esquemas](#validación-y-esquemas)
3. [Manejo de Errores](#manejo-de-errores)
4. [Seguridad](#seguridad)
5. [Performance](#performance)
6. [Documentación](#documentación)
7. [Testing](#testing)
8. [Monitoreo](#monitoreo)

---

## 🏗️ **Estructura y Diseño**

### **1. Convenciones de Nomenclatura**
```python
# ✅ BUENO - RESTful y consistente
GET    /api/v1/content/analyze
POST   /api/v1/content/compare
GET    /api/v1/trends/{timeframe}
DELETE /api/v1/reports/{report_id}

# ❌ MALO - Inconsistente
GET    /analyzeContent
POST   /compare_content
GET    /getTrends
DELETE /removeReport
```

### **2. Versionado de API**
```python
# Versión en URL
/api/v1/content/analyze
/api/v2/content/analyze

# Versión en headers
Accept: application/vnd.api+json;version=1
```

### **3. Estructura de Respuesta Estándar**
```python
{
    "success": true,
    "data": {...},
    "message": "Operation completed successfully",
    "errors": null,
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789",
    "version": "v1"
}
```

---

## ✅ **Validación y Esquemas**

### **1. Validación con Pydantic**
```python
from pydantic import BaseModel, Field, validator

class ContentAnalysisRequest(BaseModel):
    content: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="Content to analyze"
    )
    model_version: Optional[str] = Field(None)
    analysis_type: str = Field(default="comprehensive")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
```

### **2. Validación de Parámetros**
```python
# Parámetros de consulta
@app.get("/api/v1/content")
async def get_content(
    page: int = Field(1, ge=1),
    size: int = Field(20, ge=1, le=100),
    sort: Optional[str] = None
):
    pass
```

---

## 🚨 **Manejo de Errores**

### **1. Códigos de Estado HTTP**
```python
# 200 - OK
# 201 - Created
# 400 - Bad Request (validación)
# 401 - Unauthorized
# 403 - Forbidden
# 404 - Not Found
# 429 - Too Many Requests
# 500 - Internal Server Error
```

### **2. Respuestas de Error Estándar**
```python
{
    "success": false,
    "message": "Validation failed",
    "error_code": "VALIDATION_ERROR",
    "errors": [
        "Content cannot be empty",
        "Model version is required"
    ],
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
}
```

### **3. Manejo de Excepciones**
```python
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation failed",
            "errors": exc.errors()
        }
    )
```

---

## 🔒 **Seguridad**

### **1. Autenticación**
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Validar JWT token
    token = credentials.credentials
    # ... validación del token
    return user
```

### **2. Autorización**
```python
async def require_permission(permission: str):
    def permission_checker(user: dict = Depends(get_current_user)):
        if permission not in user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return permission_checker
```

### **3. Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/analyze")
@limiter.limit("10/minute")
async def analyze_content(request: Request, ...):
    pass
```

### **4. CORS y Headers de Seguridad**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Headers de seguridad
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response
```

---

## ⚡ **Performance**

### **1. Caché**
```python
from functools import lru_cache
import redis

# Caché en memoria
@lru_cache(maxsize=1000)
def cached_analysis(content_hash: str):
    return perform_analysis(content_hash)

# Caché Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def get_cached_result(key: str):
    result = redis_client.get(key)
    if result:
        return json.loads(result)
    return None
```

### **2. Paginación**
```python
class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)
    sort: Optional[str] = None
    order: str = Field(default="asc", regex="^(asc|desc)$")

@app.get("/api/v1/content")
async def get_content(pagination: PaginationParams = Depends()):
    offset = (pagination.page - 1) * pagination.size
    # ... consulta a la base de datos
    return {
        "data": results,
        "pagination": {
            "page": pagination.page,
            "size": pagination.size,
            "total": total_count,
            "total_pages": (total_count + pagination.size - 1) // pagination.size
        }
    }
```

### **3. Compresión**
```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### **4. Async/Await**
```python
# ✅ BUENO - Async
@app.post("/api/v1/analyze")
async def analyze_content(request: ContentRequest):
    result = await perform_async_analysis(request.content)
    return result

# ❌ MALO - Síncrono
@app.post("/api/v1/analyze")
def analyze_content(request: ContentRequest):
    result = perform_sync_analysis(request.content)  # Bloquea
    return result
```

---

## 📚 **Documentación**

### **1. OpenAPI/Swagger**
```python
app = FastAPI(
    title="AI History Comparison API",
    description="Comprehensive API for AI content analysis",
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)
```

### **2. Documentación de Endpoints**
```python
@app.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    summary="Analyze content",
    description="Analyze content for quality, readability, and sentiment",
    tags=["Content Analysis"],
    responses={
        200: {"description": "Analysis completed successfully"},
        400: {"description": "Invalid request data"},
        500: {"description": "Internal server error"}
    }
)
async def analyze_content(request: ContentRequest):
    """
    Analyze content with comprehensive metrics.
    
    - **content**: The text content to analyze
    - **model_version**: Optional AI model version
    - **analysis_type**: Type of analysis to perform
    
    Returns detailed analysis results including:
    - Readability score
    - Sentiment analysis
    - Complexity metrics
    """
    pass
```

### **3. Ejemplos de Uso**
```python
# Incluir ejemplos en los modelos
class ContentRequest(BaseModel):
    content: str = Field(..., example="This is sample content to analyze")
    model_version: str = Field(None, example="gpt-4")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "This is sample content to analyze",
                "model_version": "gpt-4",
                "analysis_type": "comprehensive"
            }
        }
```

---

## 🧪 **Testing**

### **1. Tests Unitarios**
```python
import pytest
from fastapi.testclient import TestClient

client = TestClient(app)

def test_analyze_content():
    response = client.post(
        "/api/v1/analyze",
        json={
            "content": "Test content",
            "analysis_type": "basic"
        }
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
```

### **2. Tests de Integración**
```python
@pytest.mark.asyncio
async def test_full_analysis_workflow():
    # Test complete workflow
    response = await client.post("/api/v1/analyze", json=test_data)
    assert response.status_code == 200
    
    # Test comparison
    comparison_response = await client.post("/api/v1/compare", json=comparison_data)
    assert comparison_response.status_code == 200
```

---

## 📊 **Monitoreo**

### **1. Logging**
```python
import logging
from loguru import logger

# Configurar logging
logger.add(
    "api.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response
```

### **2. Métricas**
```python
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### **3. Health Checks**
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "checks": {
            "database": "healthy",
            "cache": "healthy",
            "external_apis": "healthy"
        }
    }
```

---

## 🎯 **Checklist de Mejores Prácticas**

### **✅ Estructura y Diseño**
- [ ] URLs RESTful y consistentes
- [ ] Versionado de API implementado
- [ ] Respuestas estandarizadas
- [ ] Paginación implementada

### **✅ Validación**
- [ ] Validación de entrada con Pydantic
- [ ] Validación de parámetros de consulta
- [ ] Sanitización de datos
- [ ] Límites de tamaño de payload

### **✅ Seguridad**
- [ ] Autenticación implementada
- [ ] Autorización por roles/permisos
- [ ] Rate limiting configurado
- [ ] Headers de seguridad
- [ ] CORS configurado correctamente

### **✅ Performance**
- [ ] Caché implementado
- [ ] Compresión habilitada
- [ ] Operaciones asíncronas
- [ ] Optimización de consultas DB

### **✅ Documentación**
- [ ] OpenAPI/Swagger configurado
- [ ] Ejemplos de uso incluidos
- [ ] Documentación de errores
- [ ] Guías de integración

### **✅ Testing**
- [ ] Tests unitarios
- [ ] Tests de integración
- [ ] Tests de carga
- [ ] Cobertura de código

### **✅ Monitoreo**
- [ ] Logging estructurado
- [ ] Métricas de performance
- [ ] Health checks
- [ ] Alertas configuradas

---

## 🚀 **Implementación Rápida**

Para implementar estas mejores prácticas en tu sistema:

1. **Instalar dependencias**:
```bash
pip install fastapi uvicorn pydantic loguru prometheus-client slowapi
```

2. **Usar el archivo de mejores prácticas**:
```python
from api_best_practices import create_optimized_fastapi_app
app = create_optimized_fastapi_app()
```

3. **Configurar variables de entorno**:
```bash
export API_VERSION=v1
export DEBUG=false
export RATE_LIMIT=100
```

4. **Ejecutar con optimizaciones**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

**¡Con estas mejores prácticas tendrás una API robusta, segura y de alto rendimiento!** 🎉







