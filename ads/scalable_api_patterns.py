from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import asyncio
import aiohttp
import redis
import json
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import jwt
from contextlib import asynccontextmanager
from official_docs_reference import OfficialDocsReference
    import uvicorn
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Scalable API Development Patterns
================================

Patrones de desarrollo de APIs escalables usando FastAPI y el sistema
de referencias de documentación oficial.
"""


# Importar sistema de referencias

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de seguridad
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configuración de Redis
REDIS_URL = "redis://localhost:6379"
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Configuración de rate limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hora

# Modelos de datos
class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8)

class LoginRequest(BaseModel):
    username: str
    password: str

class APIRequest(BaseModel):
    library_name: str = Field(..., description="Nombre de la librería")
    api_name: Optional[str] = Field(None, description="Nombre de la API")
    category: Optional[str] = Field(None, description="Categoría")
    version: Optional[str] = Field(None, description="Versión")

class BatchRequest(BaseModel):
    requests: List[APIRequest] = Field(..., max_items=100)
    priority: str = Field("normal", regex="^(low|normal|high)$")

class CacheConfig(BaseModel):
    ttl: int = Field(300, description="Time to live en segundos")
    max_size: int = Field(1000, description="Tamaño máximo del cache")

# Middleware y utilidades
class RateLimiter:
    """Middleware para rate limiting."""
    
    def __init__(self, requests_per_hour: int = RATE_LIMIT_REQUESTS):
        
    """__init__ function."""
self.requests_per_hour = requests_per_hour
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Verificar rate limit para un cliente."""
        key = f"rate_limit:{client_id}"
        current = await redis_client.incr(key)
        
        if current == 1:
            await redis_client.expire(key, RATE_LIMIT_WINDOW)
        
        return current <= self.requests_per_hour

class CacheManager:
    """Gestor de cache distribuido."""
    
    def __init__(self, redis_client, default_ttl: int = 300):
        
    """__init__ function."""
self.redis = redis_client
        self.default_ttl = default_ttl
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generar clave de cache."""
        data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Dict]:
        """Obtener valor del cache."""
        try:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def set(self, key: str, value: Dict, ttl: int = None) -> bool:
        """Establecer valor en cache."""
        try:
            ttl = ttl or self.default_ttl
            await self.redis.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidar cache por patrón."""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0

class AuthenticationManager:
    """Gestor de autenticación."""
    
    def __init__(self, secret_key: str, algorithm: str = ALGORITHM):
        
    """__init__ function."""
self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, data: Dict, expires_delta: timedelta = None) -> str:
        """Crear token de acceso."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verificar token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            return None

# Configuración de la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación."""
    # Startup
    logger.info("🚀 Iniciando aplicación FastAPI escalable...")
    
    # Verificar conexión a Redis
    try:
        await redis_client.ping()
        logger.info("✅ Conexión a Redis establecida")
    except Exception as e:
        logger.error(f"❌ Error conectando a Redis: {e}")
    
    # Inicializar sistema de referencias
    app.state.ref_system = OfficialDocsReference()
    app.state.cache_manager = CacheManager(redis_client)
    app.state.auth_manager = AuthenticationManager(SECRET_KEY)
    app.state.rate_limiter = RateLimiter()
    
    yield
    
    # Shutdown
    logger.info("🛑 Cerrando aplicación...")

# Crear aplicación FastAPI
app = FastAPI(
    title="Scalable Official Docs Reference API",
    description="API escalable para referencias de documentación oficial",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Dependencias
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Obtener usuario actual."""
    token = credentials.credentials
    payload = app.state.auth_manager.verify_token(token)
    
    if payload is None:
        raise HTTPException(status_code=401, detail="Token inválido")
    
    return payload

async def get_rate_limited_user(request: Request):
    """Obtener usuario con rate limiting."""
    client_id = request.client.host
    
    if not await app.state.rate_limiter.check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit excedido. Intenta de nuevo más tarde."
        )
    
    return client_id

# Decoradores
def cache_response(ttl: int = 300):
    """Decorador para cachear respuestas."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generar clave de cache
            cache_key = app.state.cache_manager.generate_key(func.__name__, *args, **kwargs)
            
            # Intentar obtener del cache
            cached_result = await app.state.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {cache_key}")
                return cached_result
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            await app.state.cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def async_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorador para reintentos asíncronos."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            raise last_exception
        return wrapper
    return decorator

# Endpoints principales
@app.get("/")
async def root():
    """Endpoint raíz."""
    return {
        "message": "Scalable Official Docs Reference API",
        "version": "2.0.0",
        "features": [
            "Rate limiting",
            "Distributed caching",
            "Authentication",
            "Batch processing",
            "Async operations",
            "Error handling"
        ]
    }

@app.post("/auth/register")
async def register_user(user: User):
    """Registrar nuevo usuario."""
    # En producción, hashear password y guardar en base de datos
    user_hash = hashlib.sha256(user.password.encode()).hexdigest()
    
    # Simular guardado en base de datos
    user_data = {
        "username": user.username,
        "email": user.email,
        "password_hash": user_hash,
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Guardar en Redis temporalmente
    await redis_client.setex(
        f"user:{user.username}",
        3600,  # 1 hora
        json.dumps(user_data)
    )
    
    return {"message": "Usuario registrado exitosamente"}

@app.post("/auth/login")
async def login(login_request: LoginRequest):
    """Iniciar sesión."""
    # Obtener usuario de Redis
    user_data = await redis_client.get(f"user:{login_request.username}")
    
    if not user_data:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    
    user = json.loads(user_data)
    password_hash = hashlib.sha256(login_request.password.encode()).hexdigest()
    
    if password_hash != user["password_hash"]:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    
    # Crear token
    access_token = app.state.auth_manager.create_access_token(
        data={"sub": user["username"]}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/library/info")
@cache_response(ttl=3600)  # Cache por 1 hora
async def get_library_info(
    request: APIRequest,
    current_user: Dict = Depends(get_current_user),
    client_id: str = Depends(get_rate_limited_user)
):
    """Obtener información de librería con cache y rate limiting."""
    try:
        lib_info = app.state.ref_system.get_library_info(request.library_name)
        if not lib_info:
            raise HTTPException(
                status_code=404,
                detail=f"Librería '{request.library_name}' no encontrada"
            )
        
        return {
            "success": True,
            "library": {
                "name": lib_info.name,
                "current_version": lib_info.current_version,
                "min_supported_version": lib_info.min_supported_version,
                "documentation_url": lib_info.documentation_url,
                "github_url": lib_info.github_url,
                "pip_package": lib_info.pip_package,
                "conda_package": lib_info.conda_package
            },
            "cached": True,
            "user": current_user["sub"]
        }
    except Exception as e:
        logger.error(f"Error getting library info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reference")
@cache_response(ttl=1800)  # Cache por 30 minutos
async def get_api_reference(
    request: APIRequest,
    current_user: Dict = Depends(get_current_user),
    client_id: str = Depends(get_rate_limited_user)
):
    """Obtener referencia de API con cache."""
    try:
        if not request.api_name:
            raise HTTPException(status_code=400, detail="api_name es requerido")
        
        api_ref = app.state.ref_system.get_api_reference(
            request.library_name, 
            request.api_name
        )
        
        if not api_ref:
            raise HTTPException(
                status_code=404,
                detail=f"API '{request.api_name}' no encontrada"
            )
        
        return {
            "success": True,
            "api_reference": {
                "name": api_ref.name,
                "description": api_ref.description,
                "official_docs_url": api_ref.official_docs_url,
                "code_example": api_ref.code_example,
                "best_practices": api_ref.best_practices,
                "performance_tips": api_ref.performance_tips
            },
            "cached": True,
            "user": current_user["sub"]
        }
    except Exception as e:
        logger.error(f"Error getting API reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/process")
async def process_batch_requests(
    batch_request: BatchRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Procesar múltiples requests en batch."""
    try:
        # Procesar requests en background
        background_tasks.add_task(
            process_batch_background,
            batch_request.requests,
            current_user["sub"],
            batch_request.priority
        )
        
        return {
            "success": True,
            "message": f"Batch processing started for {len(batch_request.requests)} requests",
            "priority": batch_request.priority,
            "user": current_user["sub"]
        }
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_batch_background(requests: List[APIRequest], user: str, priority: str):
    """Procesar batch en background."""
    logger.info(f"Processing {len(requests)} requests for user {user} with priority {priority}")
    
    results = []
    for i, req in enumerate(requests):
        try:
            # Simular procesamiento
            await asyncio.sleep(0.1)  # Simular trabajo
            
            if req.api_name:
                result = app.state.ref_system.get_api_reference(req.library_name, req.api_name)
            else:
                result = app.state.ref_system.get_library_info(req.library_name)
            
            results.append({
                "request_id": i,
                "success": True,
                "result": result
            })
            
        except Exception as e:
            results.append({
                "request_id": i,
                "success": False,
                "error": str(e)
            })
    
    # Guardar resultados en Redis
    await redis_client.setex(
        f"batch_results:{user}:{int(time.time())}",
        3600,  # 1 hora
        json.dumps(results)
    )
    
    logger.info(f"Batch processing completed for user {user}")

@app.get("/stream/updates")
async def stream_updates():
    """Stream de actualizaciones en tiempo real."""
    async def generate():
        
    """generate function."""
while True:
            # Simular actualizaciones
            update = {
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Update from scalable API",
                "metrics": {
                    "active_connections": len(app.state.get("active_connections", [])),
                    "cache_hit_rate": 0.85,
                    "response_time_avg": 0.15
                }
            }
            
            yield f"data: {json.dumps(update)}\n\n"
            await asyncio.sleep(5)  # Actualizar cada 5 segundos
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/cache/invalidate")
async def invalidate_cache(
    pattern: str,
    current_user: Dict = Depends(get_current_user)
):
    """Invalidar cache por patrón."""
    try:
        count = await app.state.cache_manager.invalidate_pattern(pattern)
        return {
            "success": True,
            "message": f"Cache invalidated",
            "pattern": pattern,
            "invalidated_keys": count
        }
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Obtener métricas de la aplicación."""
    try:
        # Métricas básicas
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": time.time() - app.state.get("start_time", time.time()),
            "memory_usage": "N/A",  # En producción, obtener de sistema
            "active_connections": len(app.state.get("active_connections", [])),
            "cache_stats": {
                "keys": await redis_client.dbsize(),
                "memory": "N/A"
            }
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Middleware de logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para logging detallado."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    # Agregar headers de métricas
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Cache-Hit"] = "true" if "cached" in str(response.body) else "false"
    
    return response

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler para excepciones HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler para excepciones generales."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )

# Función para ejecutar el servidor
def run_scalable_server(host: str = "0.0.0.0", port: int = 8001, workers: int = 4):
    """Ejecutar servidor escalable."""
    
    uvicorn.run(
        "scalable_api_patterns:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )

match __name__:
    case "__main__":
    run_scalable_server() 