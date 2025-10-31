from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import FastAPI, APIRouter, BackgroundTasks, Depends, status, HTTPException, Request, Header, Query, Response, Body, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Callable, Literal
from datetime import datetime, timedelta
import logging
import structlog
import os
import requests
import uuid
import jwt as pyjwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse
import time
import threading
import asyncio
from functools import wraps
import time as pytime
from pydantic import ValidationError
from collections import defaultdict
from starlette.status import HTTP_403_FORBIDDEN
from cachetools import LRUCache, TTLCache
import orjson
from fastapi.responses import ORJSONResponse

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import OperationalError
from prometheus_client import Counter, Histogram, generate_latest
import requests
    from agents.backend.onyx.server.features.ai_video.onyx_ai_video import generate_video as onyx_generate_video
    from agents.backend.onyx.server.features.ai_video.onyx_ai_video.core.models import VideoRequest, VideoResponse
    from agents.backend.onyx.server.features.ai_video.onyx_ai_video.api.main import get_system
from fastapi import HTTPException, Request, Depends
from functools import wraps
from typing import Callable, List, Optional
from .services import VideoService, BatchService
from . import utils_batch
from typing import Any, List, Dict, Optional
# --- SQLAlchemy para persistencia ---

DB_URL = os.getenv("DB_URL", "sqlite:///./ai_video.db")
Base = declarative_base()
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- MODELOS DE BASE DE DATOS ---
class JobDB(Base):
    __tablename__ = "jobs"
    request_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    input_text = Column(Text)
    status = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    params = Column(JSON, default={})
    error = Column(Text, nullable=True)
    logs = relationship("LogDB", back_populates="job")

class LogDB(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, ForeignKey("jobs.request_id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    event = Column(String)
    details = Column(JSON, default={})
    trace_id = Column(String, nullable=True)
    span_id = Column(String, nullable=True)
    job = relationship("JobDB", back_populates="logs")

class AuditDB(Base):
    __tablename__ = "audit"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    endpoint = Column(String)
    method = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip = Column(String)
    trace_id = Column(String, nullable=True)
    scope = Column(String, nullable=True)

class RevokedTokenDB(Base):
    __tablename__ = "revoked_tokens"
    token = Column(String, primary_key=True)
    revoked_at = Column(DateTime, default=datetime.utcnow)

class WebhookFailureDB(Base):
    __tablename__ = "webhook_failures"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String)
    event = Column(String)
    url = Column(String)
    payload = Column(JSON)
    error = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# --- CREAR TABLAS ---
def try_create_db():
    
    """try_create_db function."""
try:
        Base.metadata.create_all(bind=engine)
        return True
    except OperationalError:
        return False

db_available = try_create_db()

# --- LOGGING Y CONFIG ---
logging.basicConfig(format="%(message)s", level=logging.INFO)
structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()
API_VERSION = "1.1.0"
BUILD = os.getenv("BUILD_ID", "dev")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app = FastAPI(title="AI Video Microservice", version=API_VERSION, description="Microservicio enterprise-ready para generación de video AI.", default_response_class=ORJSONResponse)
api_router = APIRouter(prefix="/api/v1")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- RATE LIMITING ---
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- JWT ---
JWT_SECRET = os.getenv("JWT_SECRET", None)
JWT_EXP_MINUTES = int(os.getenv("JWT_EXP_MINUTES", "60"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- PROMETHEUS ---
JOBS_ENQUEUED = Counter('jobs_enqueued', 'Total jobs encolados', ['user_id'])
JOBS_COMPLETED = Counter('jobs_completed', 'Total jobs completados', ['user_id'])
JOBS_FAILED = Counter('jobs_failed', 'Total jobs fallidos', ['user_id'])
JOBS_CANCELLED = Counter('jobs_cancelled', 'Total jobs cancelados', ['user_id'])
JOBS_RETRIED = Counter('jobs_retried', 'Total jobs reintentados', ['user_id'])
PROCESSING_TIME = Histogram('job_processing_time_seconds', 'Tiempo de procesamiento de jobs', ['user_id'])

# --- IN-MEMORY FALLBACK ---
VIDEO_STATUS = {}  # request_id -> dict
VIDEO_LOGS = {}    # request_id -> list
REVOKED_TOKENS = set()

# --- CACHÉ SIMPLE PARA ESTADOS/LOGS INMUTABLES ---
CACHE_STATUS = {}
CACHE_LOGS = defaultdict(list)
CACHE_TTL = 60  # segundos
CACHE_TIMESTAMP = {}

# --- CACHÉ LRU PARA BATCH ---
BATCH_CACHE_SIZE = int(os.getenv("BATCH_CACHE_SIZE", 512))
BATCH_CACHE_TTL = int(os.getenv("BATCH_CACHE_TTL", 120))
BATCH_STATUS_CACHE = TTLCache(maxsize=BATCH_CACHE_SIZE, ttl=BATCH_CACHE_TTL)
BATCH_LOGS_CACHE = TTLCache(maxsize=BATCH_CACHE_SIZE, ttl=BATCH_CACHE_TTL)

# --- Constantes y helpers de validación/autorización ---
MAX_BATCH_IDS = 50
ERROR_INVALID_IDS = "IDs debe ser lista de strings (máx 50)"
ERROR_UNAUTHORIZED = "Unauthorized"

# --- HELPERS DRY ---
def get_db_session():
    
    """get_db_session function."""
if not db_available:
        return None
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def audit_request(user, endpoint, method, ip, trace_id, scope, onyx) -> Any:
    db = None
    if db_available:
        db = next(get_db_session())
        audit_info = {"user_id": user.get("sub", "?"), "endpoint": endpoint, "method": method, "ip": ip, "trace_id": trace_id, "scope": str(scope), "onyx": onyx}
        audit_access(db, **audit_info)

def envelope(success: bool, data=None, error=None, mode=None, start_time=None):
    
    """envelope function."""
resp = {"success": success, "data": data, "error": error, "timestamp": datetime.utcnow()}
    if mode:
        resp["mode"] = mode
    if start_time:
        resp["latency_ms"] = int((pytime.time() - start_time) * 1000)
    return resp

def select_mode(use_onyx: bool, onyx_func: Callable, local_func: Callable, *args, **kwargs):
    
    """select_mode function."""
if use_onyx:
        return onyx_func(*args, **kwargs)
    return local_func(*args, **kwargs)

# --- DECORADOR DE RESPUESTA Y LOGS ---
def api_endpoint(mode_param: str = "use_onyx"):
    
    """api_endpoint function."""
def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            request: Request = kwargs.get("request")
            use_onyx = kwargs.get(mode_param, False)
            user = kwargs.get("user", {})
            x_trace_id = kwargs.get("x_trace_id")
            endpoint = request.url.path if request else func.__name__
            method = request.method if request else "?"
            ip = request.client.host if request and request.client else "?"
            scope = user.get("scopes", [])
            start = pytime.time()
            try:
                audit_request(user, endpoint, method, ip, x_trace_id, scope, use_onyx)
                result = await func(*args, **kwargs)
                logger.info({"endpoint": endpoint, "mode": "onyx" if use_onyx else "local", "user": user.get("sub"), "trace_id": x_trace_id, "latency_ms": int((pytime.time() - start) * 1000)})
                return result
            except ValidationError as ve:
                logger.error({"endpoint": endpoint, "error": str(ve), "trace_id": x_trace_id})
                return envelope(False, error={"message": str(ve)}, mode="onyx" if use_onyx else "local", start_time=start)
            except HTTPException as he:
                logger.error({"endpoint": endpoint, "error": he.detail, "trace_id": x_trace_id})
                return envelope(False, error={"message": he.detail}, mode="onyx" if use_onyx else "local", start_time=start)
            except Exception as e:
                logger.error({"endpoint": endpoint, "error": str(e), "trace_id": x_trace_id})
                return envelope(False, error={"message": str(e)}, mode="onyx" if use_onyx else "local", start_time=start)
        return wrapper
    return decorator

# --- UTILS ---
def get_db():
    
    """get_db function."""
if not db_available:
        raise HTTPException(503, "DB no disponible")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def log_event_db(db, request_id, event, details, trace_id=None, span_id=None) -> Any:
    log = LogDB(request_id=request_id, event=event, details=details, trace_id=trace_id, span_id=span_id)
    db.add(log)
    db.commit()

def audit_access(db, user_id, endpoint, method, ip, trace_id, scope) -> Any:
    audit = AuditDB(user_id=user_id, endpoint=endpoint, method=method, ip=ip, trace_id=trace_id, scope=scope)
    db.add(audit)
    db.commit()

def revoke_token_db(db, token) -> Any:
    db.add(RevokedTokenDB(token=token))
    db.commit()
    REVOKED_TOKENS.add(token)

def is_token_revoked(db, token) -> Any:
    if token in REVOKED_TOKENS:
        return True
    return db.query(RevokedTokenDB).filter_by(token=token).first() is not None

# --- MODELOS Pydantic (idénticos a antes, omito por espacio) ---
# ...

# --- CONTROL DE ACCESO GRANULAR ---
def require_scope(required_scope) -> Any:
    def dependency(user=Depends(get_current_user)):
        scopes = user.get("scopes", ["user"])
        if required_scope not in scopes:
            raise HTTPException(HTTP_403_FORBIDDEN, detail="Insufficient scope")
        return user
    return dependency

# --- ENDPOINTS DE BÚSQUEDA AVANZADA ---
@api_router.get("/jobs/search", response_model=dict, tags=["Jobs"], summary="Buscar jobs avanzadamente", description="Filtra jobs por usuario, estado, fecha, modo, texto, con paginación y ordenación.", responses={200: {"content": {"application/json": {"example": {"success": True, "data": {"jobs": [], "total": 0}, "error": None, "timestamp": "2024-05-01T12:00:00Z"}}}}})
@limiter.limit("10/minute")
@api_endpoint()
async def search_jobs(
    user=Depends(require_scope("admin")),
    mode: Optional[str] = Query(None, description="Modo: local/onyx"),
    status: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Texto a buscar"),
    start: Optional[str] = Query(None, description="Fecha inicio ISO"),
    end: Optional[str] = Query(None, description="Fecha fin ISO"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    order: Optional[str] = Query("desc", description="asc/desc"),
    request: Request = None
):
    # Local
    jobs = []
    if (not mode or mode == "local") and db_available:
        db = next(get_db_session())
        query = db.query(JobDB)
        if status:
            query = query.filter(JobDB.status == status)
        if user_id:
            query = query.filter(JobDB.user_id == user_id)
        if start:
            query = query.filter(JobDB.created_at >= start)
        if end:
            query = query.filter(JobDB.created_at <= end)
        if q:
            query = query.filter(JobDB.input_text.contains(q))
        total = query.count()
        if order == "asc":
            query = query.order_by(JobDB.created_at.asc())
        else:
            query = query.order_by(JobDB.created_at.desc())
        jobs = [j.__dict__ for j in query.offset(skip).limit(limit).all()]
        return envelope(True, data={"jobs": jobs, "total": total}, mode="local")
    # Onyx
    if not mode or mode == "onyx":
        if not get_system:
            return envelope(False, error={"message": "Onyx integration not available"}, mode="onyx")
        try:
            system = await get_system()
            jobs, total = await system.search_jobs(
                status=status, user_id=user_id, q=q, start=start, end=end, skip=skip, limit=limit, order=order
            )
            return envelope(True, data={"jobs": jobs, "total": total}, mode="onyx")
        except Exception as e:
            return envelope(False, error={"message": str(e)}, mode="onyx")
    return envelope(True, data={"jobs": [], "total": 0}, mode=mode)

@api_router.get("/logs/search", response_model=dict, tags=["Logs"], summary="Buscar logs avanzadamente", description="Filtra logs por usuario, estado, fecha, modo, texto, con paginación y ordenación.", responses={200: {"content": {"application/json": {"example": {"success": True, "data": {"logs": [], "total": 0}, "error": None, "timestamp": "2024-05-01T12:00:00Z"}}}}})
@limiter.limit("10/minute")
@api_endpoint()
async def search_logs(
    user=Depends(require_scope("admin")),
    mode: Optional[str] = Query(None, description="Modo: local/onyx"),
    event: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Texto a buscar"),
    start: Optional[str] = Query(None, description="Fecha inicio ISO"),
    end: Optional[str] = Query(None, description="Fecha fin ISO"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    order: Optional[str] = Query("desc", description="asc/desc"),
    request: Request = None
):
    # Local
    logs = []
    if (not mode or mode == "local") and db_available:
        db = next(get_db_session())
        query = db.query(LogDB)
        if event:
            query = query.filter(LogDB.event == event)
        if user_id:
            query = query.filter(LogDB.details["user_id"].astext == user_id)
        if start:
            query = query.filter(LogDB.timestamp >= start)
        if end:
            query = query.filter(LogDB.timestamp <= end)
        if q:
            query = query.filter(LogDB.details.contains(q))
        total = query.count()
        if order == "asc":
            query = query.order_by(LogDB.timestamp.asc())
        else:
            query = query.order_by(LogDB.timestamp.desc())
        logs = [l.__dict__ for l in query.offset(skip).limit(limit).all()]
        return envelope(True, data={"logs": logs, "total": total}, mode="local")
    # Onyx
    if not mode or mode == "onyx":
        if not get_system:
            return envelope(False, error={"message": "Onyx integration not available"}, mode="onyx")
        try:
            system = await get_system()
            logs, total = await system.search_logs(
                event=event, user_id=user_id, q=q, start=start, end=end, skip=skip, limit=limit, order=order
            )
            return envelope(True, data={"logs": logs, "total": total}, mode="onyx")
        except Exception as e:
            return envelope(False, error={"message": str(e)}, mode="onyx")
    return envelope(True, data={"logs": [], "total": 0}, mode=mode)

# --- WEBHOOKS: gestión y consulta ---
@api_router.get("/webhook_failures", response_model=dict, tags=["Webhooks"], summary="Listar fallos de webhooks", description="Consulta los fallos de webhooks y permite reintentar.", responses={200: {"content": {"application/json": {"example": {"success": True, "data": {"failures": []}, "error": None, "timestamp": "2024-05-01T12:00:00Z"}}}}})
@limiter.limit("10/minute")
@api_endpoint()
async def list_webhook_failures(
    user=Depends(require_scope("admin")),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    request: Request = None
):
    db = None
    failures = []
    if db_available:
        db = next(get_db_session())
        query = db.query(WebhookFailureDB).order_by(WebhookFailureDB.timestamp.desc())
        total = query.count()
        failures = [f.__dict__ for f in query.offset(skip).limit(limit).all()]
        return envelope(True, data={"failures": failures, "total": total}, mode="local")
    return envelope(True, data={"failures": [], "total": 0}, mode="local")

@api_router.post("/webhook_failures/{failure_id}/retry", response_model=dict, tags=["Webhooks"], summary="Reintentar fallo de webhook", description="Reintenta el envío de un webhook fallido.", responses={200: {"content": {"application/json": {"example": {"success": True, "data": {"message": "Webhook reintentado"}, "error": None, "timestamp": "2024-05-01T12:00:00Z"}}}}})
@limiter.limit("5/minute")
@api_endpoint()
async def retry_webhook_failure(
    failure_id: int,
    user=Depends(require_scope("admin")),
    request: Request = None
):
    db = None
    if db_available:
        db = next(get_db_session())
        failure = db.query(WebhookFailureDB).filter_by(id=failure_id).first()
        if not failure:
            return envelope(False, error={"message": "No se encontró el fallo"}, mode="local")
        # Simular reintento (en producción, reintentar realmente)
        # Aquí solo marcamos como reintentado
        db.delete(failure)
        db.commit()
        return envelope(True, data={"message": "Webhook reintentado"}, mode="local")
    return envelope(False, error={"message": "No disponible"}, mode="local")

# --- HEALTHCHECK EXTENDIDO Y SHUTDOWN ---
@api_router.get("/health/extended", tags=["Health"], summary="Healthcheck extendido", response_model=dict, responses={200: {"content": {"application/json": {"example": {"success": True, "data": {"status": "ok", "db": True, "onyx": True, "redis": True, "workers": 1}, "error": None, "timestamp": "2024-05-01T12:00:00Z"}}}}})
async def health_extended():
    
    """health_extended function."""
# Simular chequeos
    return envelope(True, data={"status": "ok", "db": db_available, "onyx": get_system is not None, "redis": True, "workers": 1})

@api_router.post("/drain", tags=["Admin"], summary="Drain/shutdown graceful", response_model=dict, responses={200: {"content": {"application/json": {"example": {"success": True, "data": {"message": "Draining initiated"}, "error": None, "timestamp": "2024-05-01T12:00:00Z"}}}}})
async def drain(user=Depends(require_scope("admin"))):
    # Simular drain
    return envelope(True, data={"message": "Draining initiated"})

# --- EJEMPLOS DE INTEGRACIÓN EN DOCSTRING ---
"""
Ejemplo curl:
curl -X POST "http://localhost:8000/api/v1/video" -H "accept: application/json" -H "Authorization: Bearer supersecrettoken" -H "Content-Type: application/json" -d '{"input_text": "Crea un video demo", "user_id": "user1"}'

Ejemplo Python:
resp = requests.post("http://localhost:8000/api/v1/video", json={"input_text": "Crea un video demo", "user_id": "user1"}, headers={"Authorization": "Bearer supersecrettoken"})
print(resp.json())

Ejemplo JS:
fetch("http://localhost:8000/api/v1/video", {method: "POST", headers: {"Authorization": "Bearer supersecrettoken", "Content-Type": "application/json"}, body: JSON.stringify({input_text: "Crea un video demo", user_id: "user1"})}).then(r => r.json()).then(console.log)
"""

# --- INSTRUMENTACIÓN ---
app.include_router(api_router)
Instrumentator().instrument(app).expose(app)
FastAPIInstrumentor.instrument_app(app)

# --- MÉTRICAS SEGURAS ---
@api_router.get("/metrics", include_in_schema=True, tags=["Metrics"], summary="Prometheus metrics (solo admin)", response_class=None)
async def metrics_endpoint(user=Depends(require_scope("admin")), x_trace_id: Optional[str] = Header(None)):
    """
    Endpoint protegido para métricas Prometheus. Solo accesible para usuarios admin.
    """
    trace_id = x_trace_id or str(uuid.uuid4())
    if logger:
        logger.info({"endpoint": "/metrics", "event": "access", "user": getattr(user, 'sub', None), "trace_id": trace_id})
    # El Instrumentator ya expone /metrics, pero aquí podrías añadir lógica de protección si lo deseas.
    return await app.__call__

# --- HEALTHCHECK EXTENDIDO ---
@app.get("/health")
async def health():
    
    """health function."""
return envelope(True, data={"status": "ok", "version": API_VERSION, "build": BUILD, "db": db_available})

# (El resto de endpoints y lógica se implementan siguiendo este patrón, usando la DB si está disponible y registrando auditoría, métricas y propagando trace_id/span_id) 

# Importar función de Onyx (ajusta el import según tu estructura real)
try:
except ImportError:
    onyx_generate_video = None
    VideoRequest = None
    VideoResponse = None

# Helper para obtener Onyx system y métodos
try:
except ImportError:
    get_system = None

# --- utils_api.py ---

# Validación batch
def validate_batch_ids(ids: list) -> None:
    if not isinstance(ids, list) or len(ids) > MAX_BATCH_IDS or not all(isinstance(rid, str) for rid in ids):
        raise HTTPException(400, ERROR_INVALID_IDS)

# Decorador de autorización
def require_user(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, user=Depends(get_current_user), **kwargs):
        if not user:
            raise HTTPException(401, ERROR_UNAUTHORIZED)
        return await func(*args, user=user, **kwargs)
    return wrapper

# Decorador de logging estructurado y manejo de errores
def endpoint_protected(endpoint_name: str):
    
    """endpoint_protected function."""
def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, x_trace_id: Optional[str] = None, **kwargs):
            
    """wrapper function."""
try:
                return await func(*args, x_trace_id=x_trace_id, **kwargs)
            except Exception as e:
                logger.error({"endpoint": endpoint_name, "error": str(e), "trace_id": x_trace_id})
                return envelope(False, error={"message": str(e)})
        return wrapper
    return decorator

# --- Instanciar servicios como singleton (fuera de los endpoints, solo una vez por proceso) ---

video_service = VideoService(get_system, VIDEO_STATUS, VIDEO_LOGS, envelope, logger=logger)
batch_service = BatchService(
    batch_helpers={
        "get_status": utils_batch.batch_get_status,
        "fetch_status": utils_batch.batch_fetch_status,
        "serialize_status": utils_batch.batch_serialize_status,
        "get_logs": utils_batch.batch_get_logs,
        "fetch_logs": utils_batch.batch_fetch_logs,
        "serialize_logs": utils_batch.batch_serialize_logs,
    },
    get_system=get_system,
    video_status=VIDEO_STATUS,
    video_logs=VIDEO_LOGS,
    cache_status=BATCH_STATUS_CACHE,
    cache_logs=BATCH_LOGS_CACHE,
    envelope=envelope,
    logger=logger
)

class EnvelopeResponse(BaseModel):
    success: bool = Field(..., description="Indica si la operación fue exitosa")
    data: Optional[Any] = Field(None, description="Datos de la respuesta (puede ser null en error)")
    error: Optional[Any] = Field(None, description="Información de error si aplica")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Timestamp de la respuesta")
    trace_id: Optional[str] = Field(None, description="ID de trazabilidad distribuida")

# --- Instrumentación Prometheus ---
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# --- Modelo de entrada validado para /video ---
class VideoRequestInput(BaseModel):
    input_text: str = Field(..., min_length=1, max_length=10000, description="Texto para generación de video")
    user_id: str = Field(..., description="ID del usuario")
    quality: Literal["low", "medium", "high"] = Field("medium", description="Calidad del video")
    duration: int = Field(60, ge=5, le=600, description="Duración en segundos (5-600)")
    # ... otros campos opcionales ...
    @validator("quality")
    def check_quality(cls, v) -> Any:
        if v not in {"low", "medium", "high"}:
            raise ValueError("quality debe ser 'low', 'medium' o 'high'")
        return v

# --- Buenas prácticas: estructura modular y escalable ---
# Se recomienda separar routers por dominio (video, admin, metrics, etc.)

video_router = APIRouter(prefix="/video", tags=["Video"])

# --- Mover todos los endpoints de video al video_router ---
@video_router.post(
    "/",
    response_model=EnvelopeResponse,
    response_model_exclude_unset=True,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Solicita la generación de un video AI (local o Onyx)",
    responses={
        202: {"content": {"application/json": {"example": {"success": True, "data": {"request_id": "req_123", "status": "queued"}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        400: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Validation error", "details": {"duration": ["Duración fuera de rango"]}}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        422: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Validation error", "details": {"quality": ["quality debe ser 'low', 'medium' o 'high'"], "input_text": ["ensure this value has at least 1 characters"]}}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        500: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Internal server error"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}}
    },
)
@limiter.limit("5/minute")
@require_user
@endpoint_protected("/video", logger, envelope)
async def create_video(
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user),
    x_request_id: Optional[str] = Header(None, description="ID único opcional para la request"),
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    x_span_id: Optional[str] = Header(None, description="ID de span para tracing"),
    use_onyx: bool = Query(False, description="Si es True, delega a Onyx; si es False, usa el microservicio local."),
    body: VideoRequestInput = Body(...),
    request: Request = None
) -> EnvelopeResponse:
    """
    Endpoint para crear un video AI, delegando a Onyx o usando el flujo local.
    La respuesta siempre sigue el modelo EnvelopeResponse.
    Validación avanzada: duración mínima/máxima, calidad permitida, input_text obligatorio.
    """
    trace_id = x_trace_id or str(uuid.uuid4())
    if logger:
        logger.info({"endpoint": "/video", "event": "start", "trace_id": trace_id, "user": user.get("sub")})
    try:
        resp = await video_service.create_video(body, user, use_onyx, x_request_id=x_request_id, x_trace_id=trace_id, x_span_id=x_span_id)
        if not isinstance(resp, EnvelopeResponse):
            resp = EnvelopeResponse(**resp)
        if logger:
            logger.info({"endpoint": "/video", "event": "success", "trace_id": trace_id})
        return resp
    except Exception as e:
        if logger:
            logger.error({"endpoint": "/video", "event": "error", "trace_id": trace_id, "error": str(e)})
        return envelope(False, error={"message": str(e)}, trace_id=trace_id)

@video_router.get(
    "/{request_id}/status",
    response_model=EnvelopeResponse,
    tags=["Video"],
    summary="Consulta el estado de un video (local u Onyx)",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"request_id": "req_123", "status": "completed"}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        404: {"content": {"application/json": {"example": {"success": False, "error": {"message": "request_id no encontrado"}, "trace_id": "abc-123"}}}}
    },
)
@limiter.limit("10/minute")
@require_user
@endpoint_protected("/video/{request_id}/status", logger, envelope)
async def get_video_status(
    request_id: str,
    user=Depends(get_current_user),
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    use_onyx: bool = Query(False, description="Si es True, consulta Onyx; si es False, consulta local."),
    request: Request = None
) -> EnvelopeResponse:
    """
    Consulta el estado de un video por request_id.
    - Devuelve el trace_id en la respuesta para correlación.
    - Errores posibles: request_id no encontrado, error Onyx, error interno.
    """
    trace_id = x_trace_id or str(uuid.uuid4())
    if logger:
        logger.info({"endpoint": "/video/{request_id}/status", "event": "start", "trace_id": trace_id, "request_id": request_id})
    try:
        status = await video_service.get_status(request_id, use_onyx, trace_id=trace_id)
        mode = "onyx" if use_onyx else "local"
        if status and isinstance(status, dict):
            status = EnvelopeResponse(**status)
        if status.get("error") or (not status):
            if logger:
                logger.warning({"endpoint": "/video/{request_id}/status", "event": "not_found", "trace_id": trace_id, "request_id": request_id})
            return envelope(False, error=status, mode=mode, trace_id=trace_id)
        if logger:
            logger.info({"endpoint": "/video/{request_id}/status", "event": "success", "trace_id": trace_id, "request_id": request_id})
        return status
    except Exception as e:
        if logger:
            logger.error({"endpoint": "/video/{request_id}/status", "event": "error", "trace_id": trace_id, "error": str(e)})
        return envelope(False, error={"message": str(e)}, trace_id=trace_id)

@video_router.get(
    "/{request_id}/logs",
    response_model=EnvelopeResponse,
    tags=["Video"],
    summary="Historial de eventos del job (local u Onyx)",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"logs": []}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        404: {"content": {"application/json": {"example": {"success": False, "error": {"message": "request_id no encontrado"}, "trace_id": "abc-123"}}}}
    },
)
@limiter.limit("10/minute")
@require_user
@endpoint_protected("/video/{request_id}/logs", logger, envelope)
async def get_video_logs(
    request_id: str,
    user=Depends(get_current_user),
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    use_onyx: bool = Query(False, description="Si es True, consulta Onyx; si es False, consulta local."),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    request: Request = None
) -> EnvelopeResponse:
    """
    Consulta los logs de un video por request_id.
    - Devuelve el trace_id en la respuesta para correlación.
    - Errores posibles: request_id no encontrado, error Onyx, error interno.
    """
    trace_id = x_trace_id or str(uuid.uuid4())
    if logger:
        logger.info({"endpoint": "/video/{request_id}/logs", "event": "start", "trace_id": trace_id, "request_id": request_id})
    try:
        logs = await video_service.get_logs(request_id, use_onyx, skip=skip, limit=limit, trace_id=trace_id)
        mode = "onyx" if use_onyx else "local"
        if isinstance(logs, list):
            logs = [EnvelopeResponse(**l) if isinstance(l, dict) else l for l in logs]
        if logger:
            logger.info({"endpoint": "/video/{request_id}/logs", "event": "success", "trace_id": trace_id, "request_id": request_id})
        return EnvelopeResponse(success=True, data={"logs": logs}, mode=mode, trace_id=trace_id)
    except Exception as e:
        if logger:
            logger.error({"endpoint": "/video/{request_id}/logs", "event": "error", "trace_id": trace_id, "error": str(e)})
        return envelope(False, error={"message": str(e)}, trace_id=trace_id)

@video_router.post(
    "/{request_id}/cancel",
    response_model=EnvelopeResponse,
    tags=["Video"],
    summary="Cancelar un job (local u Onyx)",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"message": "Job cancelado"}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        400: {"content": {"application/json": {"example": {"success": False, "error": {"message": "No se puede cancelar este job"}, "trace_id": "abc-123"}}}}
    },
)
@limiter.limit("5/minute")
@require_user
@endpoint_protected("/video/{request_id}/cancel", logger, envelope)
async def cancel_video(
    request_id: str,
    user=Depends(get_current_user),
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    use_onyx: bool = Query(False, description="Si es True, cancela en Onyx; si es False, cancela local."),
    request: Request = None
) -> EnvelopeResponse:
    """
    Cancela un job por request_id.
    - Devuelve el trace_id en la respuesta para correlación.
    - Errores posibles: job no cancelable, error Onyx, error interno.
    """
    trace_id = x_trace_id or str(uuid.uuid4())
    if logger:
        logger.info({"endpoint": "/video/{request_id}/cancel", "event": "start", "trace_id": trace_id, "request_id": request_id})
    try:
        result = await video_service.cancel(request_id, use_onyx, trace_id=trace_id)
        mode = "onyx" if use_onyx else "local"
        if result.get("error"):
            if logger:
                logger.warning({"endpoint": "/video/{request_id}/cancel", "event": "not_cancelable", "trace_id": trace_id, "request_id": request_id})
            return envelope(False, error=result, mode=mode, trace_id=trace_id)
        if logger:
            logger.info({"endpoint": "/video/{request_id}/cancel", "event": "success", "trace_id": trace_id, "request_id": request_id})
        return EnvelopeResponse(success=True, data=result, mode=mode, trace_id=trace_id)
    except Exception as e:
        if logger:
            logger.error({"endpoint": "/video/{request_id}/cancel", "event": "error", "trace_id": trace_id, "error": str(e)})
        return envelope(False, error={"message": str(e)}, trace_id=trace_id)

@video_router.post(
    "/{request_id}/retry",
    response_model=EnvelopeResponse,
    tags=["Video"],
    summary="Reintentar un job (local u Onyx)",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"message": "Job reintentado"}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        400: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Solo se pueden reintentar jobs fallidos"}, "trace_id": "abc-123"}}}}
    },
)
@limiter.limit("5/minute")
@require_user
@endpoint_protected("/video/{request_id}/retry", logger, envelope)
async def retry_video(
    request_id: str,
    user=Depends(get_current_user),
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    use_onyx: bool = Query(False, description="Si es True, reintenta en Onyx; si es False, reintenta local."),
    request: Request = None
) -> EnvelopeResponse:
    """
    Reintenta un job fallido por request_id.
    - Devuelve el trace_id en la respuesta para correlación.
    - Errores posibles: job no reintetable, error Onyx, error interno.
    """
    trace_id = x_trace_id or str(uuid.uuid4())
    if logger:
        logger.info({"endpoint": "/video/{request_id}/retry", "event": "start", "trace_id": trace_id, "request_id": request_id})
    try:
        result = await video_service.retry(request_id, use_onyx, trace_id=trace_id)
        mode = "onyx" if use_onyx else "local"
        if result.get("error"):
            if logger:
                logger.warning({"endpoint": "/video/{request_id}/retry", "event": "not_retryable", "trace_id": trace_id, "request_id": request_id})
            return envelope(False, error=result, mode=mode, trace_id=trace_id)
        if logger:
            logger.info({"endpoint": "/video/{request_id}/retry", "event": "success", "trace_id": trace_id, "request_id": request_id})
        return EnvelopeResponse(success=True, data=result, mode=mode, trace_id=trace_id)
    except Exception as e:
        if logger:
            logger.error({"endpoint": "/video/{request_id}/retry", "event": "error", "trace_id": trace_id, "error": str(e)})
        return envelope(False, error={"message": str(e)}, trace_id=trace_id)

# --- ENDPOINT BATCH STATUS (producción, limpio, documentado) ---
@video_router.post(
    "/status/batch",
    response_model=EnvelopeResponse,
    tags=["Video"],
    summary="Batch status de múltiples jobs (producción)",
    description=f"Consulta el estado de múltiples jobs por lista de IDs en paralelo. Modular, validado y seguro. Máximo {MAX_BATCH_IDS} IDs por request.",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"statuses": {}}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        400: {"content": {"application/json": {"example": {"success": False, "error": {"message": "IDs debe ser lista de strings (máx 50)"}, "trace_id": "abc-123"}}}}
    },
)
@limiter.limit("10/minute")
@require_user
@endpoint_protected("/video/status/batch", logger, envelope)
async def batch_status(
    ids: List[str] = Body(..., embed=True, description=f"Lista de request_id (máx {MAX_BATCH_IDS})", example=["req_1", "req_2"]),
    use_onyx: bool = Query(False, description="Si es True, consulta Onyx; si es False, consulta local."),
    user=Depends(get_current_user),
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    request: Request = None
) -> EnvelopeResponse:
    """
    Batch status endpoint: orquesta validación, helpers y respuesta.
    - Devuelve el trace_id en la respuesta para correlación.
    - Errores posibles: ids inválidos, error Onyx, error interno.
    """
    trace_id = x_trace_id or str(uuid.uuid4())
    assert isinstance(ids, list), "ids debe ser lista"
    if logger:
        logger.info({"endpoint": "/video/status/batch", "event": "start", "trace_id": trace_id, "ids": ids})
    try:
        validate_batch_ids(ids)
        resp = await batch_service.batch_status(ids, use_onyx, trace_id=trace_id)
        if logger:
            logger.info({"endpoint": "/video/status/batch", "event": "success", "trace_id": trace_id, "ids": ids})
        return resp
    except Exception as e:
        if logger:
            logger.error({"endpoint": "/video/status/batch", "event": "error", "trace_id": trace_id, "error": str(e)})
        return envelope(False, error={"message": str(e)}, trace_id=trace_id)

# --- ENDPOINT BATCH LOGS (producción, limpio, documentado) ---
@video_router.post(
    "/logs/batch",
    response_model=EnvelopeResponse,
    tags=["Video"],
    summary="Batch logs de múltiples jobs (producción)",
    description=f"Consulta los logs de múltiples jobs por lista de IDs en paralelo. Modular, validado y seguro. Máximo {MAX_BATCH_IDS} IDs por request.",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"logs": {}}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        400: {"content": {"application/json": {"example": {"success": False, "error": {"message": "IDs debe ser lista de strings (máx 50)"}, "trace_id": "abc-123"}}}}
    },
)
@limiter.limit("10/minute")
@require_user
@endpoint_protected("/video/logs/batch", logger, envelope)
async def batch_logs(
    ids: List[str] = Body(..., embed=True, description=f"Lista de request_id (máx {MAX_BATCH_IDS})", example=["req_1", "req_2"]),
    use_onyx: bool = Query(False, description="Si es True, consulta Onyx; si es False, consulta local."),
    user=Depends(get_current_user),
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    request: Request = None
) -> EnvelopeResponse:
    """
    Batch logs endpoint: orquesta validación, helpers y respuesta.
    - Devuelve el trace_id en la respuesta para correlación.
    - Errores posibles: ids inválidos, error Onyx, error interno.
    """
    trace_id = x_trace_id or str(uuid.uuid4())
    assert isinstance(ids, list), "ids debe ser lista"
    if logger:
        logger.info({"endpoint": "/video/logs/batch", "event": "start", "trace_id": trace_id, "ids": ids})
    try:
        validate_batch_ids(ids)
        resp = await batch_service.batch_logs(ids, use_onyx, trace_id=trace_id)
        if logger:
            logger.info({"endpoint": "/video/logs/batch", "event": "success", "trace_id": trace_id, "ids": ids})
        return resp
    except Exception as e:
        if logger:
            logger.error({"endpoint": "/video/logs/batch", "event": "error", "trace_id": trace_id, "error": str(e)})
        return envelope(False, error={"message": str(e)}, trace_id=trace_id)

def get_trace_id(request: Request) -> str:
    """
    Obtiene el trace_id de los headers de la request o genera uno nuevo único.
    Se usa para correlación y tracing distribuido en logs y respuestas.
    """
    return request.headers.get("x-trace-id") or str(uuid.uuid4())

@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    
    """service_error_handler function."""
trace_id = get_trace_id(request)
    now = datetime.utcnow().isoformat()
    if 'logger' in globals() and logger:
        logger.error({"endpoint": str(request.url), "event": "service_error", "trace_id": trace_id, "error": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=EnvelopeResponse(success=False, error={"message": str(exc)}, data=None, trace_id=trace_id, timestamp=now).dict(exclude_unset=True)
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    
    """validation_error_handler function."""
trace_id = get_trace_id(request)
    now = datetime.utcnow().isoformat()
    if 'logger' in globals() and logger:
        logger.error({"endpoint": str(request.url), "event": "validation_error", "trace_id": trace_id, "error": exc.errors()})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=EnvelopeResponse(success=False, error={"message": "Validation error", "details": exc.errors()}, data=None, trace_id=trace_id, timestamp=now).dict(exclude_unset=True)
    )

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    
    """generic_error_handler function."""
trace_id = get_trace_id(request)
    now = datetime.utcnow().isoformat()
    if 'logger' in globals() and logger:
        logger.error({"endpoint": str(request.url), "event": "internal_error", "trace_id": trace_id, "error": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=EnvelopeResponse(success=False, error={"message": "Internal server error"}, data=None, trace_id=trace_id, timestamp=now).dict(exclude_unset=True)
    )

# --- Registrar el router en la app principal ---
app.include_router(video_router)

# --- Buenas prácticas: para máxima performance en producción, usar uvicorn con --workers N y --loop uvloop ---
# Ejemplo: uvicorn agents.backend.onyx.server.features.ai_video.fastapi_microservice:app --workers 4 --loop uvloop
# ... existing code ...
# (En los helpers batch ya se usa asyncio.gather con return_exceptions=True para máxima concurrencia)
# (En la serialización de modelos Pydantic, usar .dict(exclude_unset=True) si aplica en los servicios)
# ... existing code ... 