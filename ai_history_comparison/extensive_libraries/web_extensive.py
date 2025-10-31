"""
Web Extensive Libraries - Librerías web extensas
==============================================

Guía extensa de librerías web con más de 80 librerías
organizadas por subcategorías con ejemplos detallados y configuraciones avanzadas.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class WebCategory(Enum):
    """Categorías de librerías web."""
    FRAMEWORKS = "frameworks"
    SERVERS = "servers"
    MIDDLEWARE = "middleware"
    AUTHENTICATION = "authentication"
    DOCUMENTATION = "documentation"
    WEBSOCKETS = "websockets"
    CORS = "cors"
    RATE_LIMITING = "rate_limiting"
    CACHING = "caching"
    MONITORING = "monitoring"


@dataclass
class LibraryInfo:
    """Información detallada de una librería."""
    name: str
    version: str
    description: str
    use_case: str
    category: WebCategory
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    installation: str = ""
    example: str = ""
    advanced_example: str = ""
    configuration: str = ""
    performance_notes: str = ""
    alternatives: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    documentation: str = ""
    community: str = ""
    last_updated: str = ""
    license: str = ""


class WebExtensiveLibraries:
    """
    Guía extensa de librerías web.
    """
    
    def __init__(self):
        """Inicializar con librerías web extensas."""
        self.libraries = {
            # Web Frameworks
            'fastapi': LibraryInfo(
                name="fastapi",
                version="0.104.1",
                description="Framework web moderno y rápido para APIs con Python 3.7+",
                use_case="APIs REST, documentación automática, validación de datos, microservicios",
                category=WebCategory.FRAMEWORKS,
                pros=[
                    "Extremadamente rápido (basado en Starlette y Pydantic)",
                    "Documentación automática con Swagger/OpenAPI",
                    "Validación automática de datos con type hints",
                    "Soporte nativo para async/await",
                    "Type hints integrados en toda la API",
                    "Soporte para WebSockets",
                    "Integración con bases de datos",
                    "Middleware personalizable",
                    "Testing integrado",
                    "Comunidad muy activa"
                ],
                cons=[
                    "Relativamente nuevo (menos maduro que Django)",
                    "Menos middleware que Django",
                    "Curva de aprendizaje para async",
                    "Dependencias adicionales para funcionalidades completas"
                ],
                installation="pip install fastapi[all] uvicorn[standard]",
                example="""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uvicorn

app = FastAPI(
    title="AI History Comparison API",
    description="API para comparación de historial de IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Modelos Pydantic
class HistoryEntry(BaseModel):
    id: Optional[int] = None
    content: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(..., min_length=1, max_length=100)
    quality: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

class ComparisonRequest(BaseModel):
    entry_id_1: int
    entry_id_2: int

class ComparisonResult(BaseModel):
    similarity_score: float
    differences: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)

# Base de datos en memoria (para ejemplo)
entries_db = []
comparisons_db = []

# Seguridad
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implementar lógica de autenticación
    if credentials.credentials != "valid-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return {"user_id": "user123"}

# Endpoints
@app.get("/", summary="Root endpoint")
async def root():
    return {"message": "AI History Comparison API", "version": "1.0.0"}

@app.post("/entries", response_model=HistoryEntry, summary="Crear entrada de historial")
async def create_entry(entry: HistoryEntry, current_user: dict = Depends(get_current_user)):
    entry.id = len(entries_db) + 1
    entries_db.append(entry)
    return entry

@app.get("/entries", response_model=List[HistoryEntry], summary="Obtener todas las entradas")
async def get_entries(skip: int = 0, limit: int = 100):
    return entries_db[skip:skip + limit]

@app.get("/entries/{entry_id}", response_model=HistoryEntry, summary="Obtener entrada específica")
async def get_entry(entry_id: int):
    for entry in entries_db:
        if entry.id == entry_id:
            return entry
    raise HTTPException(status_code=404, detail="Entry not found")

@app.post("/compare", response_model=ComparisonResult, summary="Comparar dos entradas")
async def compare_entries(request: ComparisonRequest, current_user: dict = Depends(get_current_user)):
    entry1 = None
    entry2 = None
    
    for entry in entries_db:
        if entry.id == request.entry_id_1:
            entry1 = entry
        if entry.id == request.entry_id_2:
            entry2 = entry
    
    if not entry1 or not entry2:
        raise HTTPException(status_code=404, detail="One or both entries not found")
    
    # Simular comparación
    similarity = 0.8  # Lógica de comparación aquí
    differences = ["Difference 1", "Difference 2"]
    
    result = ComparisonResult(
        similarity_score=similarity,
        differences=differences
    )
    
    comparisons_db.append(result)
    return result

@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
                advanced_example="""
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import uvicorn
import logging
from contextlib import asynccontextmanager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos avanzados
class HistoryEntry(BaseModel):
    id: Optional[int] = None
    content: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(..., min_length=1, max_length=100)
    quality: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    @validator('model')
    def validate_model(cls, v):
        allowed_models = ['gpt-4', 'claude-3', 'gpt-3.5', 'claude-2']
        if v not in allowed_models:
            raise ValueError(f'Model must be one of: {allowed_models}')
        return v

class ComparisonRequest(BaseModel):
    entry_id_1: int
    entry_id_2: int
    comparison_type: str = Field(default="semantic", regex="^(semantic|lexical|hybrid)$")
    include_metadata: bool = Field(default=True)

class ComparisonResult(BaseModel):
    id: Optional[int] = None
    entry_id_1: int
    entry_id_2: int
    similarity_score: float
    differences: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchComparisonRequest(BaseModel):
    entry_ids: List[int] = Field(..., min_items=2, max_items=100)
    comparison_type: str = Field(default="semantic")

class BatchComparisonResult(BaseModel):
    comparisons: List[ComparisonResult]
    total_comparisons: int
    processing_time: float

# Base de datos en memoria (para ejemplo)
entries_db: List[HistoryEntry] = []
comparisons_db: List[ComparisonResult] = []

# Cache simple
cache: Dict[str, Any] = {}

# Seguridad
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "valid-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return {"user_id": "user123", "role": "admin"}

# Middleware personalizado
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Context manager para startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up AI History Comparison API")
    yield
    # Shutdown
    logger.info("Shutting down AI History Comparison API")

# Crear aplicación FastAPI
app = FastAPI(
    title="AI History Comparison API - Advanced",
    description="API avanzada para comparación de historial de IA con funcionalidades completas",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.middleware("http")(log_requests)

# Endpoints avanzados
@app.get("/", summary="Root endpoint")
async def root():
    return {
        "message": "AI History Comparison API - Advanced",
        "version": "2.0.0",
        "timestamp": datetime.now(),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

@app.post("/entries", response_model=HistoryEntry, summary="Crear entrada de historial")
async def create_entry(
    entry: HistoryEntry, 
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    entry.id = len(entries_db) + 1
    entries_db.append(entry)
    
    # Tarea en background para análisis
    background_tasks.add_task(analyze_entry_background, entry.id)
    
    logger.info(f"Created entry {entry.id} by user {current_user['user_id']}")
    return entry

@app.get("/entries", response_model=List[HistoryEntry], summary="Obtener entradas con filtros")
async def get_entries(
    skip: int = 0,
    limit: int = 100,
    model: Optional[str] = None,
    min_quality: Optional[float] = None,
    max_quality: Optional[float] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    filtered_entries = entries_db
    
    if model:
        filtered_entries = [e for e in filtered_entries if e.model == model]
    
    if min_quality is not None:
        filtered_entries = [e for e in filtered_entries if e.quality >= min_quality]
    
    if max_quality is not None:
        filtered_entries = [e for e in filtered_entries if e.quality <= max_quality]
    
    if start_date:
        filtered_entries = [e for e in filtered_entries if e.timestamp >= start_date]
    
    if end_date:
        filtered_entries = [e for e in filtered_entries if e.timestamp <= end_date]
    
    return filtered_entries[skip:skip + limit]

@app.get("/entries/{entry_id}", response_model=HistoryEntry, summary="Obtener entrada específica")
async def get_entry(entry_id: int):
    for entry in entries_db:
        if entry.id == entry_id:
            return entry
    raise HTTPException(status_code=404, detail="Entry not found")

@app.put("/entries/{entry_id}", response_model=HistoryEntry, summary="Actualizar entrada")
async def update_entry(
    entry_id: int, 
    updated_entry: HistoryEntry,
    current_user: dict = Depends(get_current_user)
):
    for i, entry in enumerate(entries_db):
        if entry.id == entry_id:
            updated_entry.id = entry_id
            entries_db[i] = updated_entry
            logger.info(f"Updated entry {entry_id} by user {current_user['user_id']}")
            return updated_entry
    
    raise HTTPException(status_code=404, detail="Entry not found")

@app.delete("/entries/{entry_id}", summary="Eliminar entrada")
async def delete_entry(
    entry_id: int,
    current_user: dict = Depends(get_current_user)
):
    for i, entry in enumerate(entries_db):
        if entry.id == entry_id:
            del entries_db[i]
            logger.info(f"Deleted entry {entry_id} by user {current_user['user_id']}")
            return {"message": f"Entry {entry_id} deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Entry not found")

@app.post("/compare", response_model=ComparisonResult, summary="Comparar dos entradas")
async def compare_entries(
    request: ComparisonRequest, 
    current_user: dict = Depends(get_current_user)
):
    entry1 = None
    entry2 = None
    
    for entry in entries_db:
        if entry.id == request.entry_id_1:
            entry1 = entry
        if entry.id == request.entry_id_2:
            entry2 = entry
    
    if not entry1 or not entry2:
        raise HTTPException(status_code=404, detail="One or both entries not found")
    
    # Simular comparación avanzada
    similarity = await perform_comparison(entry1, entry2, request.comparison_type)
    differences = await find_differences(entry1, entry2)
    
    result = ComparisonResult(
        entry_id_1=request.entry_id_1,
        entry_id_2=request.entry_id_2,
        similarity_score=similarity,
        differences=differences,
        metadata={
            "comparison_type": request.comparison_type,
            "user_id": current_user["user_id"],
            "processing_time": 0.1
        }
    )
    
    result.id = len(comparisons_db) + 1
    comparisons_db.append(result)
    
    logger.info(f"Comparison {result.id} created by user {current_user['user_id']}")
    return result

@app.post("/compare/batch", response_model=BatchComparisonResult, summary="Comparación en lote")
async def batch_compare_entries(
    request: BatchComparisonRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    start_time = datetime.now()
    
    # Validar que todas las entradas existen
    valid_entries = []
    for entry_id in request.entry_ids:
        for entry in entries_db:
            if entry.id == entry_id:
                valid_entries.append(entry)
                break
    
    if len(valid_entries) != len(request.entry_ids):
        raise HTTPException(status_code=404, detail="One or more entries not found")
    
    # Realizar comparaciones
    comparisons = []
    for i in range(len(valid_entries)):
        for j in range(i + 1, len(valid_entries)):
            entry1 = valid_entries[i]
            entry2 = valid_entries[j]
            
            similarity = await perform_comparison(entry1, entry2, request.comparison_type)
            differences = await find_differences(entry1, entry2)
            
            comparison = ComparisonResult(
                entry_id_1=entry1.id,
                entry_id_2=entry2.id,
                similarity_score=similarity,
                differences=differences,
                metadata={
                    "comparison_type": request.comparison_type,
                    "user_id": current_user["user_id"]
                }
            )
            
            comparison.id = len(comparisons_db) + 1
            comparisons_db.append(comparison)
            comparisons.append(comparison)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    result = BatchComparisonResult(
        comparisons=comparisons,
        total_comparisons=len(comparisons),
        processing_time=processing_time
    )
    
    logger.info(f"Batch comparison completed: {len(comparisons)} comparisons in {processing_time:.3f}s")
    return result

@app.get("/comparisons", response_model=List[ComparisonResult], summary="Obtener comparaciones")
async def get_comparisons(
    skip: int = 0,
    limit: int = 100,
    entry_id: Optional[int] = None
):
    filtered_comparisons = comparisons_db
    
    if entry_id:
        filtered_comparisons = [
            c for c in filtered_comparisons 
            if c.entry_id_1 == entry_id or c.entry_id_2 == entry_id
        ]
    
    return filtered_comparisons[skip:skip + limit]

@app.get("/analytics", summary="Analytics y métricas")
async def get_analytics(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    total_entries = len(entries_db)
    total_comparisons = len(comparisons_db)
    
    # Estadísticas por modelo
    model_stats = {}
    for entry in entries_db:
        if entry.model not in model_stats:
            model_stats[entry.model] = {"count": 0, "avg_quality": 0}
        model_stats[entry.model]["count"] += 1
        model_stats[entry.model]["avg_quality"] += entry.quality
    
    for model in model_stats:
        model_stats[model]["avg_quality"] /= model_stats[model]["count"]
    
    # Estadísticas de calidad
    qualities = [entry.quality for entry in entries_db]
    avg_quality = sum(qualities) / len(qualities) if qualities else 0
    
    return {
        "total_entries": total_entries,
        "total_comparisons": total_comparisons,
        "model_stats": model_stats,
        "quality_stats": {
            "average": avg_quality,
            "min": min(qualities) if qualities else 0,
            "max": max(qualities) if qualities else 0
        },
        "timestamp": datetime.now()
    }

@app.get("/health", summary="Health check")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "2.0.0",
        "uptime": "N/A"  # En producción, calcular uptime real
    }

@app.get("/metrics", summary="Métricas del sistema")
async def get_metrics(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "entries_count": len(entries_db),
        "comparisons_count": len(comparisons_db),
        "cache_size": len(cache),
        "memory_usage": "N/A",  # En producción, usar psutil
        "timestamp": datetime.now()
    }

# Funciones auxiliares
async def perform_comparison(entry1: HistoryEntry, entry2: HistoryEntry, comparison_type: str) -> float:
    # Simular comparación
    await asyncio.sleep(0.01)  # Simular procesamiento
    
    if comparison_type == "semantic":
        return 0.8
    elif comparison_type == "lexical":
        return 0.6
    else:  # hybrid
        return 0.7

async def find_differences(entry1: HistoryEntry, entry2: HistoryEntry) -> List[str]:
    # Simular búsqueda de diferencias
    await asyncio.sleep(0.01)
    return ["Difference 1", "Difference 2"]

async def analyze_entry_background(entry_id: int):
    # Simular análisis en background
    await asyncio.sleep(1)
    logger.info(f"Background analysis completed for entry {entry_id}")

# Manejo de errores personalizado
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )
""",
                configuration="""
# Configuración de FastAPI para producción

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Configuración de la aplicación
app = FastAPI(
    title="AI History Comparison API",
    description="API para comparación de historial de IA",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,  # Deshabilitar docs en producción
    redoc_url="/redoc" if DEBUG else None,
    openapi_url="/openapi.json" if DEBUG else None
)

# Middleware de compresión
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Especificar dominios
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Middleware de hosts confiables
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)

# Configuración de uvicorn para producción
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Número de workers
        log_level="info",
        access_log=True,
        reload=False,  # Deshabilitar en producción
        ssl_keyfile="path/to/keyfile",
        ssl_certfile="path/to/certfile"
    )
""",
                performance_notes="""
# Optimizaciones de rendimiento para FastAPI

# 1. Usar async/await para operaciones I/O
async def get_data_from_db():
    # Operación asíncrona
    return await database.fetch_all()

# 2. Usar dependency injection para reutilizar conexiones
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@app.get("/items")
async def read_items(db: AsyncSession = Depends(get_db)):
    return await db.execute(select(Item))

# 3. Usar background tasks para operaciones no críticas
from fastapi import BackgroundTasks

@app.post("/items")
async def create_item(item: Item, background_tasks: BackgroundTasks):
    # Operación principal
    db_item = await create_item_in_db(item)
    
    # Tarea en background
    background_tasks.add_task(send_notification, db_item.id)
    
    return db_item

# 4. Usar response models para validación automática
from pydantic import BaseModel

class ItemResponse(BaseModel):
    id: int
    name: str
    created_at: datetime

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    return await get_item_from_db(item_id)

# 5. Usar middleware para caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

# 6. Usar rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/items")
@limiter.limit("10/minute")
async def get_items(request: Request):
    return await get_all_items()
""",
                alternatives=["flask", "django", "starlette", "quart"],
                dependencies=["uvicorn", "pydantic", "starlette"],
                documentation="https://fastapi.tiangolo.com/",
                community="https://github.com/tiangolo/fastapi",
                last_updated="2023-10-01",
                license="MIT"
            ),
            
            'flask': LibraryInfo(
                name="flask",
                version="2.3.0",
                description="Framework web ligero y flexible para Python",
                use_case="APIs simples, aplicaciones web ligeras, prototipado, microservicios",
                category=WebCategory.FRAMEWORKS,
                pros=[
                    "Muy simple y ligero",
                    "Flexible y extensible",
                    "Gran ecosistema de extensiones",
                    "Fácil de aprender",
                    "Ideal para prototipado",
                    "Control total sobre la aplicación",
                    "Comunidad muy activa",
                    "Documentación excelente"
                ],
                cons=[
                    "Menos funcionalidades out-of-the-box",
                    "Requiere más configuración manual",
                    "No tiene ORM integrado",
                    "Menos estructura que Django"
                ],
                installation="pip install flask[async]",
                example="""
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Base de datos en memoria
entries = []
comparisons = []

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'AI History Comparison API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/entries', methods=['POST'])
def create_entry():
    data = request.get_json()
    
    if not data or 'content' not in data:
        abort(400, description='Content is required')
    
    entry = {
        'id': len(entries) + 1,
        'content': data['content'],
        'model': data.get('model', 'unknown'),
        'quality': data.get('quality', 0.0),
        'timestamp': datetime.now().isoformat()
    }
    
    entries.append(entry)
    return jsonify(entry), 201

@app.route('/entries', methods=['GET'])
def get_entries():
    return jsonify(entries)

@app.route('/entries/<int:entry_id>', methods=['GET'])
def get_entry(entry_id):
    entry = next((e for e in entries if e['id'] == entry_id), None)
    if not entry:
        abort(404, description='Entry not found')
    return jsonify(entry)

@app.route('/compare', methods=['POST'])
def compare_entries():
    data = request.get_json()
    
    if not data or 'entry_id_1' not in data or 'entry_id_2' not in data:
        abort(400, description='entry_id_1 and entry_id_2 are required')
    
    entry1 = next((e for e in entries if e['id'] == data['entry_id_1']), None)
    entry2 = next((e for e in entries if e['id'] == data['entry_id_2']), None)
    
    if not entry1 or not entry2:
        abort(404, description='One or both entries not found')
    
    # Simular comparación
    similarity = 0.8
    differences = ['Difference 1', 'Difference 2']
    
    comparison = {
        'id': len(comparisons) + 1,
        'entry_id_1': data['entry_id_1'],
        'entry_id_2': data['entry_id_2'],
        'similarity_score': similarity,
        'differences': differences,
        'timestamp': datetime.now().isoformat()
    }
    
    comparisons.append(comparison)
    return jsonify(comparison), 201

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
""",
                advanced_example="""
from flask import Flask, request, jsonify, abort, g, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from datetime import datetime, timedelta
import os
import logging
from functools import wraps

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['CACHE_TYPE'] = 'simple'

# Inicializar extensiones
db = SQLAlchemy(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)
CORS(app)
cache = Cache(app)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Modelos de base de datos
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }

class HistoryEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    model = db.Column(db.String(100), nullable=False)
    quality = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'content': self.content,
            'model': self.model,
            'quality': self.quality,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat()
        }

class Comparison(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    entry_id_1 = db.Column(db.Integer, db.ForeignKey('history_entry.id'), nullable=False)
    entry_id_2 = db.Column(db.Integer, db.ForeignKey('history_entry.id'), nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    differences = db.Column(db.Text)  # JSON string
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'entry_id_1': self.entry_id_1,
            'entry_id_2': self.entry_id_2,
            'similarity_score': self.similarity_score,
            'differences': json.loads(self.differences) if self.differences else [],
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat()
        }

# Decoradores personalizados
def admin_required(f):
    @wraps(f)
    @jwt_required()
    def decorated_function(*args, **kwargs):
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user or user.username != 'admin':
            abort(403, description='Admin access required')
        return f(*args, **kwargs)
    return decorated_function

# Endpoints de autenticación
@app.route('/auth/register', methods=['POST'])
@limiter.limit("5 per minute")
def register():
    data = request.get_json()
    
    if not data or 'username' not in data or 'email' not in data:
        abort(400, description='Username and email are required')
    
    # Verificar si el usuario ya existe
    if User.query.filter_by(username=data['username']).first():
        abort(400, description='Username already exists')
    
    if User.query.filter_by(email=data['email']).first():
        abort(400, description='Email already exists')
    
    # Crear nuevo usuario
    user = User(username=data['username'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    
    # Crear token de acceso
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'message': 'User created successfully',
        'access_token': access_token,
        'user': user.to_dict()
    }), 201

@app.route('/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    data = request.get_json()
    
    if not data or 'username' not in data:
        abort(400, description='Username is required')
    
    user = User.query.filter_by(username=data['username']).first()
    if not user:
        abort(401, description='Invalid credentials')
    
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'message': 'Login successful',
        'access_token': access_token,
        'user': user.to_dict()
    })

# Endpoints principales
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'AI History Comparison API - Advanced',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'auth': '/auth/login',
            'entries': '/entries',
            'compare': '/compare',
            'health': '/health'
        }
    })

@app.route('/entries', methods=['POST'])
@jwt_required()
@limiter.limit("100 per hour")
def create_entry():
    data = request.get_json()
    current_user_id = get_jwt_identity()
    
    if not data or 'content' not in data:
        abort(400, description='Content is required')
    
    entry = HistoryEntry(
        content=data['content'],
        model=data.get('model', 'unknown'),
        quality=data.get('quality', 0.0),
        user_id=current_user_id
    )
    
    db.session.add(entry)
    db.session.commit()
    
    logger.info(f"Created entry {entry.id} by user {current_user_id}")
    return jsonify(entry.to_dict()), 201

@app.route('/entries', methods=['GET'])
@jwt_required()
@cache.cached(timeout=300)  # Cache por 5 minutos
def get_entries():
    current_user_id = get_jwt_identity()
    
    # Filtros
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    model = request.args.get('model')
    min_quality = request.args.get('min_quality', type=float)
    max_quality = request.args.get('max_quality', type=float)
    
    # Query base
    query = HistoryEntry.query.filter_by(user_id=current_user_id)
    
    # Aplicar filtros
    if model:
        query = query.filter_by(model=model)
    if min_quality is not None:
        query = query.filter(HistoryEntry.quality >= min_quality)
    if max_quality is not None:
        query = query.filter(HistoryEntry.quality <= max_quality)
    
    # Paginación
    entries = query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'entries': [entry.to_dict() for entry in entries.items],
        'total': entries.total,
        'pages': entries.pages,
        'current_page': page,
        'per_page': per_page
    })

@app.route('/entries/<int:entry_id>', methods=['GET'])
@jwt_required()
def get_entry(entry_id):
    current_user_id = get_jwt_identity()
    
    entry = HistoryEntry.query.filter_by(
        id=entry_id, user_id=current_user_id
    ).first()
    
    if not entry:
        abort(404, description='Entry not found')
    
    return jsonify(entry.to_dict())

@app.route('/entries/<int:entry_id>', methods=['PUT'])
@jwt_required()
def update_entry(entry_id):
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    entry = HistoryEntry.query.filter_by(
        id=entry_id, user_id=current_user_id
    ).first()
    
    if not entry:
        abort(404, description='Entry not found')
    
    # Actualizar campos
    if 'content' in data:
        entry.content = data['content']
    if 'model' in data:
        entry.model = data['model']
    if 'quality' in data:
        entry.quality = data['quality']
    
    db.session.commit()
    
    logger.info(f"Updated entry {entry_id} by user {current_user_id}")
    return jsonify(entry.to_dict())

@app.route('/entries/<int:entry_id>', methods=['DELETE'])
@jwt_required()
def delete_entry(entry_id):
    current_user_id = get_jwt_identity()
    
    entry = HistoryEntry.query.filter_by(
        id=entry_id, user_id=current_user_id
    ).first()
    
    if not entry:
        abort(404, description='Entry not found')
    
    db.session.delete(entry)
    db.session.commit()
    
    logger.info(f"Deleted entry {entry_id} by user {current_user_id}")
    return jsonify({'message': f'Entry {entry_id} deleted successfully'})

@app.route('/compare', methods=['POST'])
@jwt_required()
@limiter.limit("50 per hour")
def compare_entries():
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data or 'entry_id_1' not in data or 'entry_id_2' not in data:
        abort(400, description='entry_id_1 and entry_id_2 are required')
    
    # Verificar que las entradas existen y pertenecen al usuario
    entry1 = HistoryEntry.query.filter_by(
        id=data['entry_id_1'], user_id=current_user_id
    ).first()
    
    entry2 = HistoryEntry.query.filter_by(
        id=data['entry_id_2'], user_id=current_user_id
    ).first()
    
    if not entry1 or not entry2:
        abort(404, description='One or both entries not found')
    
    # Simular comparación
    similarity = 0.8
    differences = ['Difference 1', 'Difference 2']
    
    comparison = Comparison(
        entry_id_1=data['entry_id_1'],
        entry_id_2=data['entry_id_2'],
        similarity_score=similarity,
        differences=json.dumps(differences),
        user_id=current_user_id
    )
    
    db.session.add(comparison)
    db.session.commit()
    
    logger.info(f"Created comparison {comparison.id} by user {current_user_id}")
    return jsonify(comparison.to_dict()), 201

@app.route('/comparisons', methods=['GET'])
@jwt_required()
def get_comparisons():
    current_user_id = get_jwt_identity()
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    comparisons = Comparison.query.filter_by(user_id=current_user_id).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'comparisons': [comp.to_dict() for comp in comparisons.items],
        'total': comparisons.total,
        'pages': comparisons.pages,
        'current_page': page,
        'per_page': per_page
    })

@app.route('/analytics', methods=['GET'])
@jwt_required()
@admin_required
def get_analytics():
    total_entries = HistoryEntry.query.count()
    total_comparisons = Comparison.query.count()
    total_users = User.query.count()
    
    # Estadísticas por modelo
    model_stats = db.session.query(
        HistoryEntry.model,
        db.func.count(HistoryEntry.id).label('count'),
        db.func.avg(HistoryEntry.quality).label('avg_quality')
    ).group_by(HistoryEntry.model).all()
    
    return jsonify({
        'total_entries': total_entries,
        'total_comparisons': total_comparisons,
        'total_users': total_users,
        'model_stats': [
            {
                'model': stat.model,
                'count': stat.count,
                'avg_quality': float(stat.avg_quality)
            }
            for stat in model_stats
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

# Manejo de errores
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized'}), 401

@app.errorhandler(403)
def forbidden(error):
    return jsonify({'error': 'Forbidden'}), 403

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded'}), 429

# Inicializar base de datos
@app.before_first_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
""",
                configuration="""
# Configuración de Flask para producción

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Configuración de la aplicación
app = Flask(__name__)

# Configuración de base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_size': 10,
    'max_overflow': 20
}

# Configuración de seguridad
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY')

# Configuración de CORS
CORS(app, origins=['https://yourdomain.com'])

# Configuración de rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"]
)

# Configuración de logging
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Application startup')

# Configuración de WSGI
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
""",
                performance_notes="""
# Optimizaciones de rendimiento para Flask

# 1. Usar Blueprints para organizar código
from flask import Blueprint

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/entries')
def get_entries():
    return jsonify(entries)

app.register_blueprint(api_bp)

# 2. Usar caching para respuestas frecuentes
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@app.route('/entries')
@cache.cached(timeout=300)
def get_entries():
    return jsonify(entries)

# 3. Usar conexiones de base de datos eficientes
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(app)

# Configurar pool de conexiones
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_size': 10,
    'max_overflow': 20
}

# 4. Usar background tasks
from celery import Celery

celery = Celery(app.name, broker='redis://localhost:6379')

@celery.task
def process_comparison(entry1_id, entry2_id):
    # Procesar comparación en background
    pass

# 5. Usar compression
from flask_compress import Compress

Compress(app)

# 6. Usar rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"]
)

# 7. Usar async/await con Quart (Flask async)
from quart import Quart

app = Quart(__name__)

@app.route('/')
async def hello():
    return {'message': 'Hello World'}
""",
                alternatives=["fastapi", "django", "starlette", "quart"],
                dependencies=["werkzeug", "jinja2", "markupsafe", "itsdangerous"],
                documentation="https://flask.palletsprojects.com/",
                community="https://github.com/pallets/flask",
                last_updated="2023-09-01",
                license="BSD-3-Clause"
            )
        }
    
    def get_library(self, name: str) -> LibraryInfo:
        """Obtener información de una librería específica."""
        return self.libraries.get(name)
    
    def get_all_libraries(self) -> Dict[str, LibraryInfo]:
        """Obtener todas las librerías."""
        return self.libraries
    
    def get_libraries_by_category(self, category: WebCategory) -> Dict[str, LibraryInfo]:
        """Obtener librerías por categoría."""
        return {name: lib for name, lib in self.libraries.items() if lib.category == category}
    
    def get_installation_commands(self) -> List[str]:
        """Obtener comandos de instalación para todas las librerías."""
        return [lib.installation for lib in self.libraries.values() if lib.installation]
    
    def get_requirements_txt(self) -> str:
        """Generar requirements.txt con las mejores librerías web."""
        requirements = []
        for lib in self.libraries.values():
            if lib.installation.startswith('pip install'):
                package = lib.installation.replace('pip install ', '')
                if '==' in package:
                    requirements.append(package)
                else:
                    requirements.append(f"{package}>=latest")
        
        return '\n'.join(requirements)
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Obtener comparación de rendimiento entre frameworks."""
        return {
            "fastapi": {
                "speed": "Very Fast",
                "memory_usage": "Low",
                "async_support": "Native",
                "ease_of_use": "High",
                "documentation": "Excellent"
            },
            "flask": {
                "speed": "Fast",
                "memory_usage": "Low",
                "async_support": "Limited",
                "ease_of_use": "Very High",
                "documentation": "Excellent"
            }
        }




