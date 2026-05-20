"""
BUL API - Frontend Ready (Enhanced)
====================================

API mejorada lista para consumo desde frontend TypeScript.
Incluye:
- Integración con sistema BUL real
- WebSocket para actualizaciones en tiempo real
- Rate limiting mejorado
- Mejor manejo de errores
- Validaciones robustas
- CORS configurado
"""

import asyncio
import logging
import sys
import argparse
import hashlib
import time
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bul_api.log')
    ]
)

logger = logging.getLogger(__name__)

try:
    from core.continuous_processor import get_global_processor, ContinuousProcessor
    from core.truthgpt_bulk_processor import get_global_truthgpt_processor, TruthGPTBulkProcessor
    BUL_AVAILABLE = True
except ImportError:
    BUL_AVAILABLE = False
    logger.warning("Sistema BUL no disponible, usando modo simulación")

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus no disponible, métricas deshabilitadas")

app = FastAPI(
    title="BUL API - Frontend Ready",
    description="""
    API completa para generación de documentos con IA, lista para consumo desde frontend TypeScript.
    
    ## Características
    
    * ✅ **Generación de Documentos**: Crea documentos personalizados usando IA
    * 🔄 **WebSocket**: Actualizaciones en tiempo real del progreso
    * 📊 **Métricas**: Endpoint Prometheus para monitoreo
    * 🚀 **Alto Rendimiento**: Optimizado para producción
    * 🔒 **Seguro**: Rate limiting y validaciones robustas
    
    ## SDKs Disponibles
    
    * **TypeScript**: `bul-api-client.ts`
    * **JavaScript**: `bul-api-client.js`
    * **Python**: `bul-api-client.py`
    
    ## Documentación
    
    * Swagger UI: `/api/docs`
    * ReDoc: `/api/redoc`
    * OpenAPI Schema: `/api/openapi.json`
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    tags_metadata=[
        {
            "name": "System",
            "description": "Endpoints del sistema (health, stats, root)"
        },
        {
            "name": "Documents",
            "description": "Generación y gestión de documentos"
        },
        {
            "name": "Tasks",
            "description": "Gestión y seguimiento de tareas"
        },
        {
            "name": "WebSocket",
            "description": "Conexiones WebSocket para actualizaciones en tiempo real"
        }
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

tasks: Dict[str, Dict[str, Any]] = {}
documents: Dict[str, Dict[str, Any]] = {}
start_time = datetime.now()
request_count = 0
cache_store = {}
websocket_connections: Set[WebSocket] = set()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

bul_processor: Optional[ContinuousProcessor] = None
truthgpt_processor: Optional[TruthGPTBulkProcessor] = None

if BUL_AVAILABLE:
    try:
        bul_processor = get_global_processor()
        truthgpt_processor = get_global_truthgpt_processor()
        logger.info("Sistema BUL integrado exitosamente")
    except Exception as e:
        logger.warning(f"No se pudo inicializar sistema BUL: {e}")
        BUL_AVAILABLE = False

class DocumentRequest(BaseModel):
    """Request model para generación de documentos."""
    query: str = Field(..., min_length=10, max_length=5000, description="Consulta de negocio para generación de documento")
    business_area: Optional[str] = Field(None, description="Área de negocio específica")
    document_type: Optional[str] = Field(None, description="Tipo de documento a generar")
    priority: int = Field(1, ge=1, le=5, description="Prioridad de procesamiento (1-5)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos adicionales")
    user_id: Optional[str] = Field(None, description="ID de usuario")
    session_id: Optional[str] = Field(None, description="ID de sesión")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('La consulta no puede estar vacía')
        return v.strip()

class DocumentResponse(BaseModel):
    """Response model para generación de documentos."""
    task_id: str
    status: str
    message: str
    estimated_time: Optional[int] = None
    queue_position: Optional[int] = None
    created_at: str

class TaskStatus(BaseModel):
    """Model para estado de tarea."""
    task_id: str
    status: str
    progress: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    """Model para health check."""
    status: str
    timestamp: str
    uptime: str
    active_tasks: int
    total_requests: int
    version: str

class StatsResponse(BaseModel):
    """Model para estadísticas."""
    total_requests: int
    active_tasks: int
    completed_tasks: int
    success_rate: float
    average_processing_time: float
    uptime: str

class TaskListResponse(BaseModel):
    """Model para lista de tareas."""
    tasks: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
    has_more: bool

async def broadcast_task_update(task_id: str, update: Dict[str, Any]):
    """Envía actualización de tarea a todos los clientes WebSocket conectados."""
    if not websocket_connections:
        return
    
    message = {
        "type": "task_update",
        "task_id": task_id,
        "data": update,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = set()
    for connection in websocket_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Error enviando actualización WebSocket: {e}")
            disconnected.add(connection)
    
    websocket_connections.difference_update(disconnected)

async def process_document_background(task_id: str, request: DocumentRequest):
    """Procesa un documento en segundo plano usando el sistema BUL real si está disponible."""
    try:
        task = tasks[task_id]
        task["status"] = "processing"
        task["progress"] = 10
        task["updated_at"] = datetime.now()
        await broadcast_task_update(task_id, {"status": "processing", "progress": 10})
        
        document_content = None
        
        if BUL_AVAILABLE and truthgpt_processor:
            try:
                logger.info(f"Usando sistema BUL real para tarea {task_id}")
                
                from core.truthgpt_bulk_processor import DocumentGenerationTask
                
                bul_task = DocumentGenerationTask(
                    query=request.query,
                    business_area=request.business_area or "general",
                    document_type=request.document_type or "document",
                    priority=request.priority,
                    metadata=request.metadata or {},
                    user_id=request.user_id,
                    session_id=request.session_id or task_id
                )
                
                task["progress"] = 20
                await broadcast_task_update(task_id, {"status": "processing", "progress": 20})
                
                document_content = await truthgpt_processor._generate_document_content(bul_task)
                
                if document_content:
                    task["progress"] = 90
                    await broadcast_task_update(task_id, {"status": "processing", "progress": 90})
                
            except Exception as e:
                logger.error(f"Error usando sistema BUL: {e}")
        
        if not document_content:
            logger.info(f"Usando modo simulación para tarea {task_id}")
            await asyncio.sleep(1)
            task["progress"] = 50
            await broadcast_task_update(task_id, {"status": "processing", "progress": 50})
            
            await asyncio.sleep(1)
            task["progress"] = 80
            await broadcast_task_update(task_id, {"status": "processing", "progress": 80})
            
            document_content = f"""# Documento Generado

**Consulta:** {request.query}

**Área de Negocio:** {request.business_area or "General"}

**Tipo de Documento:** {request.document_type or "Documento estándar"}

## Contenido Generado

Este es un documento generado automáticamente basado en la consulta proporcionada.
El sistema procesa la información y genera contenido relevante para el negocio.

**Prioridad:** {request.priority}

**Generado el:** {datetime.now().isoformat()}

---

*Nota: Este documento fue generado usando el modo simulación. Para usar el sistema BUL completo, asegúrate de que los componentes estén correctamente configurados.*
"""
        
        await asyncio.sleep(0.5)
        task["progress"] = 100
        
        result = {
            "content": document_content,
            "format": "markdown",
            "word_count": len(document_content.split()),
            "generated_at": datetime.now().isoformat(),
            "using_bul_system": BUL_AVAILABLE and truthgpt_processor is not None
        }
        
        task["status"] = "completed"
        task["progress"] = 100
        task["result"] = result
        task["processing_time"] = (datetime.now() - task["created_at"]).total_seconds()
        task["updated_at"] = datetime.now()
        
        documents[task_id] = {
            "task_id": task_id,
            "content": document_content,
            "metadata": request.dict(),
            "created_at": datetime.now().isoformat()
        }
        
        await broadcast_task_update(task_id, {
            "status": "completed",
            "progress": 100,
            "result": result
        })
        
        logger.info(f"Documento {task_id} procesado exitosamente")
        
    except Exception as e:
        logger.error(f"Error procesando documento {task_id}: {e}", exc_info=True)
        if task_id in tasks:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)
            tasks[task_id]["updated_at"] = datetime.now()
            await broadcast_task_update(task_id, {
                "status": "failed",
                "error": str(e)
            })

@app.get("/", tags=["System"])
async def root():
    """Endpoint raíz con información del sistema."""
    return {
        "message": "BUL API - Frontend Ready",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "docs": "/api/docs",
        "health": "/api/health"
    }

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    uptime = datetime.now() - start_time
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime=str(uptime),
        active_tasks=len([t for t in tasks.values() if t["status"] in ["queued", "processing"]]),
        total_requests=request_count,
        version="1.0.0"
    )

@app.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Obtener estadísticas del sistema."""
    completed_tasks = len([t for t in tasks.values() if t["status"] == "completed"])
    total_tasks = len(tasks)
    success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
    
    processing_times = [
        t.get("processing_time", 0)
        for t in tasks.values()
        if t.get("processing_time")
    ]
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
    
    uptime = datetime.now() - start_time
    
    return StatsResponse(
        total_requests=request_count,
        active_tasks=len([t for t in tasks.values() if t["status"] in ["queued", "processing"]]),
        completed_tasks=completed_tasks,
        success_rate=success_rate,
        average_processing_time=avg_processing_time,
        uptime=str(uptime)
    )

@app.post("/api/documents/generate", response_model=DocumentResponse, tags=["Documents"])
@limiter.limit("10/minute")
async def generate_document(
    request: DocumentRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    Generar un nuevo documento.
    
    Este endpoint inicia el proceso de generación de un documento basado en la consulta proporcionada.
    Retorna un task_id que puede usarse para consultar el estado del documento.
    
    **Rate Limit:** 10 solicitudes por minuto por IP
    """
    global request_count
    request_count += 1
    
    try:
        if len(request.query.strip()) < 10:
            raise HTTPException(status_code=400, detail="La consulta debe tener al menos 10 caracteres")
        
        if len(request.query) > 5000:
            raise HTTPException(status_code=400, detail="La consulta no puede exceder 5000 caracteres")
        
        cache_key = f"doc:{hashlib.md5(request.query.encode()).hexdigest()}"
        if cache_key in cache_store:
            cached_task_id = cache_store[cache_key]
            if cached_task_id in tasks and tasks[cached_task_id]["status"] == "completed":
                return DocumentResponse(
                    task_id=cached_task_id,
                    status="completed",
                    message="Documento recuperado del caché",
                    estimated_time=0,
                    created_at=tasks[cached_task_id]["created_at"].isoformat()
                )
        
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "request": request.dict(),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "result": None,
            "error": None,
            "processing_time": None,
            "user_id": request.user_id,
            "session_id": request.session_id
        }
        
        cache_store[cache_key] = task_id
        
        background_tasks.add_task(process_document_background, task_id, request)
        
        return DocumentResponse(
            task_id=task_id,
            status="queued",
            message="Generación de documento iniciada",
            estimated_time=60,
            queue_position=len([t for t in tasks.values() if t["status"] == "queued"]),
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error iniciando generación de documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}/status", response_model=TaskStatus, tags=["Tasks"])
async def get_task_status(task_id: str):
    """Obtener el estado de una tarea."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    
    task = tasks[task_id]
    
    processing_time = None
    if task["status"] == "completed" and task.get("processing_time"):
        processing_time = task["processing_time"]
    elif task["status"] in ["processing", "completed"]:
        processing_time = (datetime.now() - task["created_at"]).total_seconds()
    
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        result=task["result"],
        error=task["error"],
        created_at=task["created_at"].isoformat(),
        updated_at=task["updated_at"].isoformat(),
        processing_time=processing_time
    )

@app.get("/api/tasks/{task_id}/document", tags=["Documents"])
async def get_task_document(task_id: str):
    """Obtener el documento generado de una tarea."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Tarea aún no completada. Estado: {task['status']}")
    
    if not task.get("result"):
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    
    return {
        "task_id": task_id,
        "document": task["result"],
        "metadata": task["request"],
        "created_at": task["created_at"].isoformat(),
        "completed_at": task["updated_at"].isoformat()
    }

@app.get("/api/tasks", response_model=TaskListResponse, tags=["Tasks"])
async def list_tasks(
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Listar tareas con filtrado y paginación."""
    filtered_tasks = []
    
    for task_id, task in tasks.items():
        if status and task["status"] != status:
            continue
        if user_id and task.get("user_id") != user_id:
            continue
        
        query_preview = task["request"]["query"]
        if len(query_preview) > 100:
            query_preview = query_preview[:100] + "..."
        
        filtered_tasks.append({
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "created_at": task["created_at"].isoformat(),
            "updated_at": task["updated_at"].isoformat(),
            "user_id": task.get("user_id"),
            "query_preview": query_preview
        })
    
    filtered_tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    total = len(filtered_tasks)
    paginated_tasks = filtered_tasks[offset:offset + limit]
    
    return TaskListResponse(
        tasks=paginated_tasks,
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + limit < total
    )

@app.delete("/api/tasks/{task_id}", tags=["Tasks"])
async def delete_task(task_id: str):
    """Eliminar una tarea."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    
    del tasks[task_id]
    
    if task_id in documents:
        del documents[task_id]
    
    return {"message": "Tarea eliminada exitosamente", "task_id": task_id}

@app.post("/api/tasks/{task_id}/cancel", tags=["Tasks"])
async def cancel_task(task_id: str):
    """Cancelar una tarea en ejecución."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    
    task = tasks[task_id]
    if task["status"] not in ["queued", "processing"]:
        raise HTTPException(status_code=400, detail="La tarea no puede ser cancelada")
    
    task["status"] = "cancelled"
    task["updated_at"] = datetime.now()
    
    return {"message": "Tarea cancelada exitosamente", "task_id": task_id}

@app.get("/api/documents", tags=["Documents"])
async def list_documents(
    limit: int = 50,
    offset: int = 0
):
    """Listar documentos generados."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="El límite debe estar entre 1 y 100")
    if offset < 0:
        raise HTTPException(status_code=400, detail="El offset debe ser mayor o igual a 0")
    
    document_list = []
    
    for task_id, doc in documents.items():
        if task_id not in tasks:
            continue
        
        task = tasks[task_id]
        if task["status"] != "completed":
            continue
        
        document_list.append({
            "task_id": task_id,
            "created_at": doc["created_at"],
            "query_preview": task["request"]["query"][:100] + "..." if len(task["request"]["query"]) > 100 else task["request"]["query"],
            "business_area": task["request"].get("business_area"),
            "document_type": task["request"].get("document_type")
        })
    
    document_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    total = len(document_list)
    paginated_docs = document_list[offset:offset + limit]
    
    return {
        "documents": paginated_docs,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total
    }

@app.websocket("/api/ws/{task_id}")
async def websocket_task_updates(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint para recibir actualizaciones en tiempo real de una tarea.
    
    Conéctate con: ws://localhost:8000/api/ws/{task_id}
    """
    await websocket.accept()
    websocket_connections.add(websocket)
    
    try:
        if task_id in tasks:
            task = tasks[task_id]
            await websocket.send_json({
                "type": "initial_state",
                "task_id": task_id,
                "data": {
                    "status": task["status"],
                    "progress": task["progress"],
                    "result": task.get("result"),
                    "error": task.get("error")
                },
                "timestamp": datetime.now().isoformat()
            })
        
        while True:
            try:
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error en WebSocket: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        pass
    finally:
        websocket_connections.discard(websocket)

@app.websocket("/api/ws")
async def websocket_all_updates(websocket: WebSocket):
    """
    WebSocket endpoint para recibir actualizaciones de todas las tareas.
    
    Conéctate con: ws://localhost:8000/api/ws
    """
    await websocket.accept()
    websocket_connections.add(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Conectado a actualizaciones en tiempo real",
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            try:
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error en WebSocket: {e}")
    
    except WebSocketDisconnect:
        pass
    finally:
        websocket_connections.discard(websocket)

if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter('bul_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
    REQUEST_DURATION = Histogram('bul_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
    ACTIVE_TASKS_METRIC = Gauge('bul_active_tasks', 'Active tasks')
    DOCUMENT_GENERATION_TIME = Histogram('bul_document_generation_seconds', 'Document generation time')
    
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        """Middleware para métricas Prometheus."""
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
    
    @app.get("/metrics")
    async def metrics():
        """Endpoint de métricas Prometheus."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

@app.on_event("startup")
async def startup_event():
    """Evento de inicio."""
    logger.info("BUL API iniciada - Lista para frontend TypeScript")
    logger.info(f"Documentación disponible en: http://localhost:8000/api/docs")
    
    if PROMETHEUS_AVAILABLE:
        logger.info(f"Métricas Prometheus disponibles en: http://localhost:8000/metrics")
    
    if BUL_AVAILABLE:
        try:
            if bul_processor:
                await bul_processor.start()
                logger.info("Procesador BUL iniciado")
            if truthgpt_processor:
                await truthgpt_processor.start_continuous_processing()
                logger.info("Procesador TruthGPT iniciado")
        except Exception as e:
            logger.warning(f"No se pudieron iniciar procesadores BUL: {e}")
    
    logger.info(f"Sistema BUL disponible: {BUL_AVAILABLE}")
    logger.info(f"Prometheus disponible: {PROMETHEUS_AVAILABLE}")

@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre."""
    for websocket in list(websocket_connections):
        try:
            await websocket.close()
        except:
            pass
    
    if BUL_AVAILABLE:
        try:
            if bul_processor:
                bul_processor.stop()
            if truthgpt_processor:
                truthgpt_processor.stop_processing()
        except:
            pass
    
    logger.info("BUL API cerrada")

def main():
    """Función principal para ejecutar el servidor."""
    parser = argparse.ArgumentParser(description="BUL API - Frontend Ready")
    parser.add_argument("--host", default="0.0.0.0", help="Host para el servidor")
    parser.add_argument("--port", type=int, default=8000, help="Puerto para el servidor")
    parser.add_argument("--reload", action="store_true", help="Recargar automáticamente en cambios")
    
    args = parser.parse_args()
    
    logger.info(f"Iniciando BUL API en http://{args.host}:{args.port}")
    logger.info(f"Documentación: http://{args.host}:{args.port}/api/docs")
    
    uvicorn.run(
        "api_frontend_ready:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()

