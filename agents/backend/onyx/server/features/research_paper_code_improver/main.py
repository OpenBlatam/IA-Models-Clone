"""
Aplicación Principal - Research Paper Code Improver
====================================================
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import router
from api.dashboard_routes import router as dashboard_router

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Inicializar métricas globales
try:
    from core.metrics_collector import MetricsCollector
    metrics_collector = MetricsCollector()
    logger.info("Metrics Collector inicializado")
except Exception as e:
    logger.warning(f"No se pudo inicializar Metrics Collector: {e}")
    metrics_collector = None

# Crear aplicación FastAPI
app = FastAPI(
    title="Research Paper Code Improver",
    description="Sistema de IA que entrena modelos basados en papers y mejora código de GitHub",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(router)
app.include_router(dashboard_router)

# Middleware para métricas
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware para recolectar métricas"""
    import time
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        if metrics_collector:
            metrics_collector.record_request(
                endpoint=str(request.url.path),
                method=request.method,
                duration_ms=duration_ms,
                success=response.status_code < 400
            )
        
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        if metrics_collector:
            metrics_collector.record_error(
                error_type=type(e).__name__,
                error_message=str(e),
                endpoint=str(request.url.path)
            )
        
        raise


@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    logger.info("=" * 60)
    logger.info("Research Paper Code Improver v1.0.0")
    logger.info("=" * 60)
    logger.info("Servidor iniciado correctamente")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre de la aplicación"""
    logger.info("Cerrando servidor...")
    logger.info("Servidor detenido")


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": "Research Paper Code Improver",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/research-paper-code-improver/health",
        "status": "operational"
    }


if __name__ == "__main__":
    # Configuración del servidor
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8030))
    
    logger.info(f"Iniciando servidor en http://{host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

