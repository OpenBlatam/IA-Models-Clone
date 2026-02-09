"""
Servidor principal para Web Content Extractor AI
"""

from core.app_factory import create_app
from api.v1.routes import router
from config import settings

app = create_app()
app.include_router(router)


@app.get("/")
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "service": "Web Content Extractor AI",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "extract": "/api/v1/extract",
            "batch_extract": "/api/v1/extract/batch",
            "cache_stats": "/api/v1/extract/cache/stats",
            "clear_cache": "/api/v1/extract/cache",
            "metrics": "/api/v1/metrics/stats",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint con verificación de dependencias"""
    from infrastructure.openrouter.client import OpenRouterClient
    from config import settings
    
    checks = {
        "status": "healthy",
        "service": "web-content-extractor-ai",
        "version": "1.0.0"
    }
    
    # Verificar OpenRouter
    openrouter_ok = bool(settings.openrouter_api_key)
    checks["openrouter"] = {
        "configured": openrouter_ok,
        "status": "ok" if openrouter_ok else "missing_api_key"
    }
    
    return checks


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)

