"""
Configuración OpenAPI/Swagger para Robot Movement AI v2.0
Documentación automática de la API
"""

from typing import Optional
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

def custom_openapi(app: FastAPI) -> dict:
    """
    Generar esquema OpenAPI personalizado
    
    Args:
        app: Instancia de FastAPI
        
    Returns:
        Esquema OpenAPI personalizado
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Robot Movement AI API v2.0",
        version="2.0.0",
        description="""
        # Robot Movement AI API v2.0
        
        API completa para control y optimización de movimiento robótico mediante IA.
        
        ## Características
        
        - ✅ Control de robots mediante comandos naturales
        - ✅ Optimización de trayectorias con Reinforcement Learning
        - ✅ Procesamiento visual con CNNs
        - ✅ Feedback en tiempo real a 1000Hz
        - ✅ Arquitectura empresarial (Clean Architecture + DDD)
        
        ## Autenticación
        
        Actualmente la API no requiere autenticación, pero se recomienda implementar
        autenticación en producción.
        
        ## Rate Limiting
        
        La API implementa rate limiting:
        - 60 requests por minuto
        - 1000 requests por hora
        - 10000 requests por día
        
        ## Códigos de Estado
        
        - `200` - Éxito
        - `201` - Creado
        - `400` - Solicitud inválida
        - `401` - No autorizado
        - `403` - Prohibido
        - `404` - No encontrado
        - `429` - Demasiadas peticiones (Rate Limit)
        - `500` - Error interno del servidor
        - `503` - Servicio no disponible
        
        ## Ejemplos
        
        Ver la sección de ejemplos en cada endpoint para más detalles.
        """,
        routes=app.routes,
        tags=[
            {
                "name": "robots",
                "description": "Operaciones relacionadas con robots",
            },
            {
                "name": "movements",
                "description": "Operaciones relacionadas con movimientos",
            },
            {
                "name": "health",
                "description": "Health checks y métricas del sistema",
            },
            {
                "name": "chat",
                "description": "Control mediante chat natural",
            },
        ],
        servers=[
            {
                "url": "http://localhost:8010",
                "description": "Servidor de desarrollo local"
            },
            {
                "url": "https://api.robot-movement-ai.com",
                "description": "Servidor de producción"
            },
        ],
    )
    
    # Agregar información adicional
    openapi_schema["info"]["contact"] = {
        "name": "Robot Movement AI Support",
        "email": "support@robot-movement-ai.com",
        "url": "https://robot-movement-ai.com/support"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "Proprietary",
        "url": "https://robot-movement-ai.com/license"
    }
    
    # Agregar esquemas de seguridad
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token authentication (opcional)"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key authentication (opcional)"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def setup_openapi(app: FastAPI):
    """
    Configurar OpenAPI personalizado en la app
    
    Args:
        app: Instancia de FastAPI
    """
    app.openapi = lambda: custom_openapi(app)




