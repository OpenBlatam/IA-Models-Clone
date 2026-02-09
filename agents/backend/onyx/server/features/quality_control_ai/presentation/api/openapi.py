"""
OpenAPI Configuration

Custom OpenAPI configuration for better API documentation.
"""

from fastapi.openapi.utils import get_openapi
from ...config.app_settings import get_settings


def custom_openapi():
    """
    Generate custom OpenAPI schema.
    
    Returns:
        OpenAPI schema dictionary
    """
    settings = get_settings()
    
    if hasattr(custom_openapi, "schema"):
        return custom_openapi.schema
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        ## Quality Control AI API
        
        Sistema completo de control de calidad con detección de defectos por cámara.
        
        ### Características
        
        - Inspección de imágenes en múltiples formatos
        - Detección de defectos con ML
        - Detección de anomalías con ML
        - Clasificación automática de defectos
        - Cálculo de calidad
        - Inspección en batch
        - Streaming en tiempo real (WebSocket)
        - Generación de reportes
        
        ### Formatos de Imagen Soportados
        
        - `numpy` - NumPy array
        - `bytes` - Bytes raw
        - `file_path` - Ruta de archivo
        - `base64` - Base64 encoded string
        
        ### Autenticación
        
        Actualmente no requiere autenticación. Para producción, se recomienda agregar autenticación.
        """,
        routes=[],
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "Inspections",
            "description": "Operaciones de inspección de imágenes",
        },
        {
            "name": "System",
            "description": "Operaciones del sistema (health, metrics, settings)",
        },
        {
            "name": "WebSocket",
            "description": "WebSocket para streaming en tiempo real",
        },
    ]
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://localhost:{settings.api_port}",
            "description": "Servidor de desarrollo",
        },
    ]
    
    # Add contact information
    openapi_schema["info"]["contact"] = {
        "name": "Blatam Academy",
        "email": "support@blatam.academy",
    }
    
    custom_openapi.schema = openapi_schema
    return openapi_schema



