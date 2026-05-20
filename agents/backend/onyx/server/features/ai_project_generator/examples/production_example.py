"""
Ejemplo Producción - Configuración para producción
=================================================

Ejemplo con todas las características habilitadas para producción.
"""

from core.easy_setup import create_app_production
import uvicorn

# Crear aplicación para producción
app = create_app_production(
    redis_url="redis://localhost:6379"  # Redis requerido
)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8020,
        workers=4,  # Múltiples workers para producción
        log_level="info"
    )















