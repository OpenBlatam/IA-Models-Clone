"""
Ejemplo Personalizado - Configuración avanzada
==============================================

Ejemplo con configuración personalizada.
"""

from core.easy_setup import create_app_easy
import uvicorn

# Crear aplicación con configuración personalizada
app = create_app_easy(
    title="Mi API de Generación de Proyectos",
    version="2.0.0",
    enable_cache=True,      # Cache habilitado
    enable_metrics=True,   # Métricas Prometheus
    enable_workers=False,   # Sin workers (más simple)
    enable_events=False,   # Sin eventos
    redis_url="redis://localhost:6379"  # Redis opcional
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)















