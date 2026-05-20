"""
AI Project Generator - Main Entry Point
========================================

Punto de entrada principal para el generador automático de proyectos de IA.

Uso fácil:
    python main.py

O con configuración personalizada, edita este archivo.
"""

import logging
import sys
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Función principal - Inicio rápido y fácil"""
    logger.info("🚀 Iniciando AI Project Generator...")

    # Opción 1: Inicio rápido (recomendado)
    from .core.easy_setup import quick_start
    app = quick_start()
    
    # Opción 2: Con configuración personalizada (descomenta para usar)
    # from .core.easy_setup import create_app_easy
    # app = create_app_easy(
    #     enable_cache=True,
    #     enable_metrics=True,
    #     redis_url="redis://localhost:6379"  # Opcional
    # )
    
    # Opción 3: Preset para producción (descomenta para usar)
    # from .core.easy_setup import create_app_production
    # app = create_app_production(redis_url="redis://localhost:6379")
    
    # Opción 4: Preset para serverless (descomenta para usar)
    # from .core.easy_setup import create_app_serverless
    # app = create_app_serverless()

    logger.info("✅ Aplicación creada exitosamente")
    logger.info("📡 Servidor disponible en http://0.0.0.0:8020")
    logger.info("🏥 Health check: http://0.0.0.0:8020/health")
    logger.info("📊 Métricas: http://0.0.0.0:8020/metrics")

    # Ejecutar servidor
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8020,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()


