from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
from loguru import logger
import structlog
import tracemalloc
import psutil
from api import app
from services.seo_service import UltraOptimizedSEOService
from services.batch_service import BatchProcessingService
from core.metrics import PerformanceTracker, MetricsCollector
from utils.config import load_config
    import pytest
    import subprocess
        from test_ultra_optimized import UltraOptimizedTester
    import argparse
from typing import Any, List, Dict, Optional
import logging
"""
Punto de entrada principal para el servicio SEO ultra-optimizado en producción.
Configuración completa con logging, métricas y manejo de señales.
"""




# Configuración global
config = None
seo_service = None
batch_service = None
performance_tracker = None
metrics_collector = None


@asynccontextmanager
async def lifespan(app) -> Any:
    """Gestión del ciclo de vida de la aplicación."""
    global config, seo_service, batch_service, performance_tracker, metrics_collector
    
    # Inicialización
    logger.info("🚀 Starting SEO Analysis API v2 - Ultra Optimized")
    
    try:
        # Cargar configuración
        config = load_config()
        logger.info("✅ Configuration loaded")
        
        # Inicializar componentes
        performance_tracker = PerformanceTracker(config.get('performance', {}))
        metrics_collector = MetricsCollector()
        
        # Inicializar servicios
        seo_service = UltraOptimizedSEOService(config.get('seo_service', {}))
        batch_service = BatchProcessingService(config.get('batch_service', {}))
        
        # Configurar logging
        setup_logging(config.get('logging', {}))
        
        # Iniciar monitoreo de memoria
        if config.get('performance', {}).get('enable_tracemalloc', True):
            tracemalloc.start()
            logger.info("✅ Memory monitoring enabled")
        
        # Configurar manejo de señales
        setup_signal_handlers()
        
        # Warm-up del servicio
        await warm_up_service()
        
        logger.info("✅ Application startup completed")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("🛑 Shutting down application")
        await cleanup()


def setup_logging(logging_config: Dict[str, Any]):
    """Configura logging estructurado."""
    # Configurar loguru
    logger.remove()  # Remover handler por defecto
    
    # Handler para consola
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level=logging_config.get('console_level', 'INFO'),
        colorize=True
    )
    
    # Handler para archivo
    logger.add(
        logging_config.get('file_path', 'logs/seo_service.log'),
        rotation=logging_config.get('rotation', '100 MB'),
        compression=logging_config.get('compression', 'zstd'),
        level=logging_config.get('file_level', 'DEBUG'),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        backtrace=True,
        diagnose=True
    )
    
    # Handler para errores
    logger.add(
        logging_config.get('error_file_path', 'logs/errors.log'),
        rotation=logging_config.get('error_rotation', '50 MB'),
        compression=logging_config.get('compression', 'zstd'),
        level='ERROR',
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        backtrace=True,
        diagnose=True
    )
    
    # Configurar structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logger.info("✅ Logging configured")


def setup_signal_handlers():
    """Configura manejo de señales del sistema."""
    def signal_handler(signum, frame) -> Any:
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("✅ Signal handlers configured")


async def warm_up_service():
    """Realiza warm-up del servicio."""
    try:
        logger.info("🔥 Warming up SEO service")
        
        # Test básico de funcionalidad
        test_url = "https://example.com"
        test_request = SEOScrapeRequest(url=test_url)
        
        result = await seo_service.scrape(test_request)
        
        if result.success:
            logger.info("✅ Service warm-up successful")
        else:
            logger.warning("⚠️ Service warm-up completed with warnings")
            
    except Exception as e:
        logger.error(f"❌ Service warm-up failed: {e}")
        # No fallar el startup por warm-up


async def cleanup():
    """Limpia recursos al cerrar."""
    global seo_service, batch_service, performance_tracker
    
    try:
        # Cerrar servicios
        if seo_service:
            await seo_service.close()
            logger.info("✅ SEO service closed")
        
        if batch_service:
            batch_service.reset_stats()
            logger.info("✅ Batch service cleaned")
        
        # Detener tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            logger.info("✅ Memory monitoring stopped")
        
        # Log final de métricas
        if performance_tracker:
            final_stats = performance_tracker.get_performance_summary()
            logger.info(f"📊 Final stats: {final_stats}")
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")


def get_server_config() -> Dict[str, Any]:
    """Obtiene configuración del servidor."""
    return {
        'host': config.get('server', {}).get('host', '0.0.0.0'),
        'port': config.get('server', {}).get('port', 8000),
        'workers': config.get('server', {}).get('workers', 4),
        'loop': config.get('server', {}).get('loop', 'uvloop'),
        'http': config.get('server', {}).get('http', 'httptools'),
        'access_log': config.get('server', {}).get('access_log', False),
        'log_level': config.get('server', {}).get('log_level', 'info'),
        'reload': config.get('server', {}).get('reload', False),
        'ssl_keyfile': config.get('server', {}).get('ssl_keyfile'),
        'ssl_certfile': config.get('server', {}).get('ssl_certfile'),
    }


def run_production():
    """Ejecuta la aplicación en modo producción."""
    global app
    
    # Configurar lifespan
    app.router.lifespan_context = lifespan
    
    # Obtener configuración del servidor
    server_config = get_server_config()
    
    logger.info(f"🚀 Starting production server with config: {server_config}")
    
    # Ejecutar servidor
    uvicorn.run(
        app,
        **server_config
    )


def run_development():
    """Ejecuta la aplicación en modo desarrollo."""
    global app
    
    # Configurar lifespan
    app.router.lifespan_context = lifespan
    
    logger.info("🚀 Starting development server")
    
    # Ejecutar servidor con reload
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )


def run_tests():
    """Ejecuta tests de la aplicación."""
    
    logger.info("🧪 Running tests")
    
    try:
        # Ejecutar tests con pytest
        result = subprocess.run([
            "pytest", "tests/", "-v", "--cov=seo", "--cov-report=html"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Tests passed")
        else:
            logger.error(f"❌ Tests failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")


def run_benchmarks():
    """Ejecuta benchmarks de rendimiento."""
    logger.info("📊 Running performance benchmarks")
    
    try:
        # Importar y ejecutar benchmarks
        
        async def run_benchmarks_async():
            
    """run_benchmarks_async function."""
tester = UltraOptimizedTester()
            await tester.run_performance_test()
        
        asyncio.run(run_benchmarks_async())
        logger.info("✅ Benchmarks completed")
        
    except Exception as e:
        logger.error(f"❌ Benchmark execution failed: {e}")


def show_status():
    """Muestra estado del sistema."""
    logger.info("📊 System Status")
    
    # Información del sistema
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    status_info = {
        'cpu_usage': f"{cpu_percent:.1f}%",
        'memory_usage': f"{memory.percent:.1f}%",
        'disk_usage': f"{disk.percent:.1f}%",
        'memory_available': f"{memory.available / 1024 / 1024 / 1024:.1f}GB",
        'disk_free': f"{disk.free / 1024 / 1024 / 1024:.1f}GB"
    }
    
    for key, value in status_info.items():
        logger.info(f"  {key}: {value}")


def main():
    """Función principal."""
    
    parser = argparse.ArgumentParser(description="SEO Analysis API v2 - Ultra Optimized")
    parser.add_argument('--mode', choices=['production', 'development', 'test', 'benchmark', 'status'],
                       default='production', help='Execution mode')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--host', help='Server host')
    parser.add_argument('--port', type=int, help='Server port')
    parser.add_argument('--workers', type=int, help='Number of workers')
    
    args = parser.parse_args()
    
    # Cargar configuración si se especifica
    if args.config:
        global config
        config = load_config(args.config)
    
    # Sobrescribir configuración con argumentos
    if args.host or args.port or args.workers:
        if not config:
            config = {}
        if 'server' not in config:
            config['server'] = {}
        
        if args.host:
            config['server']['host'] = args.host
        if args.port:
            config['server']['port'] = args.port
        if args.workers:
            config['server']['workers'] = args.workers
    
    # Ejecutar según el modo
    if args.mode == 'production':
        run_production()
    elif args.mode == 'development':
        run_development()
    elif args.mode == 'test':
        run_tests()
    elif args.mode == 'benchmark':
        run_benchmarks()
    elif args.mode == 'status':
        show_status()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


match __name__:
    case "__main__":
    main() 