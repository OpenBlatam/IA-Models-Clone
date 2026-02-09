"""
Lifespan context manager para startup y shutdown
Optimizado para serverless y microservicios
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from config.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    """Lifespan context manager para startup y shutdown"""
    # Startup
    logger.info("Starting Suno Clone AI service...")
    
    try:
        # Optimizaciones globales de rendimiento
        from core.performance_optimizer import get_performance_optimizer
        from core.memory_optimizer import get_memory_optimizer
        from core.code_optimizer import get_code_optimizer
        
        perf_optimizer = get_performance_optimizer()
        perf_optimizer.optimize_all()
        
        memory_optimizer = get_memory_optimizer()
        if settings.is_lambda:
            memory_optimizer.enable_aggressive_gc()
        else:
            memory_optimizer.optimize_gc()
        
        code_optimizer = get_code_optimizer()
        code_optimizer.optimize_imports()
        if not settings.is_lambda:  # JIT puede aumentar cold start
            code_optimizer.enable_jit()
        
        # Optimizaciones de CPU
        from core.cpu_optimizer import get_cpu_optimizer
        cpu_optimizer = get_cpu_optimizer()
        cpu_optimizer.optimize()
        
        logger.info("Performance optimizations applied")
        
        # Optimizaciones serverless (solo si es Lambda)
        if settings.is_lambda:
            from core.serverless_optimizer import get_serverless_optimizer
            optimizer = get_serverless_optimizer()
            optimizer.optimize_for_lambda()
            logger.info("Serverless optimizations applied")
        
        # Pre-cargar servicios críticos (solo si no es Lambda para reducir cold start)
        if not settings.is_lambda:
            from core.music_generator import get_music_generator
            from core.cache_manager import get_cache_manager
            from services.song_service import SongService
            
            # Inicializar servicios
            logger.info("Initializing services...")
            get_music_generator()  # Pre-cargar modelo
            get_cache_manager()  # Inicializar caché
            SongService()  # Inicializar base de datos
        
        # Inicializar connection pools
        from core.connection_pool import get_connection_pool_manager
        pool_manager = get_connection_pool_manager()
        logger.info("Connection pool manager initialized")
        
        # Inicializar optimizadores de I/O y async
        from core.io_optimizer import get_io_optimizer
        from core.async_optimizer import get_async_optimizer
        from core.prefetch_optimizer import get_prefetch_optimizer
        from core.database_optimizer import get_database_optimizer
        from core.query_preparation import get_query_preparer
        from core.serialization_optimizer import get_serialization_optimizer
        
        io_optimizer = get_io_optimizer()
        async_optimizer = get_async_optimizer()
        prefetch_optimizer = get_prefetch_optimizer()
        db_optimizer = get_database_optimizer()
        query_preparer = get_query_preparer()
        serialization_optimizer = get_serialization_optimizer()
        
        # Optimizadores adicionales
        from core.connection_reuse import get_connection_reuse
        from core.request_batching import get_request_batcher
        from core.lazy_loader import get_lazy_loader
        from utils.parallel_processor import get_parallel_processor
        from core.data_prefetch import get_data_prefetcher
        
        connection_reuse = get_connection_reuse()
        request_batcher = get_request_batcher()
        lazy_loader = get_lazy_loader()
        parallel_processor = get_parallel_processor()
        data_prefetcher = get_data_prefetcher()
        
        logger.info("All optimizers initialized")
        
        # Inicializar event bus si está habilitado
        if settings.enable_event_bus:
            from services.event_bus import get_event_bus
            event_bus = get_event_bus()
            if settings.event_bus_backend != "memory":
                from services.event_bus_backends import create_event_bus_backend
                backend = create_event_bus_backend(
                    settings.event_bus_backend,
                    redis_url=settings.redis_url if settings.event_bus_backend == "redis" else None,
                    queue_url=settings.sqs_queue_url if settings.event_bus_backend == "sqs" else None,
                    region=settings.aws_region
                )
                # Integrar backend con event bus
                logger.info(f"Event bus backend initialized: {settings.event_bus_backend}")
        
        logger.info("Suno Clone AI service started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Suno Clone AI service...")
    
    try:
        # Cerrar connection pools
        from core.connection_pool import get_connection_pool_manager
        pool_manager = get_connection_pool_manager()
        await pool_manager.close_all()
        logger.info("Connection pools closed")
        
        # Cleanup si es necesario
        logger.info("Cleaning up resources...")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    logger.info("Suno Clone AI service stopped")

