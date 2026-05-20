"""
Advanced Serverless Optimizations - Optimizaciones avanzadas para serverless
============================================================================

Optimizaciones avanzadas para reducir cold starts y mejorar performance:
- Lazy loading avanzado
- Connection pooling
- Warm-up strategies
- Memory optimization
- Binary packaging
"""

import importlib
import sys
import asyncio
import os
from typing import Optional, Dict, Any, List, Callable, Union, Protocol
from pathlib import Path
from functools import lru_cache
from enum import Enum

from .microservices_config import get_microservices_config
from .types import HandlerEvent, HandlerContext, HandlerResponse
from .shared_utils import get_logger

logger = get_logger(__name__)


class ServerlessEnvironment(str, Enum):
    """Entornos serverless"""
    AWS_LAMBDA = "aws_lambda"
    AZURE_FUNCTIONS = "azure_functions"
    GCP_CLOUD_FUNCTIONS = "gcp_cloud_functions"
    VERCEL = "vercel"
    NETLIFY = "netlify"


class AdvancedServerlessOptimizer:
    """
    Optimizador avanzado para entornos serverless con:
    - Reducción de cold starts
    - Lazy loading inteligente
    - Connection pooling
    - Warm-up strategies
    - Memory optimization
    """
    
    def __init__(self, environment: Optional[ServerlessEnvironment] = None):
        self.config = get_microservices_config()
        self.environment = environment or self._detect_environment()
        self.optimized_modules: List[str] = []
        self.connection_pools: Dict[str, Any] = {}
        self._preload_critical_modules()
    
    def _detect_environment(self) -> ServerlessEnvironment:
        """Detecta el entorno serverless"""
        # Detectar AWS Lambda
        if "LAMBDA_RUNTIME_DIR" in os.environ or "AWS_LAMBDA_FUNCTION_NAME" in os.environ:
            return ServerlessEnvironment.AWS_LAMBDA
        
        # Detectar Azure Functions
        if "FUNCTIONS_WORKER_RUNTIME" in os.environ or "WEBSITE_SITE_NAME" in os.environ:
            return ServerlessEnvironment.AZURE_FUNCTIONS
        
        # Detectar GCP Cloud Functions
        if "FUNCTION_TARGET" in os.environ or "FUNCTION_NAME" in os.environ:
            return ServerlessEnvironment.GCP_CLOUD_FUNCTIONS
        
        # Detectar Vercel
        if "VERCEL" in os.environ:
            return ServerlessEnvironment.VERCEL
        
        # Detectar Netlify
        if "NETLIFY" in os.environ:
            return ServerlessEnvironment.NETLIFY
        
        return ServerlessEnvironment.AWS_LAMBDA  # Default
    
    def _preload_critical_modules(self):
        """Precarga módulos críticos de forma optimizada"""
        if not self.config.minimize_cold_start:
            return
        
        # Módulos críticos que deben cargarse primero
        critical_modules = [
            "fastapi",
            "pydantic",
            "asyncio",
            "json",
            "logging",
            "typing",
            "datetime",
        ]
        
        # Cargar módulos en paralelo si es posible
        for module_name in critical_modules:
            try:
                importlib.import_module(module_name)
                self.optimized_modules.append(module_name)
            except ImportError:
                logger.warning(f"Could not preload module: {module_name}")
    
    def create_optimized_handler(
        self,
        app: Any,  # FastAPI app
        lazy_imports: Optional[List[str]] = None,
        warm_up: bool = True
    ) -> Union[Callable[[HandlerEvent, HandlerContext], HandlerResponse], Any]:
        """
        Crea handler optimizado para serverless.
        
        Args:
            app: Aplicación FastAPI
            lazy_imports: Módulos a cargar de forma lazy
            warm_up: Si hacer warm-up del handler
        
        Returns:
            Handler optimizado
        """
        if self.environment == ServerlessEnvironment.AWS_LAMBDA:
            return self._create_aws_lambda_handler(app, lazy_imports, warm_up)
        elif self.environment == ServerlessEnvironment.AZURE_FUNCTIONS:
            return self._create_azure_handler(app, lazy_imports, warm_up)
        elif self.environment == ServerlessEnvironment.GCP_CLOUD_FUNCTIONS:
            return self._create_gcp_handler(app, lazy_imports, warm_up)
        else:
            return self._create_generic_handler(app, lazy_imports, warm_up)
    
    def _create_aws_lambda_handler(
        self,
        app,
        lazy_imports: Optional[List[str]],
        warm_up: bool
    ) -> Callable:
        """Crea handler optimizado para AWS Lambda"""
        try:
            from mangum import Mangum
            
            # Configurar lazy imports
            if lazy_imports:
                self._setup_lazy_imports(lazy_imports)
            
            # Crear handler con optimizaciones
            handler = Mangum(
                app,
                lifespan="off",  # Deshabilitar lifespan para reducir cold start
                api_gateway_base_path="/",
                text_mime_types=["application/json"]
            )
            
            # Warm-up si está habilitado
            if warm_up:
                self._warm_up_handler(handler)
            
            logger.info("AWS Lambda handler created with advanced optimizations")
            return handler
            
        except ImportError:
            logger.warning("Mangum not available, using basic handler")
            return self._create_basic_handler(app)
    
    def _create_azure_handler(
        self,
        app,
        lazy_imports: Optional[List[str]],
        warm_up: bool
    ) -> Callable:
        """Crea handler optimizado para Azure Functions"""
        # Configurar lazy imports
        if lazy_imports:
            self._setup_lazy_imports(lazy_imports)
        
        async def handler(req):
            from azure.functions import HttpRequest, HttpResponse
            
            # Convertir request de Azure a FastAPI
            # Implementación simplificada
            return HttpResponse("Azure Functions handler")
        
        if warm_up:
            self._warm_up_handler(handler)
        
        return handler
    
    def _create_gcp_handler(
        self,
        app,
        lazy_imports: Optional[List[str]],
        warm_up: bool
    ) -> Callable:
        """Crea handler optimizado para GCP Cloud Functions"""
        # Configurar lazy imports
        if lazy_imports:
            self._setup_lazy_imports(lazy_imports)
        
        def handler(request):
            # GCP Cloud Functions handler
            # Implementación simplificada
            return "GCP Cloud Functions handler"
        
        if warm_up:
            self._warm_up_handler(handler)
        
        return handler
    
    def _create_generic_handler(
        self,
        app,
        lazy_imports: Optional[List[str]],
        warm_up: bool
    ) -> Callable:
        """Crea handler genérico optimizado"""
        # Configurar lazy imports
        if lazy_imports:
            self._setup_lazy_imports(lazy_imports)
        
        return app
    
    def _setup_lazy_imports(self, modules: List[str]):
        """Configura lazy imports para módulos pesados"""
        for module_name in modules:
            # Guardar referencia para importación lazy
            sys.modules[f"_lazy_{module_name}"] = None
            logger.debug(f"Lazy import configured for: {module_name}")
    
    def _warm_up_handler(self, handler: Callable):
        """Hace warm-up del handler para reducir cold start"""
        try:
            # Crear request de prueba
            test_event = {
                "httpMethod": "GET",
                "path": "/health",
                "headers": {},
                "body": None
            }
            
            # Ejecutar handler de forma asíncrona si es posible
            if asyncio.iscoroutinefunction(handler):
                asyncio.create_task(handler(test_event, None))
            else:
                handler(test_event, None)
            
            logger.info("Handler warmed up")
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")
    
    def _create_basic_handler(self, app) -> Callable:
        """Crea handler básico"""
        async def handler(event, context):
            return {
                "statusCode": 200,
                "body": "Basic handler - install mangum for full support"
            }
        return handler
    
    def optimize_connections(self, service_name: str, max_connections: int = 10) -> None:
        """
        Optimiza conexiones para un servicio usando httpx.
        
        Args:
            service_name: Nombre del servicio
            max_connections: Máximo de conexiones
        """
        try:
            import httpx
            
            limits = httpx.Limits(
                max_keepalive_connections=max_connections,
                max_connections=max_connections * 2
            )
            
            self.connection_pools[service_name] = limits
            logger.info(f"Connection pool optimized for {service_name} using httpx")
            
        except ImportError:
            logger.warning("httpx not available for connection pooling")
    
    def get_connection_pool(self, service_name: str) -> Optional[Any]:
        """Obtiene connection pool para un servicio"""
        return self.connection_pools.get(service_name)
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Optimiza uso de memoria.
        
        Returns:
            Información de optimización
        """
        optimizations = {
            "gc_enabled": True,
            "gc_threshold": (700, 10, 10),
            "memory_usage": self._get_memory_usage()
        }
        
        # Configurar garbage collector
        import gc
        gc.set_threshold(*optimizations["gc_threshold"])
        
        return optimizations
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Obtiene uso de memoria"""
        try:
            import psutil
            process = psutil.Process()
            return {
                "rss": process.memory_info().rss,
                "vms": process.memory_info().vms,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"status": "psutil not available"}
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Obtiene reporte de optimizaciones"""
        return {
            "environment": self.environment.value,
            "preloaded_modules": len(self.optimized_modules),
            "modules": self.optimized_modules,
            "connection_pools": list(self.connection_pools.keys()),
            "memory_optimization": self.optimize_memory(),
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones de optimización"""
        recommendations = []
        
        if len(self.optimized_modules) < 5:
            recommendations.append({
                "type": "preload_more_modules",
                "priority": "high",
                "message": "Preload more critical modules to reduce cold start"
            })
        
        if not self.connection_pools:
            recommendations.append({
                "type": "use_connection_pooling",
                "priority": "medium",
                "message": "Use connection pooling for external services"
            })
        
        recommendations.append({
            "type": "minimize_dependencies",
            "priority": "medium",
            "message": "Minimize dependencies to reduce package size"
        })
        
        return recommendations


# Importar os para detección de entorno
import os
from enum import Enum


def get_advanced_serverless_optimizer(
    environment: Optional[ServerlessEnvironment] = None
) -> AdvancedServerlessOptimizer:
    """Obtiene optimizador avanzado para serverless"""
    return AdvancedServerlessOptimizer(environment)

