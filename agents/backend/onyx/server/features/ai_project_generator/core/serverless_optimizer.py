"""
Serverless Optimizer - Optimizaciones para entornos serverless
==============================================================

Optimizaciones para reducir cold start times y mejorar performance
en entornos serverless (AWS Lambda, Azure Functions, GCP Cloud Functions).
"""

import logging
import importlib
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path

from .microservices_config import get_microservices_config

logger = logging.getLogger(__name__)


class ServerlessOptimizer:
    """Optimizador para entornos serverless"""
    
    def __init__(self):
        self.config = get_microservices_config()
        self.optimized_modules: List[str] = []
        self._preload_modules()
    
    def _preload_modules(self):
        """Precarga módulos críticos para reducir cold start"""
        if not self.config.minimize_cold_start:
            return
        
        critical_modules = [
            "fastapi",
            "pydantic",
            "uvicorn",
            "asyncio",
            "json",
            "logging",
            "typing",
            "datetime",
        ]
        
        for module_name in critical_modules:
            try:
                importlib.import_module(module_name)
                self.optimized_modules.append(module_name)
                logger.debug(f"Preloaded module: {module_name}")
            except ImportError:
                logger.warning(f"Could not preload module: {module_name}")
    
    def optimize_imports(self, app_module: str) -> Dict[str, Any]:
        """
        Optimiza imports para reducir cold start.
        
        Args:
            app_module: Módulo principal de la aplicación
        
        Returns:
            Diccionario con información de optimización
        """
        optimizations = {
            "preloaded_modules": len(self.optimized_modules),
            "suggestions": []
        }
        
        # Analizar imports del módulo
        try:
            module = importlib.import_module(app_module)
            imports = self._analyze_imports(module)
            
            # Sugerencias de optimización
            if len(imports.get("heavy_imports", [])) > 0:
                optimizations["suggestions"].append({
                    "type": "lazy_imports",
                    "message": "Consider using lazy imports for heavy modules",
                    "modules": imports["heavy_imports"]
                })
            
            if imports.get("unused_imports"):
                optimizations["suggestions"].append({
                    "type": "remove_unused",
                    "message": "Remove unused imports to reduce cold start",
                    "modules": imports["unused_imports"]
                })
        
        except Exception as e:
            logger.error(f"Failed to analyze imports: {e}")
        
        return optimizations
    
    def _analyze_imports(self, module) -> Dict[str, Any]:
        """Analiza imports de un módulo"""
        heavy_modules = [
            "pandas", "numpy", "tensorflow", "torch",
            "sklearn", "matplotlib", "seaborn"
        ]
        
        unused_imports = []
        heavy_imports = []
        
        # Análisis simplificado
        if hasattr(module, "__file__"):
            try:
                with open(module.__file__, "r") as f:
                    content = f.read()
                    for heavy in heavy_modules:
                        if heavy in content:
                            heavy_imports.append(heavy)
            except Exception:
                pass
        
        return {
            "heavy_imports": heavy_imports,
            "unused_imports": unused_imports
        }
    
    def create_lambda_handler(self, app) -> callable:
        """
        Crea handler optimizado para AWS Lambda.
        
        Args:
            app: Aplicación FastAPI
        
        Returns:
            Handler function para Lambda
        """
        from mangum import Mangum
        
        try:
            handler = Mangum(app, lifespan="off")
            logger.info("Lambda handler created with Mangum")
            return handler
        except ImportError:
            logger.warning("Mangum not available. Install with: pip install mangum")
            # Fallback handler básico
            return self._create_basic_lambda_handler(app)
    
    def _create_basic_lambda_handler(self, app) -> callable:
        """Crea handler básico para Lambda"""
        async def handler(event, context):
            from fastapi import Request
            from fastapi.responses import Response
            
            # Convertir evento Lambda a request FastAPI
            # Implementación simplificada
            return {
                "statusCode": 200,
                "body": "Lambda handler - install mangum for full support"
            }
        
        return handler
    
    def optimize_for_azure_functions(self, app) -> Dict[str, Any]:
        """
        Optimiza aplicación para Azure Functions.
        
        Args:
            app: Aplicación FastAPI
        
        Returns:
            Configuración para Azure Functions
        """
        return {
            "bindings": [
                {
                    "type": "httpTrigger",
                    "direction": "in",
                    "name": "req",
                    "methods": ["get", "post", "put", "delete", "patch"]
                },
                {
                    "type": "http",
                    "direction": "out",
                    "name": "$return"
                }
            ],
            "scriptFile": "__init__.py",
            "entryPoint": "main",
            "timeout": self.config.serverless_timeout
        }
    
    def optimize_for_gcp_functions(self, app) -> Dict[str, Any]:
        """
        Optimiza aplicación para GCP Cloud Functions.
        
        Args:
            app: Aplicación FastAPI
        
        Returns:
            Configuración para GCP Functions
        """
        return {
            "runtime": "python39",
            "entryPoint": "main",
            "timeout": f"{self.config.serverless_timeout}s",
            "memory": "256MB",
            "maxInstances": 10
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones de optimización"""
        recommendations = []
        
        if self.config.minimize_cold_start:
            recommendations.append({
                "type": "preload_modules",
                "priority": "high",
                "message": "Preload critical modules to reduce cold start",
                "action": "Modules are already preloaded"
            })
        
        recommendations.append({
            "type": "minimize_dependencies",
            "priority": "medium",
            "message": "Minimize dependencies to reduce package size",
            "action": "Review requirements.txt and remove unused dependencies"
        })
        
        recommendations.append({
            "type": "use_lazy_imports",
            "priority": "medium",
            "message": "Use lazy imports for heavy modules",
            "action": "Import heavy modules only when needed"
        })
        
        recommendations.append({
            "type": "optimize_memory",
            "priority": "low",
            "message": "Optimize memory usage",
            "action": "Use generators and streaming for large data"
        })
        
        return recommendations


def get_serverless_optimizer() -> ServerlessOptimizer:
    """Obtiene instancia de serverless optimizer"""
    return ServerlessOptimizer()















