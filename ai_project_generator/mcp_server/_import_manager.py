"""
Import Manager - Gestor de imports dinámico
===========================================

Gestiona los imports del módulo MCP Server de forma dinámica
y organizada, permitiendo verificar disponibilidad de componentes.
"""

import logging
from typing import Dict, Any, Optional, List
from ._imports import IMPORT_GROUPS, get_runtime_imports

logger = logging.getLogger(__name__)


class ImportManager:
    """
    Gestor de imports para el módulo MCP Server.
    
    Gestiona la importación de todos los componentes opcionales
    y proporciona utilidades para verificar su disponibilidad.
    """
    
    def __init__(self, namespace: Dict[str, Any]) -> None:
        """
        Inicializar gestor de imports.
        
        Args:
            namespace: Namespace donde se asignarán los imports (típicamente globals()).
        """
        self.namespace = namespace
        self._imported_modules: Dict[str, bool] = {}
        self._imported_symbols: Dict[str, Optional[Any]] = {}
    
    def import_all(self) -> None:
        """
        Importar todos los módulos disponibles.
        
        Realiza imports seguros de todos los grupos de módulos,
        registrando qué se importó exitosamente.
        """
        runtime_imports = get_runtime_imports()
        
        for module_path, symbols, group_name in runtime_imports:
            try:
                # Construir fromlist correctamente
                fromlist = [s.strip() for s in symbols.split(",")]
                module = __import__(
                    f".{module_path}",
                    fromlist=fromlist,
                    level=1
                )
                
                symbol_list = [s.strip() for s in symbols.split(",")]
                imported_count = 0
                
                for symbol_name in symbol_list:
                    if hasattr(module, symbol_name):
                        symbol_value = getattr(module, symbol_name)
                        self.namespace[symbol_name] = symbol_value
                        self._imported_symbols[symbol_name] = symbol_value
                        imported_count += 1
                    else:
                        logger.debug(
                            f"Symbol '{symbol_name}' not found in '{module_path}'"
                        )
                        self.namespace[symbol_name] = None
                        self._imported_symbols[symbol_name] = None
                
                if imported_count > 0:
                    self._imported_modules[module_path] = True
                    logger.debug(
                        f"Imported {imported_count}/{len(symbol_list)} symbols "
                        f"from {module_path} ({group_name})"
                    )
                else:
                    self._imported_modules[module_path] = False
                    
            except ImportError as e:
                self._imported_modules[module_path] = False
                logger.debug(f"{group_name} not available ({module_path}): {e}")
                
                # Asignar None a todos los símbolos del módulo
                symbol_list = [s.strip() for s in symbols.split(",")]
                for symbol_name in symbol_list:
                    self.namespace[symbol_name] = None
                    self._imported_symbols[symbol_name] = None
                    
            except Exception as e:
                self._imported_modules[module_path] = False
                logger.warning(
                    f"Error importing {group_name} ({module_path}): {e}",
                    exc_info=True
                )
    
    def check_imports(self) -> Dict[str, bool]:
        """
        Verificar qué componentes están disponibles.
        
        Returns:
            Diccionario con estado de cada componente (True = disponible).
        """
        return {
            symbol_name: symbol_value is not None
            for symbol_name, symbol_value in self._imported_symbols.items()
        }
    
    def get_available_features(self) -> Dict[str, bool]:
        """
        Obtener características disponibles agrupadas por categoría.
        
        Returns:
            Diccionario con características disponibles por categoría.
        """
        features: Dict[str, bool] = {}
        
        # Mapeo de símbolos a nombres de características
        feature_map = {
            "RateLimiter": "rate_limiting",
            "MCPCache": "caching",
            "RetryConfig": "retry",
            "CircuitBreaker": "circuit_breaker",
            "BatchProcessor": "batch_processing",
            "WebhookManager": "webhooks",
            "StreamResponse": "streaming",
            "PerformanceProfiler": "profiling",
            "AsyncTaskQueue": "task_queue",
            "MCPGraphQL": "graphql",
            "PluginManager": "plugins",
            "ResponseCompressor": "compression",
            "HealthChecker": "health_checks",
            "VersionedRouter": "api_versioning",
            "ServiceRegistry": "service_discovery",
            "ConnectionPool": "connection_pooling",
            "MetricsDashboard": "metrics_dashboard",
            "RequestQueue": "request_queue",
            "TenantManager": "multitenancy",
            "EventStore": "event_sourcing",
            "LockManager": "distributed_locking",
            "APIDocumentation": "api_documentation",
            "RequestInterceptor": "interceptors",
            "CQRSBus": "cqrs",
            "SagaOrchestrator": "saga",
            "MessageQueue": "message_queue",
            "AdvancedCache": "advanced_cache",
            "AdvancedValidator": "advanced_validation",
        }
        
        for symbol_name, feature_name in feature_map.items():
            features[feature_name] = self._imported_symbols.get(symbol_name) is not None
        
        return features
    
    def get_missing_imports(self) -> List[str]:
        """
        Obtener lista de componentes no disponibles.
        
        Returns:
            Lista de nombres de componentes faltantes.
        """
        return [
            symbol_name
            for symbol_name, symbol_value in self._imported_symbols.items()
            if symbol_value is None
        ]
    
    def get_import_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de imports.
        
        Returns:
            Diccionario con estadísticas de imports.
        """
        total_symbols = len(self._imported_symbols)
        available_symbols = sum(
            1 for v in self._imported_symbols.values() if v is not None
        )
        total_modules = len(self._imported_modules)
        available_modules = sum(1 for v in self._imported_modules.values() if v)
        
        return {
            "total_symbols": total_symbols,
            "available_symbols": available_symbols,
            "missing_symbols": total_symbols - available_symbols,
            "total_modules": total_modules,
            "available_modules": available_modules,
            "missing_modules": total_modules - available_modules,
            "availability_rate": (
                available_symbols / total_symbols * 100
                if total_symbols > 0
                else 0.0
            ),
        }
    
    def get_module_status(self, module_path: str) -> Optional[Dict[str, Any]]:
        """
        Obtener estado de un módulo específico.
        
        Args:
            module_path: Ruta del módulo.
            
        Returns:
            Diccionario con estado del módulo o None si no existe.
        """
        if module_path not in self._imported_modules:
            return None
        
        module_symbols = [
            name for name, value in self._imported_symbols.items()
            if name.startswith(module_path.split(".")[-1])
        ]
        
        return {
            "module_path": module_path,
            "available": self._imported_modules[module_path],
            "symbols": {
                name: value is not None
                for name, value in self._imported_symbols.items()
                if name in module_symbols
            },
        }

