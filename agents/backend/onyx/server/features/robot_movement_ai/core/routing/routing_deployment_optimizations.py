"""
Routing Deployment Optimizations
==================================

Optimizaciones para deployment y producción.
Incluye: Health checks, Graceful shutdown, Resource management, etc.
"""

import logging
import signal
import sys
import atexit
import time
from typing import Dict, Any, List, Optional, Callable
import threading

logger = logging.getLogger(__name__)


class HealthChecker:
    """Verificador de salud del sistema."""
    
    def __init__(self):
        """Inicializar health checker."""
        self.health_status: Dict[str, Any] = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        self.health_checks: List[Callable] = []
        self.lock = threading.Lock()
    
    def add_health_check(self, name: str, check_func: Callable):
        """
        Agregar verificación de salud.
        
        Args:
            name: Nombre de la verificación
            check_func: Función que retorna (is_healthy, message)
        """
        self.health_checks.append((name, check_func))
    
    def check_health(self) -> Dict[str, Any]:
        """Verificar salud del sistema."""
        with self.lock:
            all_healthy = True
            checks = {}
            
            for name, check_func in self.health_checks:
                try:
                    is_healthy, message = check_func()
                    checks[name] = {
                        'healthy': is_healthy,
                        'message': message,
                        'timestamp': time.time()
                    }
                    if not is_healthy:
                        all_healthy = False
                except Exception as e:
                    checks[name] = {
                        'healthy': False,
                        'message': f"Check failed: {e}",
                        'timestamp': time.time()
                    }
                    all_healthy = False
            
            self.health_status = {
                'status': 'healthy' if all_healthy else 'unhealthy',
                'timestamp': time.time(),
                'checks': checks
            }
            
            return self.health_status
    
    def get_health(self) -> Dict[str, Any]:
        """Obtener estado de salud actual."""
        with self.lock:
            return self.health_status.copy()


class GracefulShutdown:
    """Gestor de apagado graceful."""
    
    def __init__(self):
        """Inicializar gestor de shutdown."""
        self.shutdown_handlers: List[Callable] = []
        self.shutting_down = False
        self.lock = threading.Lock()
        
        # Registrar handlers de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
    
    def _signal_handler(self, signum, frame):
        """Manejador de señales."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
        sys.exit(0)
    
    def _cleanup(self):
        """Limpieza al salir."""
        if not self.shutting_down:
            self.shutdown()
    
    def register_shutdown_handler(self, handler: Callable):
        """
        Registrar handler de shutdown.
        
        Args:
            handler: Función a ejecutar durante shutdown
        """
        with self.lock:
            self.shutdown_handlers.append(handler)
    
    def shutdown(self, timeout: float = 30.0):
        """
        Ejecutar shutdown graceful.
        
        Args:
            timeout: Timeout en segundos
        """
        with self.lock:
            if self.shutting_down:
                return
            
            self.shutting_down = True
        
        logger.info("Starting graceful shutdown...")
        start_time = time.time()
        
        for handler in self.shutdown_handlers:
            if (time.time() - start_time) > timeout:
                logger.warning("Shutdown timeout exceeded")
                break
            
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")
        
        logger.info("Graceful shutdown completed")


class ResourceManager:
    """Gestor de recursos."""
    
    def __init__(self):
        """Inicializar gestor de recursos."""
        self.resources: Dict[str, Any] = {}
        self.cleanup_functions: Dict[str, Callable] = {}
        self.lock = threading.Lock()
    
    def register_resource(self, name: str, resource: Any, cleanup_func: Optional[Callable] = None):
        """
        Registrar recurso.
        
        Args:
            name: Nombre del recurso
            resource: Recurso a gestionar
            cleanup_func: Función de limpieza
        """
        with self.lock:
            self.resources[name] = resource
            if cleanup_func:
                self.cleanup_functions[name] = cleanup_func
    
    def get_resource(self, name: str) -> Optional[Any]:
        """Obtener recurso."""
        with self.lock:
            return self.resources.get(name)
    
    def cleanup_all(self):
        """Limpiar todos los recursos."""
        with self.lock:
            for name, cleanup_func in self.cleanup_functions.items():
                try:
                    cleanup_func()
                    logger.debug(f"Cleaned up resource: {name}")
                except Exception as e:
                    logger.error(f"Error cleaning up resource {name}: {e}")
            
            self.resources.clear()
            self.cleanup_functions.clear()


class DeploymentOptimizer:
    """Optimizador completo de deployment."""
    
    def __init__(self):
        """Inicializar optimizador de deployment."""
        self.health_checker = HealthChecker()
        self.graceful_shutdown = GracefulShutdown()
        self.resource_manager = ResourceManager()
        
        # Registrar cleanup de recursos en shutdown
        self.graceful_shutdown.register_shutdown_handler(
            self.resource_manager.cleanup_all
        )
    
    def add_health_check(self, name: str, check_func: Callable):
        """Agregar verificación de salud."""
        self.health_checker.add_health_check(name, check_func)
    
    def register_resource(self, name: str, resource: Any, cleanup_func: Optional[Callable] = None):
        """Registrar recurso."""
        self.resource_manager.register_resource(name, resource, cleanup_func)
    
    def get_health(self) -> Dict[str, Any]:
        """Obtener estado de salud."""
        return self.health_checker.check_health()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'health': self.health_checker.get_health(),
            'resources': list(self.resource_manager.resources.keys()),
            'shutdown_handlers': len(self.graceful_shutdown.shutdown_handlers)
        }

