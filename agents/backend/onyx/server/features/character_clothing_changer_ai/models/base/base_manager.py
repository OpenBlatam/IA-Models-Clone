"""
Base Manager Class
==================
Clase base para todos los managers del sistema
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import time
import threading


class BaseManager(ABC):
    """
    Clase base para todos los managers
    Proporciona funcionalidad común: logging, estadísticas, lifecycle
    """
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.start_time = None
        self.stats = {
            'operations_count': 0,
            'errors_count': 0,
            'last_operation_time': None,
            'total_operations_time': 0.0
        }
        self._lock = threading.Lock()
    
    def initialize(self) -> bool:
        """Inicializar el manager"""
        if self.initialized:
            return True
        
        try:
            self._initialize()
            self.initialized = True
            self.start_time = time.time()
            return True
        except Exception as e:
            self._log_error(f"Error initializing {self.name}: {e}")
            return False
    
    def shutdown(self):
        """Cerrar el manager"""
        if not self.initialized:
            return
        
        try:
            self._shutdown()
            self.initialized = False
        except Exception as e:
            self._log_error(f"Error shutting down {self.name}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del manager"""
        with self._lock:
            uptime = time.time() - self.start_time if self.start_time else 0
            return {
                **self.stats,
                'name': self.name,
                'initialized': self.initialized,
                'uptime': uptime,
                'avg_operation_time': (
                    self.stats['total_operations_time'] / self.stats['operations_count']
                    if self.stats['operations_count'] > 0 else 0
                )
            }
    
    def reset_stats(self):
        """Resetear estadísticas"""
        with self._lock:
            self.stats = {
                'operations_count': 0,
                'errors_count': 0,
                'last_operation_time': None,
                'total_operations_time': 0.0
            }
    
    def _record_operation(self, operation_time: float):
        """Registrar operación"""
        with self._lock:
            self.stats['operations_count'] += 1
            self.stats['last_operation_time'] = time.time()
            self.stats['total_operations_time'] += operation_time
    
    def _record_error(self):
        """Registrar error"""
        with self._lock:
            self.stats['errors_count'] += 1
    
    def _log_error(self, message: str):
        """Log de error (puede ser sobrescrito)"""
        print(f"[ERROR] {self.name}: {message}")
    
    def _log_info(self, message: str):
        """Log de información (puede ser sobrescrito)"""
        print(f"[INFO] {self.name}: {message}")
    
    @abstractmethod
    def _initialize(self):
        """Inicialización específica del manager"""
        pass
    
    @abstractmethod
    def _shutdown(self):
        """Cierre específico del manager"""
        pass
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


class BaseProcessor(ABC):
    """
    Clase base para procesadores
    """
    
    def __init__(self, name: str):
        self.name = name
        self.processing = False
        self.queue_size = 0
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Procesar item"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del procesador"""
        return {
            'name': self.name,
            'processing': self.processing,
            'queue_size': self.queue_size
        }


class BaseSystem(ABC):
    """
    Clase base para sistemas
    """
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.config: Dict[str, Any] = {}
    
    def configure(self, config: Dict[str, Any]):
        """Configurar sistema"""
        self.config.update(config)
        self._apply_config()
    
    @abstractmethod
    def _apply_config(self):
        """Aplicar configuración"""
        pass
    
    def enable(self):
        """Habilitar sistema"""
        self.enabled = True
    
    def disable(self):
        """Deshabilitar sistema"""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """Verificar si está habilitado"""
        return self.enabled

