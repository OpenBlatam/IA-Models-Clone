"""
Common Interfaces
=================
Interfaces comunes para sistemas similares
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


class ExecutionStatus(Enum):
    """Estado de ejecución común"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IExecutable(ABC):
    """Interfaz para sistemas ejecutables"""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Ejecutar operación"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado"""
        pass


class IProcessable(ABC):
    """Interfaz para sistemas procesables"""
    
    @abstractmethod
    def process(self, data: Any, *args, **kwargs) -> Any:
        """Procesar datos"""
        pass
    
    @abstractmethod
    def can_process(self, data: Any) -> bool:
        """Verificar si puede procesar"""
        pass


class IConfigurable(ABC):
    """Interfaz para sistemas configurables"""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]):
        """Configurar sistema"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Obtener configuración"""
        pass


class IMonitorable(ABC):
    """Interfaz para sistemas monitoreables"""
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas"""
        pass
    
    @abstractmethod
    def get_health(self) -> Dict[str, Any]:
        """Obtener salud del sistema"""
        pass


class IRetryable(ABC):
    """Interfaz para sistemas con retry"""
    
    @abstractmethod
    def execute_with_retry(
        self,
        func: Callable,
        max_attempts: int = 3,
        *args,
        **kwargs
    ) -> Any:
        """Ejecutar con retry"""
        pass


class IObservable(ABC):
    """Interfaz para sistemas observables (observer pattern)"""
    
    @abstractmethod
    def subscribe(self, observer: Callable):
        """Suscribir observer"""
        pass
    
    @abstractmethod
    def unsubscribe(self, observer: Callable):
        """Desuscribir observer"""
        pass
    
    @abstractmethod
    def notify(self, event: Any):
        """Notificar a observers"""
        pass


class IStateful(ABC):
    """Interfaz para sistemas con estado"""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Obtener estado actual"""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]):
        """Establecer estado"""
        pass
    
    @abstractmethod
    def reset_state(self):
        """Resetear estado"""
        pass


class IValidatable(ABC):
    """Interfaz para sistemas validables"""
    
    @abstractmethod
    def validate(self, data: Any) -> tuple[bool, Optional[str]]:
        """
        Validar datos
        
        Returns:
            (is_valid, error_message)
        """
        pass


class IExportable(ABC):
    """Interfaz para sistemas exportables"""
    
    @abstractmethod
    def export(self, format: str, *args, **kwargs) -> Any:
        """Exportar datos"""
        pass
    
    @abstractmethod
    def get_export_formats(self) -> List[str]:
        """Obtener formatos de exportación disponibles"""
        pass


class IImportable(ABC):
    """Interfaz para sistemas importables"""
    
    @abstractmethod
    def import_data(self, data: Any, format: str, *args, **kwargs) -> bool:
        """Importar datos"""
        pass
    
    @abstractmethod
    def get_import_formats(self) -> List[str]:
        """Obtener formatos de importación disponibles"""
        pass


class ISearchable(ABC):
    """Interfaz para sistemas buscables"""
    
    @abstractmethod
    def search(self, query: str, *args, **kwargs) -> List[Any]:
        """Buscar"""
        pass
    
    @abstractmethod
    def index(self, item: Any):
        """Indexar item"""
        pass


class ICacheable(ABC):
    """Interfaz para sistemas cacheables"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Obtener de caché"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Establecer en caché"""
        pass
    
    @abstractmethod
    def invalidate(self, key: str):
        """Invalidar caché"""
        pass


class IQueueable(ABC):
    """Interfaz para sistemas con cola"""
    
    @abstractmethod
    def enqueue(self, item: Any, priority: int = 0):
        """Agregar a cola"""
        pass
    
    @abstractmethod
    def dequeue(self) -> Optional[Any]:
        """Remover de cola"""
        pass
    
    @abstractmethod
    def get_queue_size(self) -> int:
        """Obtener tamaño de cola"""
        pass

