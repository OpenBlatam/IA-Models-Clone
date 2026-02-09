"""
Interfaces y Abstracciones Base

Define contratos para servicios y componentes del sistema.
Esto permite mayor modularidad y facilita testing y extensibilidad.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


class IMusicGenerator(ABC):
    """Interfaz para generadores de música"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Genera música desde un prompt"""
        pass


class IAudioProcessor(ABC):
    """Interfaz para procesadores de audio"""
    
    @abstractmethod
    async def process(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Procesa un archivo de audio"""
        pass


class ICacheManager(ABC):
    """Interfaz para gestores de caché"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del caché"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Almacena un valor en el caché"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Elimina un valor del caché"""
        pass


class IStorageBackend(ABC):
    """Interfaz para backends de almacenamiento"""
    
    @abstractmethod
    async def save(self, path: str, data: bytes) -> bool:
        """Guarda datos"""
        pass
    
    @abstractmethod
    async def load(self, path: str) -> Optional[bytes]:
        """Carga datos"""
        pass
    
    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Elimina datos"""
        pass


class INotificationService(ABC):
    """Interfaz para servicios de notificaciones"""
    
    @abstractmethod
    async def send(self, user_id: str, message: str, **kwargs) -> bool:
        """Envía una notificación"""
        pass


class IAnalyticsService(ABC):
    """Interfaz para servicios de analytics"""
    
    @abstractmethod
    async def track(self, event: str, data: Dict[str, Any]) -> bool:
        """Registra un evento"""
        pass
    
    @abstractmethod
    async def get_stats(self, **filters) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        pass


class IAuthenticationService(ABC):
    """Interfaz para servicios de autenticación"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Autentica un usuario"""
        pass
    
    @abstractmethod
    async def authorize(self, user_id: str, resource: str, action: str) -> bool:
        """Verifica autorización"""
        pass


class IPlugin(ABC):
    """Interfaz base para plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del plugin"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Versión del plugin"""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Inicializa el plugin"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Cierra el plugin"""
        pass

