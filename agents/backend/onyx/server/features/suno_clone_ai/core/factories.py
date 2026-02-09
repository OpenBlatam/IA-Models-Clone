"""
Factories para crear instancias de servicios

Proporciona factories centralizadas para instanciar servicios,
permitiendo fácil intercambio de implementaciones y testing.
"""

import logging
from typing import Dict, Any, Optional, Type
from config.settings import settings

logger = logging.getLogger(__name__)


class ServiceFactory:
    """Factory base para servicios"""
    
    _instances: Dict[str, Any] = {}
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, service_name: str, service_class: Type):
        """Registra una clase de servicio"""
        cls._registry[service_name] = service_class
        logger.info(f"Service registered: {service_name}")
    
    @classmethod
    def create(cls, service_name: str, **kwargs) -> Any:
        """
        Crea una instancia de servicio (singleton)
        
        Args:
            service_name: Nombre del servicio
            **kwargs: Argumentos para el constructor
        
        Returns:
            Instancia del servicio
        """
        if service_name not in cls._instances:
            if service_name not in cls._registry:
                raise ValueError(f"Service not registered: {service_name}")
            
            service_class = cls._registry[service_name]
            cls._instances[service_name] = service_class(**kwargs)
            logger.info(f"Service instance created: {service_name}")
        
        return cls._instances[service_name]
    
    @classmethod
    def get(cls, service_name: str) -> Optional[Any]:
        """Obtiene una instancia existente"""
        return cls._instances.get(service_name)
    
    @classmethod
    def reset(cls):
        """Resetea todas las instancias (útil para testing)"""
        cls._instances.clear()


class MusicGeneratorFactory:
    """Factory para generadores de música"""
    
    @staticmethod
    def create_generator(generator_type: str = "default", **kwargs):
        """
        Crea un generador de música según el tipo
        
        Args:
            generator_type: Tipo de generador (default, fast, diffusion, optimized)
            **kwargs: Argumentos adicionales
        
        Returns:
            Instancia del generador
        """
        if generator_type == "fast":
            from core.fast_generator import FastMusicGenerator
            return FastMusicGenerator(**kwargs)
        elif generator_type == "diffusion":
            from core.diffusion_generator import DiffusionMusicGenerator
            return DiffusionMusicGenerator(**kwargs)
        elif generator_type == "optimized":
            from core.optimized_generation import OptimizedMusicGenerator
            return OptimizedMusicGenerator(**kwargs)
        else:
            from core.music_generator import MusicGenerator
            return MusicGenerator(**kwargs)


class CacheFactory:
    """Factory para sistemas de caché"""
    
    @staticmethod
    def create_cache(cache_type: str = "memory", **kwargs):
        """
        Crea un sistema de caché según el tipo
        
        Args:
            cache_type: Tipo de caché (memory, disk, distributed, smart)
            **kwargs: Argumentos adicionales
        
        Returns:
            Instancia del caché
        """
        if cache_type == "disk":
            from utils.cache_manager import MusicCache
            return MusicCache(**kwargs)
        elif cache_type == "distributed":
            from utils.distributed_cache import DistributedCache
            return DistributedCache(**kwargs)
        elif cache_type == "smart":
            from utils.smart_cache import SmartCache
            return SmartCache(**kwargs)
        else:
            from utils.cache_manager import MusicCache
            return MusicCache(**kwargs)


class StorageFactory:
    """Factory para backends de almacenamiento"""
    
    @staticmethod
    def create_storage(storage_type: str = "local", **kwargs):
        """
        Crea un backend de almacenamiento
        
        Args:
            storage_type: Tipo de storage (local, s3, gcs, azure)
            **kwargs: Argumentos adicionales
        
        Returns:
            Instancia del storage
        """
        if storage_type == "s3":
            # En producción, implementar S3Storage
            from core.storage.local_storage import LocalStorage
            logger.warning("S3 storage not implemented, using local")
            return LocalStorage(**kwargs)
        elif storage_type == "gcs":
            # En producción, implementar GCSStorage
            from core.storage.local_storage import LocalStorage
            logger.warning("GCS storage not implemented, using local")
            return LocalStorage(**kwargs)
        else:
            from core.storage.local_storage import LocalStorage
            return LocalStorage(**kwargs)


class NotificationFactory:
    """Factory para servicios de notificación"""
    
    @staticmethod
    def create_notifier(notifier_type: str = "websocket", **kwargs):
        """
        Crea un servicio de notificación
        
        Args:
            notifier_type: Tipo (websocket, email, push, sms, webhook)
            **kwargs: Argumentos adicionales
        
        Returns:
            Instancia del notificador
        """
        if notifier_type == "email":
            from services.notification_service_advanced import AdvancedNotificationService
            return AdvancedNotificationService(**kwargs)
        else:
            from services.notification_service_advanced import AdvancedNotificationService
            return AdvancedNotificationService(**kwargs)

