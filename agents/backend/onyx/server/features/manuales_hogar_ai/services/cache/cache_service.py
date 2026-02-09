"""
Cache Service
=============

Servicio principal para gestión de cache persistente.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.base.service_base import BaseService
from ...database.models import ManualCache
from ...config.settings import get_settings
from .cache_repository import CacheRepository
from .cache_key_generator import CacheKeyGenerator


class CacheService(BaseService):
    """Servicio para gestionar cache persistente."""
    
    def __init__(self, db: AsyncSession, ttl_hours: int = 24):
        """
        Inicializar servicio de cache.
        
        Args:
            db: Sesión de base de datos
            ttl_hours: Tiempo de vida en horas (default: 24)
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.ttl_hours = ttl_hours
        self.settings = get_settings()
        self.repository = CacheRepository(db)
        self.key_generator = CacheKeyGenerator()
    
    async def get(
        self,
        problem_description: str,
        category: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Obtener valor del cache.
        
        Args:
            problem_description: Descripción del problema
            category: Categoría
            **kwargs: Otros parámetros
        
        Returns:
            Valor cacheado o None
        """
        try:
            cache_key = self.key_generator.generate(problem_description, category, **kwargs)
            cache_entry = await self.repository.get_by_key(cache_key)
            
            if cache_entry:
                await self.repository.update_hit(cache_entry)
                self.log_debug(f"Cache hit: {cache_key[:8]}...")
                
                return {
                    "manual": cache_entry.manual_content,
                    "category": cache_entry.category,
                    "model_used": cache_entry.model_used,
                    "tokens_used": cache_entry.tokens_used,
                    "cached": True
                }
            
            return None
        
        except Exception as e:
            self.log_error(f"Error obteniendo cache: {str(e)}")
            return None
    
    async def set(
        self,
        problem_description: str,
        category: str,
        manual_content: str,
        model_used: Optional[str] = None,
        tokens_used: int = 0,
        **kwargs
    ) -> bool:
        """
        Guardar valor en cache.
        
        Args:
            problem_description: Descripción del problema
            category: Categoría
            manual_content: Contenido del manual
            model_used: Modelo usado
            tokens_used: Tokens utilizados
            **kwargs: Otros parámetros
        
        Returns:
            True si se guardó exitosamente
        """
        try:
            cache_key = self.key_generator.generate(problem_description, category, **kwargs)
            description_hash = self.key_generator.generate_description_hash(problem_description)
            expires_at = datetime.now() + timedelta(hours=self.ttl_hours)
            
            existing = await self.repository.get_by_key(cache_key)
            
            if existing:
                existing.manual_content = manual_content
                existing.model_used = model_used
                existing.tokens_used = tokens_used
                existing.expires_at = expires_at
                existing.last_accessed = datetime.now()
                await self.repository.save(existing)
            else:
                cache_entry = ManualCache(
                    cache_key=cache_key,
                    problem_description_hash=description_hash,
                    category=category,
                    manual_content=manual_content,
                    model_used=model_used,
                    tokens_used=tokens_used,
                    expires_at=expires_at
                )
                await self.repository.save(cache_entry)
            
            self.log_debug(f"Cache set: {cache_key[:8]}...")
            return True
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error guardando cache: {str(e)}")
            return False
    
    async def clear_expired(self) -> int:
        """
        Limpiar entradas expiradas.
        
        Returns:
            Número de entradas eliminadas
        """
        deleted_count = await self.repository.delete_expired()
        self.log_info(f"Eliminadas {deleted_count} entradas expiradas del cache")
        return deleted_count
    
    async def clear_all(self) -> int:
        """
        Limpiar todo el cache.
        
        Returns:
            Número de entradas eliminadas
        """
        deleted_count = await self.repository.delete_all()
        self.log_info(f"Cache limpiado: {deleted_count} entradas eliminadas")
        return deleted_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del cache.
        
        Returns:
            Diccionario con estadísticas
        """
        return await self.repository.get_stats()

