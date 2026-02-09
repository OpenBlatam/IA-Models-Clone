"""
Music Generation Module Implementation
"""

import logging
from typing import Dict, Any, Optional
from modules.base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)


class MusicGenerationModule(BaseModule):
    """
    Módulo de generación de música
    Puede funcionar como microservicio independiente
    """
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self._generator = None
        self._cache_manager = None
    
    async def _initialize(self):
        """Inicializa el generador de música"""
        try:
            # Lazy import para reducir cold starts
            from core.music_generator import get_music_generator
            from core.cache_manager import get_cache_manager
            
            self._generator = get_music_generator()
            self._cache_manager = get_cache_manager()
            
            logger.info("Music generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize music generator: {e}")
            raise
    
    async def _shutdown(self):
        """Cierra el generador"""
        if self._generator:
            # Cleanup si es necesario
            self._generator = None
        logger.info("Music generator shut down")
    
    async def generate(
        self,
        prompt: str,
        duration: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Genera música
        
        Args:
            prompt: Descripción de la música
            duration: Duración en segundos
            **kwargs: Parámetros adicionales
            
        Returns:
            Resultado de la generación
        """
        if self.status != ModuleStatus.ACTIVE:
            raise RuntimeError(f"Module {self.name} is not active")
        
        try:
            # Verificar caché
            cache_key = f"music:{prompt}:{duration}"
            if self._cache_manager:
                cached = await self._cache_manager.get(cache_key)
                if cached:
                    logger.info(f"Cache hit for {cache_key}")
                    return cached
            
            # Generar música
            result = await self._generator.generate(
                prompt=prompt,
                duration=duration,
                **kwargs
            )
            
            # Guardar en caché
            if self._cache_manager:
                await self._cache_manager.set(cache_key, result, ttl=3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating music: {e}", exc_info=True)
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del módulo"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "generator_initialized": self._generator is not None,
            "cache_enabled": self._cache_manager is not None
        })
        return base_metrics















