"""
Music Generation Use Case
Caso de uso para generación de música
"""

import logging
from typing import Dict, Any
from modules.music_generation import MusicGenerationModule

logger = logging.getLogger(__name__)


class MusicGenerationUseCase:
    """
    Caso de uso para generación de música
    Orquesta la lógica de negocio
    """
    
    def __init__(self, music_module: MusicGenerationModule):
        self.music_module = music_module
    
    async def generate_music(
        self,
        prompt: str,
        duration: int = 30,
        user_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Genera música
        
        Args:
            prompt: Descripción de la música
            duration: Duración en segundos
            user_id: ID del usuario
            **kwargs: Parámetros adicionales
            
        Returns:
            Resultado de la generación
        """
        try:
            # Validar entrada
            if not prompt or len(prompt) < 3:
                raise ValueError("Prompt must be at least 3 characters")
            
            if duration < 5 or duration > 300:
                raise ValueError("Duration must be between 5 and 300 seconds")
            
            # Generar música usando el módulo
            result = await self.music_module.generate(
                prompt=prompt,
                duration=duration,
                **kwargs
            )
            
            # Agregar metadatos
            result['user_id'] = user_id
            result['prompt'] = prompt
            result['duration'] = duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error in music generation use case: {e}", exc_info=True)
            raise















