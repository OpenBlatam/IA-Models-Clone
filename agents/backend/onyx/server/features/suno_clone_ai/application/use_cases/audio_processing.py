"""
Audio Processing Use Case
Caso de uso para procesamiento de audio
"""

import logging
from typing import Dict, Any
from modules.audio_processing import AudioProcessingModule

logger = logging.getLogger(__name__)


class AudioProcessingUseCase:
    """
    Caso de uso para procesamiento de audio
    Orquesta la lógica de negocio
    """
    
    def __init__(self, audio_module: AudioProcessingModule):
        self.audio_module = audio_module
    
    async def process_audio_file(
        self,
        audio_path: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Procesa un archivo de audio
        
        Args:
            audio_path: Ruta del archivo
            options: Opciones de procesamiento
            
        Returns:
            Resultado del procesamiento
        """
        try:
            result = await self.audio_module.process_audio(
                audio_path,
                **(options or {})
            )
            return result
        except Exception as e:
            logger.error(f"Error in audio processing use case: {e}", exc_info=True)
            raise
    
    async def transcribe_audio(
        self,
        audio_path: str
    ) -> Dict[str, Any]:
        """
        Transcribe audio a texto
        
        Args:
            audio_path: Ruta del archivo
            
        Returns:
            Transcripción
        """
        try:
            result = await self.audio_module.transcribe(audio_path)
            return result
        except Exception as e:
            logger.error(f"Error in transcription use case: {e}", exc_info=True)
            raise















