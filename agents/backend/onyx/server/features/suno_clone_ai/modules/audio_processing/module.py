"""
Audio Processing Module Implementation
"""

import logging
from typing import Dict, Any
from modules.base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)


class AudioProcessingModule(BaseModule):
    """
    Módulo de procesamiento de audio
    Puede funcionar como microservicio independiente
    """
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self._processor = None
        self._transcriber = None
    
    async def _initialize(self):
        """Inicializa el procesador de audio"""
        try:
            from core.audio_processor import AudioProcessor
            from services.audio_transcription import AudioTranscriptionService
            
            self._processor = AudioProcessor()
            self._transcriber = AudioTranscriptionService()
            
            logger.info("Audio processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {e}")
            raise
    
    async def _shutdown(self):
        """Cierra el procesador"""
        self._processor = None
        self._transcriber = None
        logger.info("Audio processor shut down")
    
    async def process_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Procesa un archivo de audio
        
        Args:
            audio_path: Ruta del archivo de audio
            **kwargs: Opciones de procesamiento
            
        Returns:
            Resultado del procesamiento
        """
        if self.status != ModuleStatus.ACTIVE:
            raise RuntimeError(f"Module {self.name} is not active")
        
        try:
            result = await self._processor.process(audio_path, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            raise
    
    async def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio a texto
        
        Args:
            audio_path: Ruta del archivo de audio
            
        Returns:
            Transcripción
        """
        if self.status != ModuleStatus.ACTIVE:
            raise RuntimeError(f"Module {self.name} is not active")
        
        try:
            result = await self._transcriber.transcribe(audio_path)
            return result
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}", exc_info=True)
            raise















