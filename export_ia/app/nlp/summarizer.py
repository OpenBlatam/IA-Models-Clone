"""
Text Summarizer - Resumidor de texto
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import SummarizationResult

logger = logging.getLogger(__name__)


class TextSummarizer:
    """Resumidor de texto."""
    
    def __init__(self):
        self._initialized = False
    
    async def initialize(self):
        """Inicializar el resumidor."""
        self._initialized = True
        logger.info("Text Summarizer inicializado")
    
    async def shutdown(self):
        """Cerrar el resumidor."""
        self._initialized = False
        logger.info("Text Summarizer cerrado")
    
    async def summarize(self, text: str, max_sentences: int = 3) -> SummarizationResult:
        """Resumir texto."""
        if not self._initialized:
            await self.initialize()
        
        # Dividir en oraciones
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Seleccionar las primeras oraciones como resumen
        summary_sentences = sentences[:max_sentences]
        summary = '. '.join(summary_sentences) + '.'
        
        # Calcular métricas
        word_count_original = len(text.split())
        word_count_summary = len(summary.split())
        compression_ratio = word_count_summary / word_count_original if word_count_original > 0 else 0
        
        # Extraer puntos clave (primeras palabras de cada oración)
        key_points = [s.split()[0] + "..." if len(s.split()) > 1 else s for s in summary_sentences]
        
        return SummarizationResult(
            original_text=text,
            summary=summary,
            compression_ratio=compression_ratio,
            key_points=key_points,
            word_count_original=word_count_original,
            word_count_summary=word_count_summary
        )
    
    async def health_check(self) -> bool:
        """Verificar salud del resumidor."""
        return self._initialized




